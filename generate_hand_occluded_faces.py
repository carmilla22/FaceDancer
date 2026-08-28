from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# se top --> la mano è verticale con il polso in alto,
# se left --> la mano è orizzontale con il polso a sinistra, ecc.
ENTRY_ROTATION = {"top": 0.0, "bottom": 180.0, "left": 90.0, "right": -90.0}


@dataclass(frozen=True)
class HandPair:
    stem: str
    image_path: Path
    mask_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera immagini di volti occluse da mani, usando cartelle di immagini "
            "e maschere per un ritaglio preciso."
        )
    )
    parser.add_argument(
        "--hands-dir",
        type=Path,
        required=True,
        help="Cartella delle immagini delle mani (ricerca ricorsiva).",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        required=True,
        help="Cartella delle maschere PNG delle mani (ricerca ricorsiva).",
    )
    parser.add_argument(
        "--faces-dir",
        type=Path,
        required=True,
        help="Cartella delle immagini dei volti (ricerca ricorsiva).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--num-faces",
        type=int,
        default=1,
        help="Numero di volti da elaborare; 0 significa tutti i volti (default: 1).",
    )
    parser.add_argument(
        "--variants-per-face",
        type=int,
        default=1,
        help="Numero di mani/augmentation differenti per ogni volto (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-hand-size",
        type=float,
        default=0.65,
        help="Dimensione minima del lato lungo della mano rispetto al volto.",
    )
    parser.add_argument(
        "--max-hand-size",
        type=float,
        default=0.90,
        help="Dimensione massima del lato lungo della mano rispetto al volto.",
    )
    parser.add_argument(
        "--max-angle-jitter",
        type=float,
        default=18.0,
        help="Rotazione casuale massima, in gradi, dopo l'orientamento della mano.",
    )
    parser.add_argument(
        "--edge-feather",
        type=float,
        default=1.2,
        help="Sfocatura in pixel del solo bordo usato per il blending.",
    )
    parser.add_argument(
        "--sides",
        nargs="+",
        choices=tuple(ENTRY_ROTATION),
        default=["left", "right", "bottom", "top"],
        help="Bordi dai quali puo entrare il polso.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permette di sovrascrivere immagini gia presenti nell'output.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for label, path in (
        ("Cartella mani", args.hands_dir),
        ("Cartella maschere", args.masks_dir),
        ("Cartella volti", args.faces_dir),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} non trovata: {path}")

    resolved_output = args.output.resolve()
    for option, input_dir in (
        ("--hands-dir", args.hands_dir),
        ("--masks-dir", args.masks_dir),
        ("--faces-dir", args.faces_dir),
    ):
        if resolved_output.is_relative_to(input_dir.resolve()):
            raise ValueError(f"--output non puo essere dentro {option}")

    if args.num_faces < 0:
        raise ValueError("--num-faces deve essere >= 0")
    if args.variants_per_face < 1:
        raise ValueError("--variants-per-face deve essere >= 1")
    if not 0.1 <= args.min_hand_size <= args.max_hand_size <= 1.5:
        raise ValueError("Intervallo --min-hand-size/--max-hand-size non valido")
    if args.edge_feather < 0:
        raise ValueError("--edge-feather deve essere >= 0")


def discover_images(root: Path, extensions: set[str]) -> list[Path]:
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(
        paths,
        key=lambda path: (
            path.relative_to(root).as_posix().lower(),
            path.relative_to(root).as_posix(),
        ),
    )


def numeric_name_key(path: Path) -> tuple[int, int | str]:
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def build_hand_pairs(hand_paths: list[Path], mask_paths: list[Path]) -> list[HandPair]:
    images_by_stem: dict[str, Path] = {}
    for image_path in hand_paths:
        key = image_path.stem.lower()
        if key in images_by_stem:
            raise RuntimeError(
                f"Immagini mano duplicate per il nome '{image_path.stem}': "
                f"{images_by_stem[key]} e {image_path}"
            )
        images_by_stem[key] = image_path

    pairs = []
    paired_masks: dict[str, Path] = {}
    for mask_path in mask_paths:
        stem = mask_path.stem
        key = stem.lower()
        image_path = images_by_stem.get(key)
        if image_path:
            if key in paired_masks:
                raise RuntimeError(
                    f"Maschere duplicate per il nome '{stem}': "
                    f"{paired_masks[key]} e {mask_path}"
                )
            paired_masks[key] = mask_path
            pairs.append(HandPair(stem, image_path, mask_path))
    if not pairs:
        raise RuntimeError("Nessuna coppia mano/maschera con lo stesso nome")
    return pairs


def open_image(path: Path, mode: str) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert(mode)


def binary_mask(mask: Image.Image) -> Image.Image:
    gray = mask.convert("L")  # conversione dell'immagine in gray scale
    maximum = gray.getextrema()[
        1
    ]  # prene il max per capire se l'immagine è già binaria o meno (se è tra 0 e 1 è già binaria, se è tra 0 e 255 no)
    threshold = 0 if maximum <= 1 else 127
    return gray.point(
        lambda value: 255 if value > threshold else 0, mode="L"
    )  # binarizzazione


# data l'immagine della mano con lo sfondo e la maschera binaria, ritaglia l'immagine della mano e la maschera
# in modo che contengano solo la mano con un padding di 3 pixel
def crop_to_mask(
    image: Image.Image, mask: Image.Image, padding: int = 3
) -> tuple[Image.Image, Image.Image]:
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("Maschera vuota")
    left = max(
        0, bbox[0] - padding
    )  # aggiunta di 3 pixel di padding per evitare di tagliare fuori pixel della mano
    top = max(0, bbox[1] - padding)
    right = min(mask.width, bbox[2] + padding)
    bottom = min(mask.height, bbox[3] + padding)
    box = (left, top, right, bottom)
    return image.crop(box), mask.crop(
        box
    )  # sia l'immagine della mano rgb sia la maschera binaria vengono ritagliate nello stesso modo


# dedurre dove si trova il polso della mano in base ai valori della maschera binaria
# crea come 4 fasce di 4% della dimensione dell'immagine e calcola la media dei pixel della maschera in quelle fasce
# il valore medio della fascia di bordo con valore più alto indica dove si trova il polso della mano
def edge_foreground_score(mask: Image.Image, side: str) -> float:
    band_x = max(2, round(mask.width * 0.04))
    band_y = max(2, round(mask.height * 0.04))
    boxes = {
        "top": (0, 0, mask.width, band_y),
        "bottom": (0, mask.height - band_y, mask.width, mask.height),
        "left": (0, 0, band_x, mask.height),
        "right": (mask.width - band_x, 0, mask.width, mask.height),
    }
    return ImageStat.Stat(mask.crop(boxes[side])).mean[0]


# dopo aver dedotto dove si trova il polso della mano, ruota l'immagine della mano e la maschera binaria
# in modo che il polso sia sempre in alto --> normalizziamo i dati così da avere un dataset più coerenti
def normalize_wrist_to_top(
    image: Image.Image, mask: Image.Image
) -> tuple[Image.Image, Image.Image, str]:
    # individuazione del lato dell'immagine in cui si trova il polso della mano
    scores = {side: edge_foreground_score(mask, side) for side in ENTRY_ROTATION}
    wrist_side = max(scores, key=scores.get)
    # tabella che dice in base alla posizione del polso, di quanto deve ruotare l'immagine per avere il polso nella posizione top
    rotation_to_top = {"top": 0.0, "bottom": 180.0, "left": -90.0, "right": 90.0}
    angle = rotation_to_top[wrist_side]
    if angle:
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)
        # dopo aver ruotato l'immagine PIL crea spazi vuoto intorno all'immagine, quindi sia la hand image rgb viene tagliata
        # sia la maschera binaria viene tagliata in modo da contenere solo la mano.
        image, mask = crop_to_mask(image, binary_mask(mask))
    return image, mask, wrist_side


# modifiche della mano per rendere il dataset più vario e robusto, modificando luminosità, contrasto e saturazione
def color_augment(
    image: Image.Image, rng: random.Random
) -> tuple[Image.Image, float, float, float]:
    brightness = rng.uniform(0.88, 1.12)
    contrast = rng.uniform(0.92, 1.08)
    saturation = rng.uniform(0.90, 1.10)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)
    return image, brightness, contrast, saturation


# prepara l'immagine della mano e la maschera binaria per essere sovrapposte al volto
def prepare_hand(
    hand: Image.Image,
    raw_mask: Image.Image,
    face_size: tuple[int, int],
    rng: random.Random,
    min_hand_size: float,
    max_hand_size: float,
    max_angle_jitter: float,
    side: str,
) -> tuple[Image.Image, Image.Image, dict[str, object]]:
    # 1. ridimensionamento della maschera binaria se necessario
    if hand.size != raw_mask.size:
        raw_mask = raw_mask.resize(hand.size, Image.Resampling.NEAREST)
    mask = binary_mask(raw_mask)

    # 2. ritaglio dell'immagine della mano e della maschera binaria
    hand, mask = crop_to_mask(hand, mask)
    # 3. normalizzazione della mano in modo che il polso sia sempre in alto
    hand, mask, source_wrist_side = normalize_wrist_to_top(hand, mask)

    # 4. eventuale flip orizzontale della mano e della maschera binaria
    flipped = rng.random() < 0.5
    if flipped:
        hand = ImageOps.mirror(hand)
        mask = ImageOps.mirror(mask)
    # 5. modifica dei colori della mano per aumentare la varietà del dataset
    hand, brightness, contrast, saturation = color_augment(hand, rng)

    # 6. ridimensionamento della mano e della maschera binaria in base alla dimensione del volto e ai parametri min_hand_size e max_hand_size
    target_fraction = rng.uniform(min_hand_size, max_hand_size)
    target_long_side = max(1, round(min(face_size) * target_fraction))
    resize_factor = target_long_side / max(hand.size)
    resized = (
        max(1, round(hand.width * resize_factor)),
        max(1, round(hand.height * resize_factor)),
    )
    hand = hand.resize(resized, Image.Resampling.LANCZOS)
    mask = mask.resize(resized, Image.Resampling.NEAREST)

    # 7. rotazione della mano e della maschera binaria in base al lato di ingresso e a un jitter casuale
    angle_jitter = rng.uniform(-max_angle_jitter, max_angle_jitter)
    final_angle = ENTRY_ROTATION[side] + angle_jitter
    hand = hand.rotate(final_angle, resample=Image.Resampling.BICUBIC, expand=True)
    mask = mask.rotate(final_angle, resample=Image.Resampling.NEAREST, expand=True)

    # 8. ritaglio finale della mano e della maschera binaria per rimuovere eventuali spazi vuoti creati dalla rotazione
    mask = binary_mask(mask)
    hand, mask = crop_to_mask(hand, mask)

    metadata = {
        "entry_side": side,
        "source_wrist_side": source_wrist_side,
        "angle_degrees": round(final_angle, 3),
        "size_fraction": round(target_fraction, 4),
        "horizontal_flip": flipped,
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "saturation": round(saturation, 4),
    }
    return hand, mask, metadata


# viene deciso dove posizionare la mano sul volto, prima si lavorava solo sulla mano, ora pure con il volto
# per poter applicare la mano sul volto
def placement_for_side(
    face_size: tuple[int, int],
    hand_size: tuple[int, int],
    side: str,
    rng: random.Random,
) -> tuple[int, int]:
    face_w, face_h = face_size
    hand_w, hand_h = hand_size
    outside_x = round(face_w * 0.025)
    outside_y = round(face_h * 0.025)

    if side in {"top", "bottom"}:
        center_x = round(rng.uniform(0.38, 0.62) * face_w)
        left = center_x - hand_w // 2
        top = -outside_y if side == "top" else face_h - hand_h + outside_y
    else:
        center_y = round(rng.uniform(0.36, 0.68) * face_h)
        top = center_y - hand_h // 2
        left = -outside_x if side == "left" else face_w - hand_w + outside_x
    return left, top


# composite_hand prende il volto, la mano, la maschera binaria e la posizione in cui posizionare la mano sul volto
# e restituisce l'immagine risultante e la maschera binaria posizionata
def composite_hand(
    face: Image.Image,
    hand: Image.Image,
    mask: Image.Image,
    position: tuple[int, int],
    edge_feather: float,
) -> tuple[Image.Image, Image.Image]:
    # Un pixel di erosione elimina il tipico alone chiaro dello sfondo originale.
    hard_mask = mask.filter(ImageFilter.MinFilter(3))
    hard_mask = binary_mask(hard_mask)
    alpha = hard_mask
    if edge_feather > 0:
        alpha = hard_mask.filter(ImageFilter.GaussianBlur(edge_feather))

    result = face.copy()
    result.paste(hand, position, alpha)

    placed_mask = Image.new("L", face.size, 0)
    placed_mask.paste(hard_mask, position)
    return result, placed_mask


def ensure_output_is_safe(output: Path, overwrite: bool) -> tuple[Path, Path]:
    images_dir = output / "images"
    masks_dir = output / "hand_masks"
    existing = []
    if images_dir.exists():
        existing.extend(images_dir.glob("*.jpg"))
    if masks_dir.exists():
        existing.extend(masks_dir.glob("*.png"))
    metadata_path = output / "metadata.csv"
    if metadata_path.exists():
        existing.append(metadata_path)
    if existing and not overwrite:
        raise FileExistsError(
            f"L'output contiene gia {len(existing)} file. Usa una nuova cartella oppure --overwrite."
        )
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, masks_dir


def ensure_unique_face_stems(face_paths: list[Path]) -> None:
    paths_by_stem: dict[str, Path] = {}
    for face_path in face_paths:
        key = face_path.stem.lower()
        if key in paths_by_stem:
            raise RuntimeError(
                f"Volti con nome duplicato '{face_path.stem}' produrrebbero lo stesso "
                f"output: {paths_by_stem[key]} e {face_path}"
            )
        paths_by_stem[key] = face_path


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> int:
    args = parse_args()
    validate_args(args)
    rng = random.Random(args.seed)

    images_dir, masks_dir = ensure_output_is_safe(args.output, args.overwrite)
    hand_paths = discover_images(args.hands_dir, IMAGE_EXTENSIONS)
    mask_paths = discover_images(args.masks_dir, {".png"})
    face_paths = discover_images(args.faces_dir, IMAGE_EXTENSIONS)
    face_paths.sort(key=numeric_name_key)
    pairs = build_hand_pairs(hand_paths, mask_paths)

    selected_faces = (
        face_paths if args.num_faces == 0 else face_paths[: args.num_faces]
    )
    if not selected_faces:
        raise RuntimeError(f"Nessuna immagine del volto trovata in {args.faces_dir}")
    ensure_unique_face_stems(selected_faces)

    print(
        f"Trovati {len(face_paths)} volti, {len(hand_paths)} immagini di mani, "
        f"{len(mask_paths)} maschere e {len(pairs)} coppie valide."
    )

    metadata_path = args.output / "metadata.csv"
    fieldnames = [
        "output_image",
        "output_mask",
        "face_path",
        "hand_path",
        "mask_path",
        "variant",
        "seed",
        "entry_side",
        "source_wrist_side",
        "angle_degrees",
        "size_fraction",
        "horizontal_flip",
        "brightness",
        "contrast",
        "saturation",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        total = len(selected_faces) * args.variants_per_face
        completed = 0
        for face_path in selected_faces:
            face = open_image(face_path, "RGB")
            face_stem = face_path.stem

            for variant in range(args.variants_per_face):
                pair = rng.choice(pairs)
                hand = open_image(pair.image_path, "RGB")
                with Image.open(pair.mask_path) as opened_mask:
                    raw_mask = opened_mask.copy()

                side = rng.choice(args.sides)
                hand, mask, augmentation = prepare_hand(
                    hand,
                    raw_mask,
                    face.size,
                    rng,
                    args.min_hand_size,
                    args.max_hand_size,
                    args.max_angle_jitter,
                    side,
                )
                position = placement_for_side(face.size, hand.size, side, rng)
                result, placed_mask = composite_hand(
                    face, hand, mask, position, args.edge_feather
                )

                output_stem = f"{face_stem}_v{variant + 1:02d}"
                image_path = images_dir / f"{output_stem}.jpg"
                output_mask_path = masks_dir / f"{output_stem}.png"
                result.save(image_path, quality=95, subsampling=0)
                placed_mask.save(output_mask_path)

                writer.writerow(
                    {
                        "output_image": image_path.name,
                        "output_mask": output_mask_path.name,
                        "face_path": relative_posix(face_path, args.faces_dir),
                        "hand_path": relative_posix(pair.image_path, args.hands_dir),
                        "mask_path": relative_posix(pair.mask_path, args.masks_dir),
                        "variant": variant + 1,
                        "seed": args.seed,
                        **augmentation,
                    }
                )
                completed += 1
                if completed == total or completed % 25 == 0:
                    print(f"Generate {completed}/{total} immagini")

    print(f"Dataset completato in: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        raise SystemExit(1)

