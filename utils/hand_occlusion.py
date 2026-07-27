from pathlib import Path
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np


# Topologia e parametri derivati dal debugger MediaPipe allegato.
HAND_LANDMARK_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
HAND_PALM_INDICES = (0, 1, 2, 5, 9, 13, 17)
HAND_FINGER_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
HAND_MASK_PADDING_RATIO = 0.060
HAND_MASK_PADDING_MIN = 8
HAND_MASK_PADDING_MAX = 64
HAND_PALM_RADIUS_RATIO = 1.5
HAND_WRIST_HALF_WIDTH_RATIO = 0.55
HAND_WRIST_EXTENSION_RATIO = 0.12
HAND_WRIST_BORDER_RATIO = 0.20
HAND_THUMB_WEB_CURVE_FACTOR = 0.25
HAND_THUMB_RADIUS_FACTORS = (0.0, 0.18, 0.15, 0.11, 0.09)
HAND_THUMB_RADIUS_MAX_BASE_RATIO = 2.25
MEDIAPIPE_CONTEXT_MARGIN_RATIO = 0.10


class VideoHandLandmarker:
    """Adapter stateful per il Hand Landmarker MediaPipe in modalità VIDEO."""

    def __init__(self, landmarker):
        self._landmarker = landmarker
        self._last_timestamp_ms = -1

    def detect(self, image, timestamp_ms=None):
        if timestamp_ms is None:
            timestamp_ms = self._last_timestamp_ms + 1

        timestamp_ms = int(timestamp_ms)
        if timestamp_ms <= self._last_timestamp_ms:
            raise ValueError(
                'I timestamp MediaPipe VIDEO devono essere strettamente '
                'crescenti (ricevuto {}, precedente {}).'.format(
                    timestamp_ms, self._last_timestamp_ms
                )
            )

        result = self._landmarker.detect_for_video(image, timestamp_ms)
        self._last_timestamp_ms = timestamp_ms
        return result

    def close(self):
        self._landmarker.close()


def create_hand_landmarker(model_path, num_hands=2):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            'Modello MediaPipe Hand Landmarker non trovato: {}'.format(
                model_path
            )
        )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=int(num_hands),
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    return VideoHandLandmarker(landmarker)


def create_square_mediapipe_frame(image_rgb):
    """
    Inserisce il frame in un canvas quadrato con margine di contesto.

    Il margine riduce la scala relativa delle mani molto grandi o tagliate
    dal bordo, che altrimenti possono non produrre alcuna detection.
    """
    frame_height, frame_width = image_rgb.shape[:2]
    content_size = max(frame_width, frame_height)
    context_margin = int(round(
        content_size * MEDIAPIPE_CONTEXT_MARGIN_RATIO
    ))
    square_size = content_size + context_margin * 2
    pad_left = (content_size - frame_width) // 2 + context_margin
    pad_top = (content_size - frame_height) // 2 + context_margin
    square_frame = np.zeros(
        (square_size, square_size, 3), dtype=image_rgb.dtype
    )
    square_frame[
        pad_top:pad_top + frame_height,
        pad_left:pad_left + frame_width,
    ] = image_rgb
    return square_frame, square_size, pad_left, pad_top


def remap_square_landmarks_to_frame(
    hand_landmarker_result,
    frame_size,
    square_size,
    pad_left,
    pad_top,
):
    """Riporta sul frame originale i landmark rilevati nel canvas quadrato."""
    frame_height, frame_width = frame_size
    remapped_hands = []

    for landmarks in getattr(hand_landmarker_result, 'hand_landmarks', []):
        remapped_landmarks = []
        for landmark in landmarks:
            x = landmark.x * square_size - pad_left
            y = landmark.y * square_size - pad_top
            remapped_landmarks.append(
                SimpleNamespace(
                    x=float(np.clip(x / max(frame_width, 1), 0.0, 1.0)),
                    y=float(np.clip(y / max(frame_height, 1), 0.0, 1.0)),
                    z=getattr(landmark, 'z', 0.0),
                    visibility=getattr(landmark, 'visibility', 0.0),
                    presence=getattr(landmark, 'presence', 0.0),
                )
            )

        remapped_hands.append(remapped_landmarks)

    return SimpleNamespace(
        handedness=getattr(hand_landmarker_result, 'handedness', []),
        hand_landmarks=remapped_hands,
        hand_world_landmarks=getattr(
            hand_landmarker_result, 'hand_world_landmarks', []
        ),
    )


def detect_hand_landmarks(image_rgb, landmarker, timestamp_ms=None):
    """Rileva le mani senza distorcere frame non quadrati."""
    image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    square_frame, square_size, pad_left, pad_top = (
        create_square_mediapipe_frame(image_rgb)
    )
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=square_frame,
    )
    result = landmarker.detect(mp_image, timestamp_ms=timestamp_ms)
    return remap_square_landmarks_to_frame(
        result,
        image_rgb.shape[:2],
        square_size,
        pad_left,
        pad_top,
    )


def get_hand_landmark_points(landmarks, frame_size):
    frame_height, frame_width = frame_size
    return [
        [
            int(np.clip(landmark.x * frame_width, 0, frame_width - 1)),
            int(np.clip(landmark.y * frame_height, 0, frame_height - 1)),
        ]
        for landmark in landmarks
    ]


def calculate_hand_mask_padding(
    points,
    padding_ratio=HAND_MASK_PADDING_RATIO,
    padding_min=HAND_MASK_PADDING_MIN,
    padding_max=HAND_MASK_PADDING_MAX,
):
    _, _, width, height = cv2.boundingRect(
        np.asarray(points, dtype=np.int32)
    )
    hand_size = max(width, height)
    padding = int(round(hand_size * float(padding_ratio)))
    return int(np.clip(padding, int(padding_min), int(padding_max)))


def create_augmented_palm_points(points, frame_size):
    """
    Aggiunge due estremi sintetici del polso al palm hull.

    MediaPipe fornisce un solo landmark centrale per il polso (indice 0).
    I punti aggiuntivi servono soltanto quando il polso è vicino a un bordo:
    applicarli a una mano interamente visibile può creare un triangolo troppo
    ampio tra palmo e avambraccio.
    """
    points_array = np.asarray(points, dtype=np.float32)
    wrist = points_array[0]
    index_base = points_array[5]
    little_base = points_array[17]

    across_palm = index_base - little_base
    palm_width = float(np.linalg.norm(across_palm))
    if palm_width < 1.0:
        return np.asarray(
            [points[index] for index in HAND_PALM_INDICES],
            dtype=np.int32,
        )

    frame_height, frame_width = frame_size
    wrist_edge_distance = min(
        wrist[0],
        wrist[1],
        frame_width - 1 - wrist[0],
        frame_height - 1 - wrist[1],
    )
    if wrist_edge_distance > palm_width * HAND_WRIST_BORDER_RATIO:
        return np.asarray(
            [points[index] for index in HAND_PALM_INDICES],
            dtype=np.int32,
        )

    across_direction = across_palm / palm_width
    palm_base_center = (index_base + little_base) * 0.5
    wrist_direction = wrist - palm_base_center
    wrist_direction_norm = float(np.linalg.norm(wrist_direction))
    if wrist_direction_norm >= 1.0:
        wrist_direction /= wrist_direction_norm
    else:
        # La perpendicolare mantiene una geometria valida anche in pose
        # degeneri nelle quali il polso coincide con il centro del palmo.
        wrist_direction = np.asarray(
            [-across_direction[1], across_direction[0]],
            dtype=np.float32,
        )

    extended_wrist = (
        wrist
        + wrist_direction * palm_width * HAND_WRIST_EXTENSION_RATIO
    )
    wrist_half_width = palm_width * HAND_WRIST_HALF_WIDTH_RATIO
    wrist_corner_a = extended_wrist + across_direction * wrist_half_width
    wrist_corner_b = extended_wrist - across_direction * wrist_half_width

    palm_points = [
        points_array[index] for index in HAND_PALM_INDICES
    ]
    palm_points.extend((wrist_corner_a, wrist_corner_b))
    return np.rint(palm_points).astype(np.int32)


def create_thumb_web_curve(points, sample_count=7):
    """Crea il raccordo concavo fra la base del pollice e dell'indice."""
    points_array = np.asarray(points, dtype=np.float32)
    thumb_base = points_array[2]
    index_base = points_array[5]
    palm_center = np.mean(
        points_array[[0, 5, 9, 13, 17]],
        axis=0,
    )
    web_center = (thumb_base + index_base) * 0.5
    control = web_center + (
        palm_center - web_center
    ) * HAND_THUMB_WEB_CURVE_FACTOR

    curve = []
    for t in np.linspace(0.0, 1.0, sample_count):
        point = (
            ((1.0 - t) ** 2) * thumb_base
            + 2.0 * (1.0 - t) * t * control
            + (t ** 2) * index_base
        )
        curve.append(point)
    return np.rint(curve).astype(np.int32)


def create_anatomical_palm_contour(points, frame_size):
    """Ordina il bordo del palmo preservando la concavità pollice-indice."""
    points_array = np.asarray(points, dtype=np.int32)
    augmented_points = create_augmented_palm_points(points, frame_size)
    thumb_web_curve = create_thumb_web_curve(points)

    if len(augmented_points) > len(HAND_PALM_INDICES):
        wrist_corners = augmented_points[-2:]
        # Il vertice più vicino alla base del pollice appartiene al lato
        # radiale; l'altro chiude il lato del mignolo.
        thumb_side_index = int(np.argmin(
            np.linalg.norm(
                wrist_corners.astype(np.float32) - points_array[1],
                axis=1,
            )
        ))
        thumb_wrist_corner = wrist_corners[thumb_side_index]
        little_wrist_corner = wrist_corners[1 - thumb_side_index]
        contour = [
            thumb_wrist_corner,
            points_array[1],
            points_array[2],
        ]
    else:
        little_wrist_corner = None
        contour = [
            points_array[0],
            points_array[1],
            points_array[2],
        ]

    contour.extend(thumb_web_curve[1:-1])
    contour.extend(
        points_array[index] for index in (5, 9, 13, 17)
    )
    if little_wrist_corner is not None:
        contour.append(little_wrist_corner)
    return np.asarray(contour, dtype=np.int32)


def draw_tapered_capsule(mask, start, end, start_radius, end_radius):
    """Disegna un segmento con spessore interpolato fra le due estremità."""
    start_point = np.asarray(start, dtype=np.float32)
    end_point = np.asarray(end, dtype=np.float32)
    direction = end_point - start_point
    length = float(np.linalg.norm(direction))

    if length < 1.0:
        cv2.circle(
            mask,
            tuple(np.rint(start_point).astype(np.int32)),
            max(int(start_radius), int(end_radius)),
            255,
            -1,
        )
        return

    normal = np.asarray(
        [-direction[1], direction[0]], dtype=np.float32
    ) / length
    polygon = np.asarray(
        [
            start_point + normal * start_radius,
            end_point + normal * end_radius,
            end_point - normal * end_radius,
            start_point - normal * start_radius,
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, polygon, 255)
    cv2.circle(
        mask,
        tuple(np.rint(start_point).astype(np.int32)),
        int(start_radius),
        255,
        -1,
    )
    cv2.circle(
        mask,
        tuple(np.rint(end_point).astype(np.int32)),
        int(end_radius),
        255,
        -1,
    )


def create_landmark_hand_segmentation_mask(
    points,
    frame_size,
    padding_ratio=HAND_MASK_PADDING_RATIO,
    padding_min=HAND_MASK_PADDING_MIN,
    padding_max=HAND_MASK_PADDING_MAX,
):
    """Costruisce la segmentazione anatomica da 21 landmark MediaPipe."""
    frame_height, frame_width = frame_size
    hand_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    points_array = np.asarray(points, dtype=np.int32)

    _, _, width, height = cv2.boundingRect(points_array)
    hand_size = max(width, height)
    base_radius = int(np.clip(hand_size * 0.045, 4, 22))
    palm_width = max(
        float(np.linalg.norm(points_array[5] - points_array[17])),
        1.0,
    )

    # Il palmo usa una dilatazione dedicata e due estremi sintetici del
    # polso. Le dita continuano a usare i raggi progressivi sottostanti,
    # evitando di gonfiare gli spazi interdigitali.
    palm_radius = int(base_radius * HAND_PALM_RADIUS_RATIO)
    palm_contour = create_anatomical_palm_contour(points, frame_size)
    cv2.fillPoly(hand_mask, [palm_contour], 255)
    hand_mask = cv2.dilate(
        hand_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (palm_radius * 2 + 1, palm_radius * 2 + 1),
        ),
        iterations=1,
    )

    thumb_radii = [int(base_radius * 1.25)]
    for factor in HAND_THUMB_RADIUS_FACTORS[1:]:
        thumb_radii.append(
            int(np.clip(
                palm_width * factor,
                base_radius * 0.75,
                base_radius * HAND_THUMB_RADIUS_MAX_BASE_RATIO,
            ))
        )

    for finger_index, chain in enumerate(HAND_FINGER_CHAINS):
        for chain_index in range(len(chain) - 1):
            start_index = chain[chain_index]
            end_index = chain[chain_index + 1]
            start = tuple(points[start_index])
            end = tuple(points[end_index])

            if finger_index == 0:
                draw_tapered_capsule(
                    hand_mask,
                    start,
                    end,
                    thumb_radii[chain_index],
                    thumb_radii[chain_index + 1],
                )
                continue

            radius = int(
                base_radius
                * np.interp(chain_index, [0, 3], [1.25, 0.75])
            )
            cv2.line(hand_mask, start, end, 255, max(radius * 2, 1))
            cv2.circle(hand_mask, start, radius, 255, -1)
            cv2.circle(hand_mask, end, radius, 255, -1)

    close_size = max(5, base_radius)
    if close_size % 2 == 0:
        close_size += 1
    hand_mask = cv2.morphologyEx(
        hand_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        ),
        iterations=1,
    )
    hand_mask = cv2.GaussianBlur(hand_mask, (5, 5), 0)
    hand_mask = cv2.threshold(
        hand_mask, 50, 255, cv2.THRESH_BINARY
    )[1]

    padding = max(
        0,
        calculate_hand_mask_padding(
            points_array,
            padding_ratio=padding_ratio,
            padding_min=padding_min,
            padding_max=padding_max,
        ) - base_radius,
    )
    if padding > 0:
        kernel_size = padding * 2 + 1
        hand_mask = cv2.dilate(
            hand_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            ),
            iterations=1,
        )

    return hand_mask


def create_hand_segmentation_mask(
    hand_landmarker_result,
    frame_size,
    padding_ratio=HAND_MASK_PADDING_RATIO,
    padding_min=HAND_MASK_PADDING_MIN,
    padding_max=HAND_MASK_PADDING_MAX,
):
    """Unisce le segmentation mask di tutte le mani rilevate."""
    frame_height, frame_width = frame_size
    hand_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    for landmarks in getattr(hand_landmarker_result, 'hand_landmarks', []):
        points = get_hand_landmark_points(landmarks, frame_size)
        if len(points) < 21:
            continue
        per_hand_mask = create_landmark_hand_segmentation_mask(
            points,
            frame_size,
            padding_ratio=padding_ratio,
            padding_min=padding_min,
            padding_max=padding_max,
        )
        hand_mask = cv2.bitwise_or(hand_mask, per_hand_mask)

    return hand_mask


def detect_hand_mask(
    image_rgb,
    landmarker,
    dilation_size=None,
    timestamp_ms=None,
):
    """
    Ritorna una segmentation mask float32 [H, W, 1], con valori 0/1.

    ``dilation_size`` è mantenuto per compatibilità: se specificato applica
    una dilatazione aggiuntiva alla segmentazione anatomica.
    """
    result = detect_hand_landmarks(
        image_rgb, landmarker, timestamp_ms=timestamp_ms
    )
    mask = create_hand_segmentation_mask(result, image_rgb.shape[:2])

    if dilation_size is not None and dilation_size > 0:
        kernel_size = int(dilation_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            ),
            iterations=1,
        )

    return (mask > 0).astype(np.float32)[..., np.newaxis]


def draw_hand_landmarks(
    frame_bgr,
    hand_landmarker_result,
    point_color=(0, 255, 255),
    connection_color=(0, 180, 255),
):
    """Disegna punti e connessioni dei landmark sul frame BGR."""
    output = np.ascontiguousarray(frame_bgr.copy())
    for landmarks in getattr(hand_landmarker_result, 'hand_landmarks', []):
        points = get_hand_landmark_points(landmarks, output.shape[:2])
        if len(points) < 21:
            continue
        for point in points:
            cv2.circle(output, tuple(point), 3, point_color, -1)
        for start_index, end_index in HAND_LANDMARK_CONNECTIONS:
            cv2.line(
                output,
                tuple(points[start_index]),
                tuple(points[end_index]),
                connection_color,
                2,
            )
    return output


def draw_hand_segmentation(
    frame_bgr,
    hand_landmarker_result=None,
    hand_mask=None,
    color=(255, 128, 0),
    thickness=2,
):
    """Disegna il contorno della segmentation mask sul frame BGR."""
    output = np.ascontiguousarray(frame_bgr.copy())
    if hand_mask is None:
        if hand_landmarker_result is None:
            return output
        hand_mask = create_hand_segmentation_mask(
            hand_landmarker_result, output.shape[:2]
        )
    if hand_mask.ndim == 3:
        hand_mask = hand_mask[..., 0]
    binary_mask = (hand_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(output, contours, -1, color, thickness)
    return output


def build_canonical_condition(
    source_rgb,
    source_affine_256,
    landmarker,
    timestamp_ms=None,
):
    """
    Costruisce la condizione SOA ``[mask, RGB mascherato]`` nel crop 256x256.
    """
    source_canon = cv2.warpAffine(
        source_rgb,
        source_affine_256,
        (256, 256),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_frame = detect_hand_mask(
        source_rgb,
        landmarker,
        timestamp_ms=timestamp_ms,
    )
    mask_canon = cv2.warpAffine(
        mask_frame[..., 0],
        source_affine_256,
        (256, 256),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_canon = (mask_canon >= 0.5).astype(np.float32)[..., np.newaxis]
    source_canon_normalized = (
        source_canon.astype(np.float32) - 127.5
    ) / 127.5
    masked_occluder = source_canon_normalized * mask_canon
    condition = np.concatenate(
        [mask_canon, masked_occluder], axis=-1
    ).astype(np.float32)
    return condition, mask_canon, source_canon
