import argparse
from pathlib import Path

import cv2
import numpy as np

from utils.hand_occlusion import (
    create_hand_landmarker,
    create_hand_segmentation_mask,
    detect_hand_landmarks,
    draw_hand_landmarks,
    draw_hand_segmentation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Visualizza detection, landmark e segmentation mask MediaPipe '
            'su un’immagine o su tutti i frame di un video.'
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--vid_path', help='Percorso del video di input.')
    input_group.add_argument(
        '--img_path', help='Percorso dell’immagine di input.'
    )
    parser.add_argument(
        '--output',
        default=None,
        help=(
            'Percorso dell’output. Default: hand_mask_preview.mp4 per i '
            'video oppure hand_mask_preview.png per le immagini.'
        ),
    )
    parser.add_argument(
        '--hand_task_path',
        default='./models/hand_landmarker.task',
        help='Percorso del modello MediaPipe Hand Landmarker.',
    )
    parser.add_argument(
        '--view',
        choices=(
            'overlay',
            'mask',
            'landmarks',
            'segmentation',
            'side-by-side',
        ),
        default='side-by-side',
        help='Tipo di visualizzazione da salvare.',
    )
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.55,
        help='Opacità della mask verde nella vista overlay (0-1).',
    )
    parser.add_argument(
        '--dilation_size',
        type=int,
        default=None,
        help='Dilatazione aggiuntiva opzionale della segmentation mask.',
    )
    return parser.parse_args()


def apply_optional_dilation(mask, dilation_size):
    if dilation_size is None or dilation_size == 0:
        return mask
    kernel_size = dilation_size if dilation_size % 2 == 1 else dilation_size + 1
    return cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        ),
        iterations=1,
    )


def make_preview(frame_bgr, result, mask, view, opacity):
    mask_bool = mask > 0
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    overlay = frame_bgr.copy()
    green = np.zeros_like(frame_bgr)
    green[..., 1] = 255
    blended = cv2.addWeighted(
        frame_bgr, 1.0 - opacity, green, opacity, 0.0
    )
    overlay[mask_bool] = blended[mask_bool]

    landmarks = draw_hand_landmarks(frame_bgr, result)
    segmentation = draw_hand_segmentation(frame_bgr, hand_mask=mask)
    annotated = draw_hand_segmentation(landmarks, hand_mask=mask)

    if view == 'mask':
        return mask_bgr
    if view == 'overlay':
        return overlay
    if view == 'landmarks':
        return landmarks
    if view == 'segmentation':
        return segmentation
    return np.concatenate((frame_bgr, annotated, mask_bgr), axis=1)


def detect_preview(frame_bgr, landmarker, args, timestamp_ms):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detect_hand_landmarks(
        frame_rgb, landmarker, timestamp_ms=timestamp_ms
    )
    mask = create_hand_segmentation_mask(result, frame_bgr.shape[:2])
    mask = apply_optional_dilation(mask, args.dilation_size)
    return make_preview(frame_bgr, result, mask, args.view, args.opacity)


def process_image(args, landmarker):
    image = cv2.imread(args.img_path)
    if image is None:
        raise ValueError(
            'Impossibile aprire l’immagine: {}'.format(args.img_path)
        )

    output_path = Path(args.output or './results/hand_mask_preview.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview = detect_preview(image, landmarker, args, timestamp_ms=0)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(
            'Impossibile salvare l’immagine: {}'.format(output_path)
        )
    print('Creata {}.'.format(output_path))


def process_video(args, landmarker):
    video = cv2.VideoCapture(args.vid_path)
    if not video.isOpened():
        raise ValueError(
            'Impossibile aprire il video: {}'.format(args.vid_path)
        )

    fps = video.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = width * 3 if args.view == 'side-by-side' else width
    output_path = Path(args.output or './results/hand_mask_preview.mp4')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (output_width, height),
    )
    if not writer.isOpened():
        video.release()
        raise RuntimeError(
            'Impossibile creare il video di output: {}'.format(output_path)
        )

    frame_index = 0
    try:
        while True:
            success, frame_bgr = video.read()
            if not success:
                break
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            writer.write(
                detect_preview(
                    frame_bgr, landmarker, args, timestamp_ms
                )
            )
            frame_index += 1
    finally:
        writer.release()
        video.release()

    if frame_index == 0:
        raise RuntimeError('Il video non contiene frame leggibili.')
    print(
        'Creato {} ({} frame, {:.3f} FPS).'.format(
            output_path, frame_index, fps
        )
    )


def main():
    args = parse_args()
    if not 0.0 <= args.opacity <= 1.0:
        raise ValueError('--opacity deve essere compresa tra 0 e 1.')
    if args.dilation_size is not None and args.dilation_size < 0:
        raise ValueError('--dilation_size non può essere negativa.')

    landmarker = create_hand_landmarker(args.hand_task_path)
    try:
        if args.img_path:
            process_image(args, landmarker)
        else:
            process_video(args, landmarker)
    finally:
        landmarker.close()


if __name__ == '__main__':
    main()
