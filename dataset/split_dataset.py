"""Split an image dataset into deterministic train and validation folders."""

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {
    '.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'
}


def collect_images(input_dir, output_dir):
    """Return supported images, excluding an output nested in the input."""
    images = []
    for path in input_dir.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path == output_dir or output_dir in path.parents:
            continue
        images.append(path)
    return sorted(images)


def copy_split(images, input_dir, destination):
    for image_path in images:
        relative_path = image_path.relative_to(input_dir)
        output_path = destination / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, output_path)


def split_dataset(input_dir, output_dir, train_fraction, seed):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(
            'Dataset directory not found: {}'.format(input_dir)
        )
    if not 0.0 < train_fraction < 1.0:
        raise ValueError('--train_fraction must be between 0 and 1')

    train_dir = output_dir / 'train'
    validation_dir = output_dir / 'validation'
    if train_dir.exists() or validation_dir.exists():
        raise FileExistsError(
            '{} already contains train or validation. Choose an empty output '
            'directory to avoid mixing different splits.'.format(output_dir)
        )

    images = collect_images(input_dir, output_dir)
    if len(images) < 2:
        raise ValueError(
            'At least 2 supported images are required; found {}.'.format(
                len(images)
            )
        )

    random.Random(seed).shuffle(images)
    split_index = int(len(images) * train_fraction)
    split_index = min(max(split_index, 1), len(images) - 1)

    train_images = images[:split_index]
    validation_images = images[split_index:]
    copy_split(train_images, input_dir, train_dir)
    copy_split(validation_images, input_dir, validation_dir)

    print('Dataset:', input_dir)
    print('Output:', output_dir)
    print('Seed:', seed)
    print('Training images:', len(train_images))
    print('Validation images:', len(validation_images))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Copy an image dataset into deterministic train and validation '
            'subdirectories.'
        )
    )
    parser.add_argument(
        '--input_dir', required=True,
        help='path to the image dataset to split'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='destination that will contain train and validation'
    )
    parser.add_argument(
        '--train_fraction', type=float, default=0.8,
        help='fraction assigned to training (default: 0.8)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='random seed used for a reproducible split (default: 42)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    split_dataset(
        Path(args.input_dir),
        Path(args.output_dir),
        args.train_fraction,
        args.seed,
    )
