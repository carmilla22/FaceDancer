import csv
import random
from pathlib import Path

import tensorflow as tf


image_feature_description = {
    'face': tf.io.FixedLenFeature([], tf.string),
    }


def _get_occlusion_pairs(occlusion_data_dir, split, train_fraction,
                         split_seed):
    data_dir = Path(occlusion_data_dir)
    metadata_path = data_dir / 'metadata.csv'
    if not metadata_path.is_file():
        raise FileNotFoundError(
            'Occlusion metadata not found: {}'.format(metadata_path)
        )

    image_paths = []
    mask_paths = []
    with metadata_path.open(newline='', encoding='utf-8-sig') as metadata_file:
        reader = csv.DictReader(metadata_file)
        required_fields = {'output_image', 'output_mask'}
        if not required_fields.issubset(reader.fieldnames or []):
            raise ValueError(
                '{} must contain output_image and output_mask.'.format(
                    metadata_path
                )
            )

        for row in reader:
            image_paths.append(str(data_dir / 'images' / row['output_image']))
            mask_paths.append(str(data_dir / 'hand_masks' / row['output_mask']))

    if not image_paths:
        raise ValueError('No occluded samples found in {}'.format(metadata_path))

    missing = [
        path for path in image_paths + mask_paths
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            'Occlusion dataset contains {} missing files. First: {}'.format(
                len(missing), ', '.join(missing[:3])
            )
        )

    pairs = list(zip(image_paths, mask_paths))
    random.Random(split_seed).shuffle(pairs)
    split_index = int(len(pairs) * train_fraction)

    if split == 'train':
        pairs = pairs[:split_index]
    elif split in {'validation', 'eval', 'test'}:
        pairs = pairs[split_index:]
    else:
        raise ValueError(
            "occlusion_split must be 'train' or 'validation', got: {}".format(
                split
            )
        )

    if not pairs:
        raise ValueError(
            'The {} occlusion split is empty.'.format(split)
        )

    split_images, split_masks = zip(*pairs)
    return list(split_images), list(split_masks)


def get_tf_dataset(tfrecords_paths, im_size=256, batchsize=10, repeat=False,
                   occlusion_data_dir=None, occlusion_split='train',
                   occlusion_train_fraction=0.7, occlusion_split_seed=42):
    def decode_img(img):
        img = tf.image.decode_png(img, channels=3)
        return (tf.image.resize(img, [im_size, im_size]) - 127.5) / 127.5

    def _parse_image_function(example_proto):
        data_dict = tf.io.parse_single_example(example_proto, image_feature_description)
        face = decode_img(data_dict['face'])
        return face

    def _decode_occluded_sample(image_path, mask_path):
        source = tf.io.read_file(image_path)
        source = tf.image.decode_jpeg(source, channels=3)
        source = tf.image.resize(source, [im_size, im_size])
        source = (tf.cast(source, tf.float32) - 127.5) / 127.5
        source.set_shape((im_size, im_size, 3))

        source_mask = tf.io.read_file(mask_path)
        source_mask = tf.image.decode_png(source_mask, channels=1)
        source_mask = tf.image.resize(
            source_mask, [im_size, im_size], method='nearest'
        )
        source_mask = tf.cast(source_mask >= 127.5, tf.float32)
        source_mask.set_shape((im_size, im_size, 1))

        return source, source_mask

    files_targets = tf.io.matching_files(tfrecords_paths).numpy()
    files_target = tf.random.shuffle(files_targets)
    shards_target = tf.data.Dataset.from_tensor_slices(files_target)
    dataset_target = shards_target.interleave(tf.data.TFRecordDataset)
    dataset_target = dataset_target.shuffle(buffer_size=1000)
    dataset_target = dataset_target.map(map_func=_parse_image_function,
                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset_target = dataset_target.batch(batchsize, drop_remainder=True)

    if occlusion_data_dir is None:
        raise ValueError(
            'occlusion_data_dir is required and must point to dataset_finale.'
        )

    if not 0.0 < occlusion_train_fraction < 1.0:
        raise ValueError('occlusion_train_fraction must be between 0 and 1.')

    occluded_images, hand_masks = _get_occlusion_pairs(
        occlusion_data_dir,
        split=occlusion_split,
        train_fraction=occlusion_train_fraction,
        split_seed=occlusion_split_seed
    )
    dataset_source = tf.data.Dataset.from_tensor_slices(
        (occluded_images, hand_masks)
    )
    dataset_source = dataset_source.shuffle(len(occluded_images)).repeat()
    dataset_source = dataset_source.map(
        _decode_occluded_sample,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset_source = dataset_source.batch(batchsize, drop_remainder=True)

    dataset_total = tf.data.Dataset.zip(
        (dataset_target, dataset_source)
    )
    dataset_total = dataset_total.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )

    if repeat:
        dataset_total = dataset_total.repeat()

    return dataset_total
