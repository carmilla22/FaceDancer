import tensorflow as tf


image_feature_description = {
    'face': tf.io.FixedLenFeature([], tf.string),
    }


def _parse_image_function(example_proto, im_size):
    data_dict = tf.io.parse_single_example(
        example_proto, image_feature_description
    )
    image = tf.image.decode_image(
        data_dict['face'], channels=3, expand_animations=False
    )
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, [im_size, im_size])
    return (tf.cast(image, tf.float32) - 127.5) / 127.5


def _get_tfrecord_dataset(tfrecords_paths, im_size, batchsize,
                          repeat=False):
    files = tf.io.matching_files(tfrecords_paths).numpy()
    if len(files) == 0:
        raise FileNotFoundError(
            'No TFRecord files match: {}'.format(tfrecords_paths)
        )

    files = tf.random.shuffle(files)
    shards = tf.data.Dataset.from_tensor_slices(files)
    dataset = shards.interleave(tf.data.TFRecordDataset)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(
        lambda example: _parse_image_function(example, im_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if repeat:
        dataset = dataset.repeat()
    return dataset.batch(batchsize, drop_remainder=True)


def get_tf_dataset(tfrecords_paths, source_tfrecords_paths, im_size=256,
                   batchsize=10, repeat=False):
    """Load target and hand-occluded source images from TFRecords.

    Hand masks are intentionally not loaded here: they are detected online by
    MediaPipe in the training loop.
    """
    dataset_target = _get_tfrecord_dataset(
        tfrecords_paths, im_size, batchsize, repeat=repeat
    )
    dataset_source = _get_tfrecord_dataset(
        source_tfrecords_paths, im_size, batchsize, repeat=True
    )

    return tf.data.Dataset.zip((dataset_target, dataset_source)).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )
