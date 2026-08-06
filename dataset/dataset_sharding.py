from tqdm import tqdm

import tensorflow as tf
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')

import os
import random


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def image_example(file_path):
    """Turn data set into tf.Example."""

    face = open(file_path, 'rb').read()

    feature = {
        'face': _bytes_feature(face),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="I:/Datasets/DeepFakeChallenge/image_dataset/",
                        help='path to dataset directory of face images'
                             '(should already be aligned and resized)')
    parser.add_argument('--target_dir', type=str, default="I:/Datasets/DeepFakeChallenge/tfrecords/",
                        help='path to save the shards.')
    parser.add_argument('--data_name', type=str, default="vggface2",
                        help='name of the data set')
    parser.add_argument('--data_type', type=str, default="train",
                        help='is it training, testing or validation data?')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='shuffle that order of the image paths.')
    parser.add_argument('--num_shards', type=int, default=1000,
                        help='number of images per shard')
    parser.add_argument('--max_images', type=int, default=None,
                        help='optional maximum number of images to shard')

    opt = parser.parse_args()

    data_path = opt.data_dir
    folder_list = os.listdir(data_path)
    images_list = []

    for fld in tqdm(folder_list):
        folder_path = os.path.join(data_path, fld)
        if os.path.isfile(folder_path):
            images_list.append(folder_path)
            continue
        for im in os.listdir(folder_path):
            image_path = os.path.join(folder_path, im)
            if os.path.isfile(image_path):
                images_list.append(image_path)

    if opt.shuffle:
        random.shuffle(images_list)
    if opt.max_images is not None:
        if opt.max_images <= 0:
            raise ValueError('--max_images must be greater than zero')
        images_list = images_list[:opt.max_images]
    index = 0
    n_images_shard = opt.num_shards
    n_shards = int(len(images_list) / n_images_shard) + (1 if len(images_list) % n_images_shard != 0 else 0)

    dataset_name = opt.data_name
    train_val_test = opt.data_type
    os.makedirs(opt.target_dir, exist_ok=True)
    tfrecords_path = os.path.join(
        opt.target_dir, "{}_{}_{}.records"
    )

    for shard in tqdm(range(n_shards)):
        tfrecords_shard_path = tfrecords_path.format(dataset_name,
                                                     train_val_test,
                                                     '%.5d-of-%.5d' % (shard, n_shards - 1))
        end = min(index + n_images_shard, len(images_list))
        images_shard_list = images_list[index: end]
        with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
            for filename in images_shard_list:
                tf_example = image_example(filename)
                writer.write(tf_example.SerializeToString())
        index = end