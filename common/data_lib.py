import tensorflow as tf
import numpy as np
import configs
from common import image_utils


def read_png(filename, channels=3):
  """Loads a PNG image file."""
  string = tf.io.read_file(filename)
  return tf.image.decode_image(string, channels=channels)


def write_png(filename, image):
  """Saves an image to a PNG file."""
  string = tf.image.encode_png(image)
  tf.io.write_file(filename, string)


def check_image_size(image, patchsize):
  shape = tf.shape(image)
  return shape[0] >= patchsize and shape[1] >= patchsize and shape[-1] == 3


def normalize_image(image):
  return image / 255. - 0.5


def unnormalize_image(float_tensor):
  return (float_tensor + 0.5) * 255.


def process_image(image, crop=None, patchsize=None, normalize=True, image_channels=3):
  if crop is not None:
    assert patchsize > 0
    if crop == "random":
      image = tf.image.random_crop(image, (patchsize, patchsize, image_channels))
    elif crop == "center":
      image = image_utils.center_crop_image(image, patchsize, patchsize)
    else:
      raise NotImplementedError(crop)

  image = tf.cast(image, tf.float32)
  if normalize:
    image = normalize_image(image)
  return image


def floats_to_pixels(x, training):
  x = unnormalize_image(x)
  if not training:
    x = image_utils.quantize_image(x)
  return x


def get_tfds_dataset(name, split, shuffle, repeat, drop_remainder, batchsize, crop=None,
                     patchsize=None, normalize=True):
  """Creates input data pipeline from a TF Datasets dataset.
  :param repeat:
  """
  import tensorflow_datasets as tfds
  with tf.device("/cpu:0"):
    dataset = tfds.load(name, split=split, shuffle_files=shuffle)
    if split == 'test' and shuffle:
      print(
        'Loaded test split with shuffle=True; you may want to use False instead for running evaluation.')
    # if split == "train":
    if repeat:
      dataset = dataset.repeat()
    img_channels = 3
    if patchsize is not None:  # filter out imgs smaller than patchsize (if not using full-sized images)
      if 'cifar' in name:
        assert patchsize <= 32
      elif 'mnist' in name:  # FYI tfds MNIST dataset has image shape [28, 28, 1]
        assert patchsize <= 28
        img_channels = 1
      else:
        dataset = dataset.filter(
          lambda x: check_image_size(x["image"], patchsize))
    dataset = dataset.map(
      lambda x: process_image(x["image"], crop=crop, patchsize=patchsize, normalize=normalize,
                              image_channels=img_channels))
    dataset = dataset.batch(batchsize, drop_remainder=drop_remainder)
  return dataset


def get_dataset_from_glob(file_glob, shuffle, repeat, drop_remainder, batchsize, crop=None,
                          patchsize=None,
                          normalize=True, preprocess_threads=16):
  """Creates input data pipeline from custom PNG images. Formerly named 'get_custom_dataset'.
  :param file_glob:
  :param args:
  """
  import glob
  with tf.device("/cpu:0"):
    files = sorted(glob.glob(file_glob))
    if not files:
      raise RuntimeError(f"No images found with glob '{file_glob}'.")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
      dataset = dataset.shuffle(len(files), reshuffle_each_iteration=True)
    if repeat:
      dataset = dataset.repeat()

    dataset = dataset.map(
      lambda x: process_image(read_png(x), crop=crop, patchsize=patchsize, normalize=normalize),
      num_parallel_calls=preprocess_threads)

    dataset = dataset.batch(batchsize, drop_remainder=drop_remainder)
  return dataset


# TODO: write a unified data loader.
def get_dataset(data_spec, split, batchsize, patchsize, normalize=True):
  if split == "train":
    shuffle = True
    repeat = True
    drop_remainder = True
    crop = "random" if patchsize is not None else None
  else:
    shuffle = False
    repeat = False
    drop_remainder = False
    crop = "center" if patchsize is not None else None

  if data_spec in ("clic", "mnist", "cifar10", "cifar100"):
    if split != "train":
      if data_spec == "clic":
        split = "validation"
      else:
        split = "test"
    dataset = get_tfds_dataset(data_spec, split, shuffle, repeat, drop_remainder, batchsize,
                               crop=crop,
                               patchsize=patchsize,
                               normalize=normalize)
  elif data_spec in configs.dataset_to_globs.keys():
    file_glob = configs.dataset_to_globs[data_spec]
    dataset = get_dataset_from_glob(file_glob, shuffle, repeat, drop_remainder, batchsize,
                                    crop=crop,
                                    patchsize=patchsize, normalize=normalize)
  else:
    # Assume data_spec is a valid glob
    file_glob = data_spec
    dataset = get_dataset_from_glob(file_glob, shuffle, repeat, drop_remainder, batchsize,
                                    crop=crop,
                                    patchsize=patchsize, normalize=normalize)

  return dataset
