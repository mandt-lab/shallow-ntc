import tensorflow as tf
import numpy as np
from typing import Optional, Callable


def center_crop_image(image, target_height, target_width):
  # Based on https://github.com/keras-team/keras/blob/v2.10.0/keras/layers/preprocessing/image_preprocessing.py#L202
  input_shape = tf.shape(image)
  H_AXIS = -3
  W_AXIS = -2
  h_diff = input_shape[H_AXIS] - target_height
  w_diff = input_shape[W_AXIS] - target_width

  tf.debugging.assert_greater_equal(h_diff, 0)
  tf.debugging.assert_greater_equal(w_diff, 0)

  h_start = tf.cast(h_diff / 2, tf.int32)
  w_start = tf.cast(w_diff / 2, tf.int32)
  return tf.image.crop_to_bounding_box(image, h_start, w_start, target_height, target_width)


def quantize_image(image):
  return tf.saturate_cast(tf.round(image), tf.uint8)


def mse_psnr(x, y, max_val=255.):
  """Compute MSE and PSNR b/w two image tensors."""
  x = tf.cast(x, tf.float32)
  y = tf.cast(y, tf.float32)

  squared_diff = tf.math.squared_difference(x, y)
  axes_except_batch = list(range(1, len(squared_diff.shape)))

  # Results have shape [batch_size]
  mses = tf.reduce_mean(tf.math.squared_difference(x, y), axis=axes_except_batch)  # per img
  # psnrs = -10 * (np.log10(mses) - 2 * np.log10(255.))
  psnrs = -10 * (tf.math.log(mses) - 2 * tf.math.log(max_val)) / tf.math.log(10.)
  return mses, psnrs


def pad_images(x, div: int, padding_mode='reflect'):
  """
  Pad a batch of images so their height and width are divisible by div. Will always pad in the
  bottom right corner for easy inverse.
  :param x: a batch of imgs; 4D tensor [B, H, W, C]
  :param div: the integer that the padded image dimension (both height and width) should be
  divisible by.
  :return:
  """
  # Based on https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention
  # /blob/c13652f7b77fade0641fb31eca90da696c6f5ff9/network.py#L475
  x_shape = tf.shape(x)  # [B, H, W, C]
  div = tf.constant([1, div, div, 1], dtype=tf.int32)
  ratio = tf.math.ceil(x_shape / div)  # e.g., ceil([2, 768, 512, 3] / [1, 100, 100, 1])
  ratio = tf.cast(ratio, tf.int32)
  padded_size = tf.multiply(ratio, div)  # e.g., [2, 800, 600, 3]
  if tf.reduce_all(padded_size == x_shape):  # special case, no need for padding
    return x

  # Will pad to the bottom right.
  slack = padded_size - x_shape  # e.g., [2, 800, 600, 3] - [2, 768, 512, 3] = [0, 32, 88, 0]
  slack = tf.expand_dims(slack, 1)
  paddings = tf.concat([tf.convert_to_tensor(np.zeros((4, 1)), dtype=tf.int32), slack], axis=1)
  x_padded = tf.pad(x, paddings, padding_mode)

  return x_padded


def unpad_images(x, unpadded_shape):
  # Undo the above.
  return x[:, :unpadded_shape[1], :unpadded_shape[2], :]
