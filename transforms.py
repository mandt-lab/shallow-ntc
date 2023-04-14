"""NN transforms used by various NTC methods from the paper."""
import tensorflow as tf
import tensorflow_compression as tfc
import functools
from common import elic


class GDN1(tfc.GDN):
  def __init__(self,
               inverse=False,
               rectify=False,
               data_format="channels_last",
               beta_parameter=None,
               gamma_parameter=None,
               **kwargs):
    super().__init__(inverse, rectify, data_format, alpha_parameter=1,
                     beta_parameter=beta_parameter,
                     gamma_parameter=gamma_parameter, epsilon_parameter=1,
                     **kwargs)

  def call(self, inputs) -> tf.Tensor:
    # Overriding parent method to use only the computation path for GDN1.
    # Also see https://interdigitalinc.github.io/CompressAI/_modules/compressai/layers/gdn.html
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    rank = inputs.shape.rank
    if rank is None or rank < 2:
      raise ValueError(f"Input tensor must have at least rank 2, received "
                       f"shape {inputs.shape}.")

    if self.rectify:
      inputs = tf.nn.relu(inputs)

    norm_pool = abs(inputs)  # alpha == 1

    # Compute normalization pool.
    if rank == 2:
      norm_pool = tf.linalg.matmul(norm_pool, self.gamma)
      norm_pool = tf.nn.bias_add(norm_pool, self.beta)
    elif self.data_format == "channels_last" and rank <= 5:
      shape = self.gamma.shape
      gamma = tf.reshape(self.gamma, (rank - 2) * [1] + shape)
      norm_pool = tf.nn.convolution(norm_pool, gamma, padding="VALID")
      norm_pool = tf.nn.bias_add(norm_pool, self.beta)
    else:  # generic implementation
      # This puts channels in the last dimension regardless of input.
      norm_pool = tf.linalg.tensordot(
        norm_pool, self.gamma, [[self._channel_axis], [0]])
      norm_pool += self.beta
      if self.data_format == "channels_first":
        # Return to channels_first format if necessary.
        axes = list(range(rank - 1))
        axes.insert(1, rank - 1)
        norm_pool = tf.transpose(norm_pool, axes)

    if self.inverse:
      return inputs * norm_pool
    else:
      return inputs / norm_pool


def get_activation_op(activation):
  if activation is None:
    return activation
  if activation == 'siren':
    return SIREN_act
  if activation == 'prelu':
    return tf.keras.layers.PReLU()
  if activation.lower() in ('gdn', 'gdn1'):
    return GDN1()
  if activation.lower() in ('igdn', 'igdn1'):
    return GDN1(inverse=True)

  if activation == 'lrelu':
    activation = 'leaky_relu'
  return getattr(tf.nn, activation)


conv_k5s2 = functools.partial(tf.keras.layers.Conv2D, kernel_size=5, strides=2, padding="SAME",
                              use_bias=True)
conv_k3s1 = functools.partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1, padding="SAME",
                              use_bias=True)
conv_t_k5s2 = functools.partial(tf.keras.layers.Conv2DTranspose, kernel_size=5, strides=2,
                                padding="SAME",
                                use_bias=True)
conv_t_k3s1 = functools.partial(tf.keras.layers.Conv2DTranspose, kernel_size=3, strides=1,
                                padding="SAME",
                                use_bias=True)


class BLS2017Analysis(tf.keras.Sequential):
  def __init__(self, num_filters):
    def get_act():
      return get_activation_op('gdn')

    super().__init__(name="analysis")
    self.add(tfc.SignalConv2D(
      num_filters, (9, 9), name="layer_0", corr=True, strides_down=4,
      padding="same_zeros", use_bias=True,
      activation=get_act()))
    self.add(tfc.SignalConv2D(
      num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
      padding="same_zeros", use_bias=True,
      activation=get_act()))
    self.add(tfc.SignalConv2D(
      num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
      padding="same_zeros", use_bias=False,
      activation=None))


class BLS2017Synthesis(tf.keras.Sequential):
  def __init__(self, num_filters):
    def get_act():
      return get_activation_op('igdn')

    super().__init__(name="synthesis")
    self.add(tfc.SignalConv2D(
      num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
      padding="same_zeros", use_bias=True,
      activation=get_act()))
    self.add(tfc.SignalConv2D(
      num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
      padding="same_zeros", use_bias=True,
      activation=get_act()))
    self.add(tfc.SignalConv2D(
      3, (9, 9), name="layer_2", corr=False, strides_up=4,
      padding="same_zeros", use_bias=True,
      activation=None))



class CNNAnalysis(tf.keras.Sequential):
  """Analysis transform from mbt2018 (mean-scale hyperprior)."""

  def __init__(self, channels_base, output_channels=None, activation_type="leaky_relu"):
    activation = get_activation_op(activation_type)
    if output_channels is None:
      output_channels = channels_base
    layers = [
      conv_k5s2(channels_base, activation=activation),
      conv_k5s2(channels_base, activation=activation),
      conv_k5s2(channels_base, activation=activation),
      conv_k5s2(output_channels, activation=None)
    ]
    super().__init__(layers=layers)


class CNNSynthesis(tf.keras.Sequential):
  """Synthesis transform from mbt2018 (mean-scale hyperprior)."""

  def __init__(self, channels_base, output_channels=3, activation_type="leaky_relu"):
    activation = get_activation_op(activation_type)
    layers = [
      conv_t_k5s2(channels_base, activation=activation),
      conv_t_k5s2(channels_base, activation=activation),
      conv_t_k5s2(channels_base, activation=activation),
      conv_t_k5s2(output_channels, activation=None)
    ]
    super().__init__(layers=layers)



class HyperAnalysis(tf.keras.Sequential):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, bottleneck_size, activation_type="relu"):
    activation = get_activation_op(activation_type)
    layers = [
      conv_k3s1(bottleneck_size, activation=activation),
      conv_k5s2(bottleneck_size, activation=activation),
      conv_k5s2(bottleneck_size, activation=None)
    ]
    super().__init__(layers=layers, name="HyperAnalysis")


class HyperSynthesis(tf.keras.Sequential):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, bottleneck_size, activation_type="relu"):
    activation = get_activation_op(activation_type)
    layers = [
      conv_t_k5s2(bottleneck_size, activation=activation),
      conv_t_k5s2(int(bottleneck_size * 1.5), activation=activation),
      conv_t_k3s1(bottleneck_size * 2, activation=None),
    ]
    super().__init__(layers=layers, name="HyperSynthesis")



class JPEGLikeSynthesis(tf.keras.Model):
  def __init__(self, output_channels=3, kernel_size=16, strides=16, padding='SAME', use_bias=True):
    """
    :param use_offset:
    :param output_channels:
    :param kernel_size:
    :param strides:
    :param padding:
    :param use_bias:
    :param kernel_init:
    """
    super().__init__()
    self.conv = tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=kernel_size,
                                              strides=strides,
                                              padding=padding,
                                              use_bias=use_bias)

  def call(self, x, training):
    x = self.conv(x)
    return x



class TwoLayerSynthesis(tf.keras.Model):
  def __init__(self, channels=(24, 3), strides=(8, 2), kernel_sizes=(13, 5),
          activation_type="igdn"):
    """
    """
    super().__init__()
    activation = get_activation_op(activation_type)
    self.conv1 = tf.keras.layers.Conv2DTranspose(filters=channels[0], kernel_size=kernel_sizes[0],
                                                 strides=strides[0], activation=activation,
                                                 padding="SAME", use_bias=True)

    self.conv2 = tf.keras.layers.Conv2DTranspose(filters=channels[1], kernel_size=kernel_sizes[1],
                                                 strides=strides[1], activation=None,
                                                 padding="SAME", use_bias=True)

  def call(self, z, training):
    x = self.conv2(self.conv1(z))
    return x



class TwoLayerResSynthesis(tf.keras.Model):
  def __init__(self, channels=(12, 3), strides=(8, 2), kernel_sizes=(13, 5),
               activation_type="igdn", res_type="conv"):
    """
    Same as above, but with a residual connection.
    """
    super().__init__()
    activation = get_activation_op(activation_type)
    self.base_conv = tf.keras.layers.Conv2DTranspose(filters=channels[0],
                                                     kernel_size=kernel_sizes[0],
                                                     strides=strides[0], activation=activation,
                                                     padding="SAME", use_bias=True)
    if res_type == "conv":
      self.res = tf.keras.layers.Conv2DTranspose(filters=channels[0], kernel_size=kernel_sizes[0],
                                                 strides=strides[0], activation=None,
                                                 padding="SAME", use_bias=True)
    elif res_type == "d2s":
      k = 1
      self.res = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)),
        tf.keras.layers.Conv2D(192, kernel_size=k, padding='SAME', activation='leaky_relu'),
        tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)),
        tf.keras.layers.Conv2D(channels[0] * 4, kernel_size=k, padding='SAME',
                               activation='leaky_relu'),
        tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2)),
      ])
    else:
      raise NotImplementedError

    self.activation = activation

    self.out_conv = tf.keras.layers.Conv2DTranspose(filters=channels[1],
                                                    kernel_size=kernel_sizes[1],
                                                    strides=strides[1], activation=None,
                                                    padding="SAME", use_bias=True)

  def call(self, z, training):
    x = self.out_conv(self.base_conv(z) + self.res(z))
    return x


