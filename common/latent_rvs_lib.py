"""
A small library for working with latent variables in variational compression models.

The random variables are analogous to tensorflow-probability distributions, and are governed by
variational parameters that are either tf.Tensors or tf.Variables (the latter are optimized w.r.t
during iterative inference). The "sample" method of each random variable can be thought of as
adding quantization noise to the (usually location) variational parameter as a differentiable
approximation to hard quantization, or as drawing from a proper encoding distribution like in
reverse channel coding.
"""

import copy
from typing import Any, NamedTuple, Optional
from typing import Mapping, Sequence

from absl import logging
from ml_collections import config_dict
import tensorflow as tf
import tensorflow_compression as tfc

from common import immutabledict
from common import latent_rvs_utils as utils


class LatentRV(tf.Module):
  """Base class for a latent random variable governed by variational parameters."""

  def sample(self, training: bool, **kwargs) -> tf.Tensor:
    raise NotImplementedError

  def quantize(self) -> tf.Tensor:
    """
    :return: this should return the "hard-quantized" value used at test time for NTC methods.
    """
    raise NotImplementedError

  @property
  def params(self):
    return self._params

  @params.setter
  def params(self, value):
    self._params = value

  @tf.Module.with_name_scope
  def get_trainable_copy(self):
    """Get a copy of the current object, replacing the params with tf.Variables initialized to
    the corresponding values.
    """
    trainable_copy = copy.copy(self)
    # trainable_copy.params = tf.nest.map_structure(tf.Variable, self.params)
    trainable_copy.params = {
      key: tf.Variable(val, trainable=True)
      for (key, val) in self.params.items()
    }
    return trainable_copy


class UQLatentRV(LatentRV):
  """A continuous latent variable that is expected to be uniformly quantized/rounded."""

  def __init__(self, loc: tf.Tensor):
    """
    :param loc: the location parameter; typically predicted by an encoder network.
    """
    super().__init__(name='UQLatentRV')
    self._params = dict(loc=loc)

  @property
  def loc(self):
    return self._params['loc']

  @property
  def shape(self):
    return self.loc.shape

  def quantize(self, offset: Optional[tf.Tensor] = None):
    return tfc.round_st(self.loc, offset=offset)

  @tf.Module.with_name_scope
  def sample(self,
             training: bool,
             method: Optional[str] = None,
             offset: Optional[tf.Tensor] = None,
             **kwargs):
    """
    Sample from the latent distribution, using the specified method.
    :param training: set this to False for evaluation.
    :param method:
    :param offset:
    :param kwargs:
    :return:
    """

    def _apply_op_with_offset(op, x, offset):
      if offset is None:
        return op(x)
      else:
        return op(x - offset) + offset

    if not training:
      return _apply_op_with_offset(tf.round, self.loc, offset)
    else:
      if method == 'unoise':
        u = tf.random.uniform(
          tf.shape(self.loc), minval=-.5, maxval=.5, dtype=self.loc.dtype)
        return u + self.loc
      elif method == 'sga':
        tau = kwargs['tau']
        return utils.sga_round(self.loc, tau=tau, offset=offset)
      elif method == 'soft_round':
        alpha = kwargs['alpha']
        return _apply_op_with_offset(lambda x: tfc.soft_round(x, alpha=alpha),
                                     self.loc, offset)
      else:
        raise NotImplementedError


class CategoricalLatentRV(LatentRV):
  pass


class LatentRVSamples(NamedTuple):
  """Container for the result of sampling from a collection of latent rvs."""

  uq: Sequence[tf.Tensor] = tuple()
  categorical: Sequence[tf.Tensor] = tuple()


class LatentRVCollection(NamedTuple):
  """Container for a collection of latent rvs."""

  uq: Sequence[UQLatentRV] = tuple()
  categorical: Sequence[CategoricalLatentRV] = tuple()

  def sample(
        self,
        training,
        latent_config: Mapping[str, Any] = immutabledict.immutabledict()
  ) -> LatentRVSamples:
    """
    Sample from a collection of latent rvs, using the specified method.
    :param training:
    :param latent_config:
    :return: container of samples
    """

    result_dict = dict()
    for attr in self._fields:
      rvs = getattr(self, attr)
      config_for_rvs = latent_config.get(attr, {})
      samples = [rv.sample(training, **config_for_rvs) for rv in rvs]
      result_dict[attr] = samples

    return LatentRVSamples(**result_dict)

  def get_trainable_copy(self):
    """Get a copy of the current object, replacing the params with tf.Variables initialized to
    the corresponding values.
    """
    return tf.nest.map_structure(lambda rv: rv.get_trainable_copy(), self)

  @property
  def trainable_variables(self):
    return tf.nest.flatten(
      tf.nest.map_structure(lambda rv: rv.trainable_variables, self))
