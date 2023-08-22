"""
Hyperparameter configs for training the proposed model with ELIC analysis and two-layer synthesis.
"""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.train_data_config = dict(
    dataset="cocotrain",
    batchsize=8,
    patchsize=256,
  )
  config.val_data_config = dict(
    dataset="kodak_landscape",
    batchsize=1,
    patchsize=None,
  )
  config.train_eval_config = dict(
    num_steps=1800000,
    log_metrics_every_steps=1000,
    checkpoint_every_steps=10000,
    eval_every_steps=10000,
  )

  config.model_config = dict(
    scheduled_num_steps=config.train_eval_config.num_steps,
    rd_lambda=0.08,
    optimizer_config=dict(
      learning_rate=1e-4, reduce_lr_after=0.8, reduce_lr_factor=0.1,
      global_clipnorm=1.0,
    ),
    transform_config=dict(
      analysis=dict(cls="ElicAnalysis", channels=(192, 192, 192, 320)),
      synthesis=dict(cls="TwoLayerResSynthesis", channels=(12, 3), strides=(8, 2),
                     kernel_sizes=(13, 5), activation_type="igdn", res_type="conv")
    ),
    latent_config=dict(
      # uq=dict(method='mixedq')  # Train with mixed quantization.
      uq=dict(method='unoise')  # Train with uniform noise injection (default).
    )
  )

  return config


def get_cfg_str(config):
  from collections import OrderedDict
  runname_dict = OrderedDict()
  runname_dict['rd_lambda'] = config.model_config.rd_lambda
  runname_dict['bottleneck_size'] = config.model_config.transform_config.analysis.channels[-1]
  runname_dict['hidden_channels'] = config.model_config.transform_config.synthesis.channels[0]
  runname_dict['k1'] = config.model_config.transform_config.synthesis.kernel_sizes[0]
  runname_dict['k2'] = config.model_config.transform_config.synthesis.kernel_sizes[1]
  runname_dict['act'] = config.model_config.transform_config.synthesis.activation_type
  runname_dict['uq_method'] = config.model_config.latent_config.uq.method

  from common import utils
  return utils.config_dict_to_str(runname_dict)


def get_hyper():
  """
  Produce a list of flattened dicts, each containing a hparam configuration overriding the one in
  get_config(), corresponding to one hparam trial/experiment/work unit.
  :return:
  """
  from common import hyper
  # Example sweep over combinations of rd_lambdas and hidden activations.
  rd_lambdas = [0.08, 0.02, 0.005, 0.00125, 0.04, 0.01, 0.0025]
  rd_lambdas = hyper.sweep('model_config.rd_lambda', rd_lambdas)
  activations = ['gelu', 'lrelu']
  activations = hyper.sweep('model_config.transform_config.synthesis.activation_type',
                            activations)

  # # Also sweep over channels of the two syn layers.
  # hidden_channels = [12]
  # channels = [(hc, 3) for hc in hidden_channels]
  # channels = hyper.sweep('model_config.transform_config.synthesis.channels', channels)

  hparam_cfgs = hyper.product(rd_lambdas, activations)

  return hparam_cfgs
