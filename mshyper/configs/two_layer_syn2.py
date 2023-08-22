"""
Hyperparameter configs for training an improved model with a four-layer conv analysis and two-layer
synthesis. The differences compared to the proposed two-layer syn architecture are:
 - We use the cheaper CNN analysis (similar to in Hyperprior) than the one with attention in ELIC.
 - Here we drop the residual connection in the two-layer synthesis, which allows us to double the number of conv channels without too much increase in FLOPs (the first conv layer
  dominates the FLOP count).
 - We also train with mixed quantization as in Minnen et al. 2020.
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
      analysis=dict(cls="CNNAnalysis", channels_base=256, output_channels=320),
      synthesis=dict(cls="TwoLayerSynthesis", channels=(12, 3), strides=(8, 2),
                     kernel_sizes=(13, 5), activation_type="igdn")
    ),
    latent_config=dict(
      uq=dict(method='mixedq')  # Train with mixed quantization.
      # uq=dict(method='unoise')  # Train with uniform noise injection (default).
    )
  )

  return config


def get_cfg_str(config):
  from collections import OrderedDict
  runname_dict = OrderedDict()
  runname_dict['ana'] = config.model_config.transform_config.analysis.cls
  runname_dict['ana_cb'] = config.model_config.transform_config.analysis.channels_base
  runname_dict['rd_lambda'] = config.model_config.rd_lambda
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
  rd_lambdas = [0.08, 0.02, 0.005, 0.00125]
  rd_lambdas = hyper.sweep('model_config.rd_lambda', rd_lambdas)
  hidden_channels = [24, 48]
  channels = [(hc, 3) for hc in hidden_channels]
  channels = hyper.sweep('model_config.transform_config.synthesis.channels', channels)
  uq_methods = ['unoise', 'mixedq']
  uq_methods = hyper.sweep('model_config.latent_config.uq.method', uq_methods)
  hparam_cfgs = hyper.product(rd_lambdas, channels, uq_methods)
  return hparam_cfgs
