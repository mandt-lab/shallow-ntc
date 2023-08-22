"""Hyperparameter configs for training the mean-scale hyperprior baseline."""

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
    num_steps=2000000,
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
      analysis=dict(cls="MBT2018Analysis", channels_base=192,
                    output_channels=320),
      synthesis=dict(cls="MBT2018Synthesis", channels_base=192,
                     output_channels=3)
    ),
  )

  return config


def get_cfg_str(config):
  from collections import OrderedDict
  runname_dict = OrderedDict()
  runname_dict['rd_lambda'] = config.model_config.rd_lambda
  runname_dict['bottleneck_size'] = config.model_config.transform_config.analysis.output_channels
  runname_dict['channels_base'] = config.model_config.transform_config.analysis.channels_base

  from common import utils
  return utils.config_dict_to_str(runname_dict)


def get_hyper():
  """
  Produce a list of flattened dicts, each containing a hparam configuration overriding the one in
  get_config(), corresponding to one hparam trial/experiment/work unit.
  :return:
  """
  from common import hyper
  # Here we simply sweep over rd_lambda.
  rd_lambdas = [0.08, 0.02, 0.005, 0.00125, 0.04, 0.01, 0.0025]
  rd_lambdas = hyper.sweep('model_config.rd_lambda', rd_lambdas)
  hparam_cfgs = rd_lambdas
  return hparam_cfgs
