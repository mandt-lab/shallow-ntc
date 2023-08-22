import tensorflow as tf
import tensorflow_probability as tfp


# round_st = tfc.round_st


@tf.function
def sga_round_no_offset(mu: tf.Tensor, tau: float, epsilon: float = 1e-5):
  """
  Draw a single sample from a stochastic rounding distribution defined by SGA.
  :param mu: "location" variational parameter of the stochastic rounding
    distribution in SGA.
  :param tau: temperature of the rounding distribution, as well as in the
    Gumbel-softmax trick for sampling.
  :param epsilon: small constant for numerical stability
  :return:
  """
  mu_floor = tf.math.floor(mu)
  mu_ceil = tf.math.ceil(mu)
  mu_bds = tf.stack([mu_floor, mu_ceil], axis=-1)
  round_dir_dist_logits = tf.stack(
    [
      -tf.math.atanh(
        tf.clip_by_value(mu - mu_floor, -1 + epsilon, 1 - epsilon)) / tau,
      -tf.math.atanh(
        tf.clip_by_value(mu_ceil - mu, -1 + epsilon, 1 - epsilon)) / tau
    ],
    axis=-1)  # last dim are logits for DOWN or UP
  # Create a Concrete distribution of the rounding direction r.v.s.
  round_dir_dist = tfp.distributions.RelaxedOneHotCategorical(
    tau, logits=round_dir_dist_logits
  )  # We can use a different temperature here, but it hasn't been explored.
  round_dir_sample = round_dir_dist.sample()
  stoch_round_sample = tf.reduce_sum(
    mu_bds * round_dir_sample, axis=-1)  # inner product in last dim
  return stoch_round_sample


@tf.function
def sga_round(mu: tf.Tensor, tau: float, offset=None, epsilon: float = 1e-5):
  """
  Same as sga_round_no_offset but allow rounding to a shifted integer grid.
  """
  if offset is None:
    return sga_round_no_offset(mu, tau, epsilon)
  else:
    return sga_round_no_offset(mu - offset, tau, epsilon) + offset


def get_sga_schedule(r, ub, lb=1e-8, scheme='exp', t0=200.0):
  """
  Get the annealing schedule for the temperature (tau) param in SGA.
  Based on
  https://github.com/mandt-lab/improving-inference-for-neural-image-compression/blob/c9b5c1354a38e0bb505fc34c6c8f27170f62a75b/utils.py#L151
  :param r: decay strength
  :param ub: maximum/init temperature
  :param lb: small const like 1e-8 to prevent numerical issue when temperature too
      close to 0
  :param scheme: 'exp' or 'linear'; default is 'exp'.
  :param t0: the number of "warmup" iterations, during which the temperature is fixed
      at the value ub.
  :return callable t -> tau(t)
  """
  backend = tf

  def schedule(t):
    # :param t: step/iteration number
    t = tf.cast(t, tf.float32)  # step variable is usually tf.int64
    if scheme == 'exp':
      tau = ub * backend.exp(-r * (t - t0))
    elif scheme == 'linear':
      # Cool temperature linearly from ub after the initial t0 iterations
      tau = -r * (t - t0) + ub
    else:
      raise NotImplementedError

    if backend is None:
      return min(max(tau, lb), ub)
    else:
      return backend.minimum(backend.maximum(tau, lb), ub)

  return schedule


default_sga_schedule = get_sga_schedule(
  r=1e-3, ub=0.5, scheme='exp', t0=700.0)  # default from paper


def sga_schedule_at_step(t, r, ub, lb=1e-8, t0=200.0):
  """
  Evaluate SGA schedule at step t.
  :param t: decay rate
  :param r:
  :param ub:
  :param lb:
  :param t0: initial num steps of no decay.
  :return:
  """
  t = tf.cast(t, tf.float32)  # step variable is usually tf.int64
  tau = ub * tf.exp(-r * (t - t0))
  tau = tf.minimum(tf.maximum(tau, lb), ub)
  return tau


