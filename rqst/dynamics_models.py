# python3
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Each dynamics model consists of both a transition model p(s'|s,a) and an absorption model that
predicts the probability of s being an absorbing state.
"""

from __future__ import division

from copy import deepcopy
import random
import uuid
import os

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from rqst.models import TFModel
from rqst import utils


class AbsorptionModel(TFModel):
  """A binary classifier that predicts whether or not a state is absorbing"""

  def __init__(
      self,
      *args,
      n_layers=1,
      layer_size=32,
      **kwargs):

    super().__init__(*args, **kwargs)

    self.n_layers = n_layers
    self.layer_size = layer_size

    self.data_keys = ['next_obses', 'dones']

    self.obs_ph = tf.placeholder(tf.float32, [None, self.env.n_obs_dim])
    self.done_ph = tf.placeholder(tf.float32, [None])

    logits = self.build_abs_net(self.obs_ph)
    self.prob_abs_of_obs_ph = self.prob_abs(self.obs_ph)
    xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=self.done_ph, logits=logits)
    self.loss = tf.reduce_mean(xent)

  def build_abs_net(self, obs):
    outputs = utils.build_mlp(
        obs,
        1,
        self.scope,
        n_layers=self.n_layers,
        size=self.layer_size,
        activation=tf.nn.relu,
        output_activation=None)
    return tf.squeeze(outputs, axis=[1])

  def prob_abs(self, obs):
    """Represent the probability of absorption
    Args:
     obs: a tf.Tensor with dimensions (batch_size, n_obs_dim)
    Returns:
     a tf.Tensor with dimensions (batch_size)
    """
    return tf.math.sigmoid(self.build_abs_net(obs))

  def compute_prob_abs(self, obses):
    """Evaluate the probability of absorption
    Args:
     obses: a np.array with dimensions (batch_size, n_obs_dim)
    Returns:
     a np.array with dimensions (batch_size)
    """
    feed_dict = {self.obs_ph: obses}
    return self.sess.run(self.prob_abs_of_obs_ph, feed_dict=feed_dict)

  def sample_abs_mask(self, obses):
    """Sample absorption indicators
    Args:
     obses: a np.array with dimensions (batch_size, n_obs_dim)
    Returns:
     a np.array with dimensions (batch_size, 1)
    """
    abs_mask = np.random.random(obses.shape[0]) < self.compute_prob_abs(obses)
    abs_mask = abs_mask.astype(float)[:, np.newaxis]
    return abs_mask

  def format_batch(self, batch):
    """Format a batch of (s', done) training data
    Args:
      batch: a dict containing the output of a call to rqst.utils.vectorize_rollouts
    Returns:
      a dict containing the input for a call to rqst.models.TFModel.compute_batch_loss
    """
    feed_dict = {
        self.obs_ph:
            batch['next_obses'
                 ],  # recall that dones apply to s', not s, in (s, a, r, s', d)
        self.done_ph: batch['dones']
    }
    return feed_dict


class DynamicsModel(TFModel):

  def __init__(self, *args, abs_model=None, abs_var=1.e-6, **kwargs):
    """
    Args:
     abs_model: an AbsorptionModel object
     abs_var: a float that scales the effect of absorption predictions
      (lower -> stronger absorption)
    """

    super().__init__(*args, **kwargs)

    self.abs_model = abs_model
    self.abs_var = abs_var

    self.encoder = None

  def run_ep(self, *args, **kwargs):
    return utils.run_ep(*args, **kwargs)

  def log_prob_init_obs(self, init_obs):
    """Represent the initial state distribution p(s_0)
    Args:
     init_obs: a tf.Tensor with dimensions (batch_size, n_obs_dim)
    Returns:
     a tf.Tensor with dimensions (batch_size)
    """
    return None

  def log_prob_next_obs(self, next_obs, obs, act):
    """Represent the transition dynamics p(s'|s,a)
    Args:
     next_obs: a tf.Tensor with dimensions (batch_size, n_obs_dim)
     obs: a tf.Tensor with dimensions (batch_size, n_obs_dim)
     act: a tf.Tensor with dimensions (batch_size, n_act_dim)
    Returns:
     a tf.Tensor with dimensions (batch_size)
    """
    return None

  def log_prob_rollout(self, traj, act_seq, init_state=None, mixact_seq=None):
    """Represent the trajectory likelihood p(s_0)*p(s_1|s_0,a_0)*p(s_2|s_1,a_1)*...
    Args:
     traj: a tf.Tensor with dimensions (traj_len, n_obs_dim)
     act_seq: a tf.Tensor with dimensions (traj_len - 1, n_act_dim)
    Returns:
     a tf.Tensor with dimensions (1)
    """
    init_log_prob = self.log_prob_init_obs(traj[0, :])
    trans_log_prob = self.log_prob_transes(
        traj, act_seq, init_state=init_state, mixact_seq=mixact_seq)
    if init_log_prob is not None and trans_log_prob is None:
      return init_log_prob
    elif init_log_prob is None and trans_log_prob is not None:
      return tf.reduce_mean(trans_log_prob)
    elif init_log_prob is not None and trans_log_prob is not None:
      return tf.reduce_mean(
          tf.concat([tf.expand_dims(init_log_prob, 0), trans_log_prob], axis=0))
    else:
      return 0.

  def log_prob_transes(self, traj, act_seq, **kwargs):
    """Represent the transition likelihoods p(s_1|s_0,a_0), p(s_2|s_1,a_1), ...
    Args:
     traj: a tf.Tensor with dimensions (traj_len, n_obs_dim)
     act_seq: a tf.Tensor with dimensions (traj_len - 1, n_act_dim)
    Returns:
     a tf.Tensor with dimensions (traj_len - 1)
    """
    next_obs = traj[1:, :]
    obs = traj[:-1, :]
    return self.log_prob_next_obs(next_obs, obs, act_seq)

  def log_prob_next_obs_with_abs(self, next_obs, obs, act):
    """Represent transition likelihoods p(s'|s,a), marginalizing out absorption indicator
    Args:
     next_obs: a tf.Tensor with dimensions (batch_size, n_obs_dim)
     obs: a tf.Tensor with dimensions (batch_size, n_obs_dim)
     act: a tf.Tensor with dimensions (batch_size, n_act_dim)
    Returns:
     a tf.Tensor with dimensions (batch_size)
    """
    log_prob_without_abs = self.log_prob_next_obs(next_obs, obs, act)
    if self.abs_model is None:
      log_prob = log_prob_without_abs
    else:
      prob_abs = self.abs_model.prob_abs(obs)
      log_prob_with_abs = tf.reduce_mean(
          (next_obs - obs)**2 / self.abs_var, axis=1)
      log_prob = prob_abs * log_prob_with_abs + (
          1. - prob_abs) * log_prob_without_abs
    return log_prob

  def trajs_of_act_seqs(self, init_obses, act_seqs, traj_len, **kwargs):
    """Represent predicted trajectory E[s_1|s_0,a_0], E[s_2|s_0,a_0,a_1], ...
    Args:
     init_obses: a tf.Tensor with dimensions (n_trajs, n_obs_dim)
     act_seqs: a tf.Tensor with dimensions (n_trajs, traj_len - 1, n_act_dim)
    Returns:
     a tf.Tensor with dimensions (n_trajs, traj_len, n_obs_dim)
    """
    trajs = [init_obses]
    obs = init_obses
    for t in range(traj_len - 1):
      obs = self.next_obs(obs, act_seqs[:, t, :])['next_obs_mean']
      trajs.append(tf.cast(obs, dtype=init_obses.dtype))
    trajs = tf.stack(trajs, axis=1)
    return {'trajs': trajs}

  def save(self):
    super().save()
    if self.abs_model is not None:
      self.abs_model.save()

  def load(self):
    super().load()
    if self.abs_model is not None:
      self.abs_model.load()

  def init_tf_vars(self):
    super().init_tf_vars()
    if self.abs_model is not None:
      self.abs_model.init_tf_vars()

  def train(self, *args, **kwargs):
    super().train(*args, **kwargs)
    if self.abs_model is not None:
      self.abs_model.train(*args, **kwargs)

  def rnn_encode_rollouts(self, traj_data):
    return traj_data

  def next_obs(self, obs, act):
    assert self.env.name.endswith('bandit')
    next_obs = tf.constant(self.env.absorbing_state)
    batch_size = tf.shape(obs)[0]
    next_obs = tf.reshape(
        tf.tile(next_obs, tf.expand_dims(batch_size, 0)),
        [batch_size, self.env.n_obs_dim])
    return {'next_obs_mean': next_obs}


class ObsPriorModel(DynamicsModel):
  """Dynamics model for contextual bandit
  Only models p(s), not p(s'|s,a)
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.encoder = kwargs.get('encoder')

  def log_prob_init_obs(self, init_obs):
    return self.encoder.log_prob_latent(init_obs)

  def run_ep(self, *args, **kwargs):
    return utils.run_ep(*args, **kwargs, proc_obs=self.encoder.encode_frame)


class MLPDynamicsModel(DynamicsModel):
  """Multi-layer perceptron"""

  def __init__(self, *args, n_layers=1, layer_size=32, **kwargs):

    super().__init__(*args, **kwargs)

    self.n_layers = n_layers
    self.layer_size = layer_size

    self.data_keys = ['obses', 'actions', 'next_obses']

    self.obs_ph = tf.placeholder(tf.float32, [None, self.env.n_obs_dim])
    self.act_ph = tf.placeholder(tf.float32, [None, self.env.n_act_dim])
    self.next_obs_ph = tf.placeholder(tf.float32, [None, self.env.n_obs_dim])

    self.next_obs_pred = self.build_dyn_net(self.obs_ph, self.act_ph)

    # spherical gaussian p(s'|s,a)
    self.loss = tf.reduce_mean((self.next_obs_ph - self.next_obs_pred)**2)

  def next_obs(self, obs, act):
    """Represent predicted next state E[s'|s,a]
    Args:
     obs: a tf.Tensor with dimensions (batch_size, n_obs_dim)
     act: a tf.Tensor with dimensions (batch_size, n_act_dim)
    Returns:
     a tf.Tensor with dimensions (batch_size, n_obs_dim)
    """
    next_obs_pred = self.build_dyn_net(obs, act)

    var = 1.

    if self.env.name == 'pointmass':
      # use ground-truth dynamics, not trained dynamics model
      unstuck_var = 1.
      stuck_var = 1.
      dist_mult = 10.
      # polar -> cartesian
      vel = tf.concat([
          tf.expand_dims(act[:, 1] * tf.cos(act[:, 0]), 1),
          tf.expand_dims(act[:, 1] * tf.sin(act[:, 0]), 1)
      ],
                      axis=1)
      dist_from_loc = lambda loc: tf.expand_dims(
          tf.norm(obs - loc[np.newaxis, :], ord='euclidean', axis=1), 1)
      dist_from_goal = dist_from_loc(self.env.goal)
      dist_from_trap = dist_from_loc(self.env.trap)
      stuck = (tf.exp(-dist_mult * dist_from_trap) +
               tf.exp(-dist_mult * dist_from_goal)) / 2.

      next_obs_pred = obs + vel  #* (1. - stuck) # uncomment to include absorption in dynamics

      # respect box constraints
      softop = lambda mult: (lambda a, b: tf.reduce_logsumexp(
          mult * tf.stack([a, b], axis=0), axis=0) / mult)
      softop_mult = 100.
      softmax = softop(softop_mult)
      softmin = softop(-softop_mult)

      batch_size = tf.shape(next_obs_pred)[0]
      make_tf_bound = lambda bound: tf.reshape(
          tf.tile(tf.constant(bound), tf.expand_dims(batch_size, 0)),
          [batch_size, self.env.n_obs_dim])
      lows = make_tf_bound(self.env.observation_space.low)
      highs = make_tf_bound(self.env.observation_space.high)

      next_obs_pred = softmax(next_obs_pred, lows)
      next_obs_pred = softmin(next_obs_pred, highs)

      var = unstuck_var * (1. - stuck) + stuck_var * stuck
    elif self.env.name.endswith('bandit'):
      next_obs_pred = self.env.absorbing_state[np.newaxis, :]

    return {'next_obs_mean': next_obs_pred, 'next_obs_var': var}

  def log_prob_next_obs(self, next_obs, obs, act):
    data = self.next_obs(obs, act)
    # spherical gaussian p(s'|s,a)
    return -tf.reduce_mean(
        (next_obs - data['next_obs_mean'])**2 / data['next_obs_var'], axis=1)

  def build_dyn_net(self, obs, act):
    """Represent predicted next state E[s'|s,a,weights] using parametric model
    Args:
     obs: a tf.Tensor with dimensions (batch_size, n_obs_dim)
     act: a tf.Tensor with dimensions (batch_size, n_act_dim)
    Returns:
     a tf.Tensor with dimensions (batch_size, n_obs_dim)
    """
    return utils.build_mlp(
        tf.concat([obs, act], axis=1),
        self.env.n_obs_dim,
        self.scope,
        n_layers=self.n_layers,
        size=self.layer_size,
        activation=tf.nn.relu,
        output_activation=None)

  def format_batch(self, batch):
    feed_dict = {
        self.obs_ph: batch['obses'],
        self.act_ph: batch['actions'],
        self.next_obs_ph: batch['next_obses']
    }
    return feed_dict

  def compute_next_obs(self, obs, act, **kwargs):
    """
    Args:
     trajs: a np.array with dimensions (n_trajs, traj_len, n_obs_dim)
     acts: a np.array with dimensions (n_trajs, traj_len - 1, n_act_dim)
    Returns:
      a dict that maps from 'next_obs' to a np.array with dimensions (n_trajs, n_obs_dim)
      only contains a single k-v pair to maintain compatibility with
       MDNRNNDynamicsModel.compute_next_obs
    """
    feed_dict = {self.obs_ph: obs, self.act_ph: act}
    next_obs = self.sess.run(self.next_obs_pred, feed_dict=feed_dict)
    if self.abs_model is not None:
      abs_mask = self.abs_model.sample_abs_mask(obs)
      next_obs = abs_mask * obs + (1. - abs_mask) * next_obs
    return {'next_obs': next_obs}


class MDNRNNDynamicsModel(DynamicsModel):
  """Adapted from https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/
  """

  def __init__(self,
               encoder,
               *args,
               grad_clip=1.0,
               num_mixture=5,
               use_layer_norm=0,
               use_recurrent_dropout=0,
               recurrent_dropout_prob=0.90,
               use_input_dropout=0,
               input_dropout_prob=0.90,
               use_output_dropout=0,
               output_dropout_prob=0.90,
               **kwargs):

    super().__init__(*args, **kwargs)

    self.encoder = encoder
    self.grad_clip = grad_clip
    self.num_mixture = num_mixture
    self.use_layer_norm = use_layer_norm
    self.use_recurrent_dropout = use_recurrent_dropout
    self.recurrent_dropout_prob = recurrent_dropout_prob
    self.use_input_dropout = use_input_dropout
    self.input_dropout_prob = input_dropout_prob
    self.use_output_dropout = use_output_dropout
    self.output_dropout_prob = output_dropout_prob

    self.rnn_size = self.env.rnn_size

    self.input_x_ph = tf.placeholder(
        dtype=tf.float32,
        shape=[None, None, self.env.n_z_dim + self.env.n_act_dim])
    self.output_x_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, None, self.env.n_z_dim])
    self.seq_lens_ph = tf.placeholder(dtype=tf.int32, shape=[None])
    self.initial_state_c_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, self.rnn_size])
    self.initial_state_h_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, self.rnn_size])

    initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.initial_state_c_ph,
                                                  self.initial_state_h_ph)

    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      dyn_out_vars = self.build_dyn_net(
          self.input_x_ph,
          seq_lens=self.seq_lens_ph,
          initial_state=initial_state)
    self.__dict__.update(dyn_out_vars)

    self.loss = tf.reduce_mean(
        self.build_loss(self.out_logmix, self.out_mean, self.out_logstd,
                        self.output_x_ph))

    self.obs_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, self.env.n_z_dim])
    self.act_ph = tf.placeholder(
        dtype=tf.float32, shape=[None, self.env.n_act_dim])
    data = self.next_obs(self.obs_ph, self.act_ph, init_state=initial_state)
    self.next_obs_pred = data['next_obs_mean']
    self.next_state_pred = data['hidden_state']

  def build_loss(self, logmix, mean, logstd, output_x):
    shape = tf.shape(output_x)
    n_trajs = shape[0]
    traj_len = shape[1]
    shape = [n_trajs, traj_len, self.env.n_z_dim, self.num_mixture]

    mean = tf.reshape(mean, shape)
    logmix = tf.reshape(logmix, shape)
    logstd = tf.reshape(logstd, shape)
    output_x = tf.expand_dims(output_x, 3)

    v = logmix + utils.tf_lognormal(output_x, mean, logstd)
    v = tf.reduce_logsumexp(v, axis=3)
    v = tf.reduce_mean(v, axis=2)
    return -v

  def build_dyn_net(self, input_x, seq_lens=None, initial_state=None):
    cell_fn = utils.LSTMCellWrapper

    if self.use_recurrent_dropout:
      cell = cell_fn(
          self.rnn_size,
          layer_norm=self.use_layer_norm,
          dropout_keep_prob=self.recurrent_dropout_prob)
    else:
      cell = cell_fn(self.rnn_size, layer_norm=self.use_layer_norm)

    if self.use_input_dropout:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, input_keep_prob=self.input_dropout_prob)
    if self.use_output_dropout:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=self.output_dropout_prob)

    NOUT = self.env.n_z_dim * self.num_mixture * 3

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable('output_w', [self.rnn_size, NOUT])
      output_b = tf.get_variable('output_b', [NOUT])

    hidden_states, last_state = tf.nn.dynamic_rnn(
        cell,
        input_x,
        initial_state=initial_state,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='RNN',
        sequence_length=seq_lens)
    c_states = hidden_states[0].c
    hidden_states = hidden_states[0].h

    output = tf.reshape(hidden_states, [-1, self.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    output = tf.reshape(output, [-1, self.num_mixture * 3])
    out_logmix, out_mean, out_logstd = tf.split(output, 3, 1)
    out_logmix -= tf.reduce_logsumexp(out_logmix, 1, keepdims=True)

    return {
        'hidden_states': hidden_states,
        'c_states': c_states,
        'last_state': last_state,
        'out_logmix': out_logmix,
        'out_mean': out_mean,
        'out_logstd': out_logstd,
        'cell': cell
    }

  def build_dyn_out_vars(self, obs, act, init_state=None):
    """
    Args:
     obs: a tf.Tensor with dimensions (n_trajs, traj_len - 1, n_z_dim)
     act: a tf.Tensor with dimensions (n_trajs, traj_len - 1, n_act_dim)
    Returns:
     a dict containing the output of a call to build_dyn_net
    """
    # add dummy dim for n_trajs=1, concat obs/act
    input_x = tf.concat([obs, act], axis=2)
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      if init_state is None:
        init_state = self.cell.zero_state(
            batch_size=tf.shape(obs)[0], dtype=tf.float32)
      dyn_out_vars = self.build_dyn_net(
          input_x, seq_lens=None, initial_state=init_state)
    return dyn_out_vars

  def log_prob_init_obs(self, init_obs):
    return self.encoder.log_prob_latent(init_obs)

  def log_prob_transes(self,
                       traj,
                       act_seq,
                       init_state=None,
                       mixact_seq=None,
                       temperature=0.1):
    """
    Args:
     traj: a tf.Tensor with dimensions (traj_len, n_z_dim)
     act_seq: a tf.Tensor with dimensions (traj_len - 1, n_act_dim)
    Returns:
     a tf.Tensor with dimensions (traj_len - 1)
    """
    if init_state is not None:
      f = lambda x: tf.stop_gradient(tf.expand_dims(x, 0))
      init_state = tf.nn.rnn_cell.LSTMStateTuple(
          c=f(init_state.c), h=f(init_state.h))

    obs = traj[:-1, :]
    next_obs = traj[1:, :]

    dyn_out_vars = self.build_dyn_out_vars(
        tf.expand_dims(obs, 0),
        tf.expand_dims(act_seq, 0),
        init_state=init_state)

    output_x = tf.expand_dims(next_obs, 0)

    out_logmix = dyn_out_vars['out_logmix']
    if mixact_seq is None:
      trans_log_prob = -self.build_loss(out_logmix, dyn_out_vars['out_mean'],
                                        dyn_out_vars['out_logstd'],
                                        output_x)[0, :]
    else:
      traj_len = tf.shape(traj)[0]

      out_logmix = tf.reshape(
          out_logmix, [traj_len - 1, self.env.n_z_dim, self.num_mixture])
      out_logmix /= temperature
      out_logmix -= tf.reduce_logsumexp(out_logmix, axis=2, keep_dims=True)

      mixact_seq -= tf.reduce_logsumexp(mixact_seq, axis=2, keep_dims=True)

      trans_log_prob = tf.reduce_sum(tf.exp(mixact_seq) * out_logmix, axis=2)
      trans_log_prob = tf.reduce_mean(trans_log_prob, axis=1)

    return trans_log_prob

  def rnn_encode_traj(self, traj, act_seq, init_state=None):
    """Concatenate obses with hidden states (useful for rqst.traj_opt)
    Args:
     traj: a tf.Tensor with dimensions (traj_len, n_z_dim)
     act_seq: a tf.Tensor with dimensions (traj_len - 1, n_act_dim)
    Returns:
     a tf.Tensor with dimensions (traj_len, n_z_dim + 2*rnn_size)
    """
    if init_state is not None:
      init_state = tf.nn.rnn_cell.LSTMStateTuple(
          c=tf.expand_dims(init_state.c, 0), h=tf.expand_dims(init_state.h, 0))
    else:
      init_state = self.cell.zero_state(batch_size=1, dtype=tf.float32)

    dyn_out_vars = self.build_dyn_out_vars(
        tf.expand_dims(traj[:-1, :], 0),
        tf.expand_dims(act_seq, 0),
        init_state=init_state)

    c_states = dyn_out_vars['c_states'][0, :, :]
    h_states = dyn_out_vars['hidden_states'][0, :, :]
    c_states = tf.concat([init_state.c, c_states], axis=0)
    h_states = tf.concat([init_state.h, h_states], axis=0)

    hidden_states = tf.concat([c_states, h_states], axis=1)

    init_obses = tf.concat(
        [tf.expand_dims(traj[0, :], 0),
         tf.stop_gradient(hidden_states[:1, :])],
        axis=1)

    suffixes = tf.concat(
        [traj[1:, :], tf.stop_gradient(hidden_states[1:, :])], axis=1)

    return tf.concat([init_obses, suffixes], axis=0)

  def sample_seq_batch(self, size, rollout_data, idxes_key):
    batch = utils.sample_batch(
        size=size,
        data=rollout_data,
        data_keys=['obses', 'logvars', 'actions', 'traj_lens'],
        idxes_key=idxes_key)

    # stochastic samples
    z = batch['obses'] + np.exp(
        batch['logvars'] / 2.0) * np.random.randn(*(batch['logvars'].shape))

    # offset inputs/outputs by one timestep
    # use 0:t-1 to predict t
    inputs = np.concatenate((z[:, :-1, :], batch['actions'][:, :-1, :]), axis=2)
    outputs = z[:, 1:, :]
    traj_lens = batch['traj_lens'] - 1

    return {'inputs': inputs, 'outputs': outputs, 'traj_lens': traj_lens}

  def preproc_rollouts(self, raw_rollout_data):
    rollout_data = deepcopy(raw_rollout_data)

    for prefix in ['', 'next_']:
      raw_obses = raw_rollout_data['%sobses' % prefix]
      flat_raw_obses = raw_obses.reshape(
          (raw_obses.shape[0] * raw_obses.shape[1], raw_obses.shape[2],
           raw_obses.shape[3], raw_obses.shape[4]))

      mus = []
      logvars = []
      chunk_size = 512
      n_chunks = int(np.ceil(flat_raw_obses.shape[0] / chunk_size))
      for i in range(n_chunks):
        more_mus, more_logvars = self.encoder.encode_mu_logvar(
            flat_raw_obses[i * chunk_size:(i + 1) * chunk_size])
        mus.extend(more_mus)
        logvars.extend(more_logvars)
      mus = np.array(mus)
      logvars = np.array(logvars)

      rollout_data['%sobses' % prefix] = mus.reshape(
          (raw_obses.shape[0], raw_obses.shape[1], mus.shape[1]))
      rollout_data['%slogvars' % prefix] = logvars.reshape(
          (raw_obses.shape[0], raw_obses.shape[1], logvars.shape[1]))

    return rollout_data

  def format_batch(self, batch):
    """
    Args:
     batch: a dict containing the output of a call to
      rqst.utils.vectorize_rollouts(, preserve_trajs=True)
    """
    init_state = np.zeros((batch['traj_lens'].shape[0], self.rnn_size))
    feed_dict = {
        self.input_x_ph: batch['inputs'],
        self.output_x_ph: batch['outputs'],
        self.seq_lens_ph: batch['traj_lens'],
        self.initial_state_c_ph: init_state,
        self.initial_state_h_ph: init_state
    }
    return feed_dict

  def train(self,
            raw_rollout_data,
            iterations=1000,
            learning_rate=1e-3,
            ftol=1e-4,
            batch_size=32,
            val_update_freq=1,
            verbose=False):
    """
    Args:
     raw_rollout_data: a dict containing the output of a call to
      rqst.utils.vectorize_rollouts(, preserve_trajs=True)
      contains processed (but not yet encoded) frames
      raw_rollout_data['obses'] maps to a np.array with dimensions (n_trajs, traj_len, 64, 64, 3)
    """

    rollout_data = self.preproc_rollouts(raw_rollout_data)

    opt_scope = utils.opt_scope_of_obj(self)
    with tf.variable_scope(opt_scope, reuse=tf.AUTO_REUSE):
      optimizer = tf.train.AdamOptimizer(learning_rate)
      gvs = optimizer.compute_gradients(self.loss)
      capped_gvs = [
          (utils.tf_clip(grad, self.grad_clip), var) for grad, var in gvs
      ]
      global_step = tf.Variable(0, name='global_step', trainable=False)
      self.update_op = optimizer.apply_gradients(
          capped_gvs, global_step=global_step, name='train_step')

    utils.init_tf_vars(self.sess, [self.scope, opt_scope])

    val_losses = []
    val_batch = self.sample_seq_batch(
        len(rollout_data['val_idxes']), rollout_data, idxes_key='val_idxes')

    if verbose:
      print('iters total_iters train_loss val_loss')
    for t in range(iterations):
      batch = self.sample_seq_batch(
          batch_size, rollout_data, idxes_key='train_idxes')
      train_loss = self.compute_batch_loss(self.format_batch(batch), update=True)
      if t % val_update_freq == 0:
        val_loss = self.compute_batch_loss(self.format_batch(val_batch), update=False)
        if verbose:
          print('%d %d %f %f' % (t, iterations, train_loss, val_loss))
        val_losses.append(val_loss)
        if utils.converged(val_losses, ftol):
          break
    if verbose:
      plt.plot(val_losses)
      plt.show()

    if self.abs_model is not None:
      self.abs_model.train(
          utils.flatten_traj_data(self.rnn_encode_rollouts(rollout_data)),
          iterations=iterations,
          learning_rate=learning_rate,
          ftol=ftol,
          batch_size=batch_size,
          val_update_freq=val_update_freq,
          verbose=verbose)

  def next_obs(self,
               obs,
               act,
               init_state=None,
               temperature=0.1,
               mixact_seq=None):
    """
    Args:
     obs: a tf.Tensor with dimensions (batch_size, n_z_dim)
     act: a tf.Tensor with dimensions (batch_size, n_act_dim)
    Returns:
     a dict, where 'next_obs_mean' maps to a tf.Tensor with dimensions (batch_size, n_z_dim)
    """
    dyn_out_vars = self.build_dyn_out_vars(
        tf.expand_dims(obs, 1), tf.expand_dims(act, 1), init_state=init_state)

    traj_len = 1
    n_trajs = tf.shape(obs)[0]
    shape = [n_trajs, traj_len, self.env.n_z_dim, self.num_mixture]

    mean = tf.reshape(dyn_out_vars['out_mean'], shape)

    def f(logmix):
      logmix = tf.reshape(logmix, shape)
      logmix -= tf.reduce_logsumexp(logmix, axis=3, keep_dims=True)
      return tf.exp(logmix)

    if mixact_seq is not None:
      mix_coeffs = f(mixact_seq)
    else:
      mix_coeffs = f(dyn_out_vars['out_logmix'] / temperature)

    mixed_means = tf.reduce_sum(mix_coeffs * mean, axis=3)

    next_obs = mixed_means[:, -1, :]
    hidden_state = dyn_out_vars['last_state']
    return {'next_obs_mean': next_obs, 'hidden_state': hidden_state}

  def trajs_of_act_seqs(self,
                        init_obses,
                        act_seqs,
                        traj_len,
                        init_state=None,
                        mixact_seqs=None):
    """
    Args:
     init_obses: a tf.Tensor with dimensions (n_trajs, n_z_dim)
     act_seqs: a tf.Tensor with dimensions (n_trajs, traj_len - 1, n_act_dim)
    Returns:
     a tf.Tensor with dimensions (n_trajs, traj_len, n_z_dim)
    """
    trajs = [init_obses]
    obs = init_obses
    hidden_state = init_state
    for t in range(traj_len - 1):
      data = self.next_obs(
          obs,
          act_seqs[:, t, :],
          init_state=hidden_state,
          mixact_seq=(mixact_seqs[:, t, :, :]
                      if mixact_seqs is not None else None))
      obs = data['next_obs_mean']
      hidden_state = data['hidden_state']
      trajs.append(obs)
    trajs = tf.stack(trajs, axis=1)
    return {'trajs': trajs, 'hidden_state': hidden_state}

  def compute_next_obs(self, obs, act, init_state=None, temperature=0.1):
    """
    Args:
     trajs: a np.array with dimensions (n_trajs, traj_len, n_z_dim + 2*rnn_size)
     acts: a np.array with dimensions (n_trajs, traj_len - 1, n_act_dim)
     init_state: either None, or a tuple containing
      (a np.array with dimensions (n_trajs, 2*rnn_size),
       a np.array with dimensions (n_trajs, 2*rnn_size))
    Returns:
     a dict, where
      'next_obs' maps to a np.array with dimensions (n_trajs, n_z_dim + 2*rnn_size)
       that contains encoded frames concatenated with hidden states
      'next_state' maps to a tuple containing (a np.array with dimensions  (n_trajs, 2*rnn_size),
       a np.array with dimensions (n_trajs, 2*rnn_size)) that just contains hidden states
    """
    n_trajs = obs.shape[0]
    if init_state is None:
      init_state_c = init_state_h = np.zeros((n_trajs, self.env.rnn_size))
    else:
      init_state_c, init_state_h = init_state

    feed_dict = {
        self.obs_ph: obs,
        self.act_ph: act,
        self.initial_state_c_ph: init_state_c,
        self.initial_state_h_ph: init_state_h
    }

    next_obs_pred, next_state_pred = self.sess.run(
        [self.next_obs_pred, self.next_state_pred], feed_dict=feed_dict)

    return {
        'next_obs':
            np.concatenate(
                (next_obs_pred, next_state_pred.c, next_state_pred.h), axis=1),
        'next_state':
            next_state_pred
    }

  def rnn_encode_rollouts(self, rollout_data):
    """
    Args:
     rollout_data: a dict containing the output of a call to
      rqst.utils.vectorize_rollouts(, preserve_trajs=True)
      contains processed and encoded frames
      rollout_data['obses'] maps to a np.array with dimensions (n_trajs, traj_len, n_z_dim)
    Returns:
      a dict containing a copy of rollout_data, where obses and next_obses have been concatenated
       with hidden states
    """
    input_x = np.concatenate((rollout_data['obses'], rollout_data['actions']),
                             axis=2)
    seq_lens = rollout_data['traj_lens']
    init_state = np.zeros((len(seq_lens), self.rnn_size))

    feed_dict = {
        self.input_x_ph: input_x,
        self.seq_lens_ph: seq_lens,
        self.initial_state_c_ph: init_state,
        self.initial_state_h_ph: init_state
    }

    c_states, hidden_states = self.sess.run([self.c_states, self.hidden_states],
                                            feed_dict=feed_dict)
    hidden_states = np.concatenate((c_states, hidden_states), axis=2)
    hidden_states = np.concatenate((np.zeros(
        (hidden_states.shape[0], 1, hidden_states.shape[2])), hidden_states),
                                   axis=1)

    data = deepcopy(rollout_data)

    data['obses'] = np.zeros((data['obses'].shape[0], data['obses'].shape[1],
                              self.env.n_z_dim + 2 * self.env.rnn_size))

    data['next_obses'] = np.zeros(
        (data['next_obses'].shape[0], data['next_obses'].shape[1],
         self.env.n_z_dim + 2 * self.env.rnn_size))

    for rollout_idx in range(data['obses'].shape[0]):
      data['obses'][rollout_idx] = np.concatenate(
          (rollout_data['obses'][rollout_idx], hidden_states[rollout_idx, :-1]),
          axis=1)

      data['next_obses'][rollout_idx] = np.concatenate(
          (rollout_data['next_obses'][rollout_idx], hidden_states[rollout_idx,
                                                                  1:]),
          axis=1)

    return data

  def run_ep(self,
             policy,
             env,
             max_ep_len=None,
             render=False,
             store_raw_obs=False):
    """Useful for rqst.reward_opt.InteractiveRewardOptimizer.run and rqst.utils.evaluate_policy
    Args:
     store_raw_obs: a bool, where
      True -> store encoded frames concatenated with hidden states
      False -> use concatenated obses for evaluating the policy, but only store
       the encoded frames
    """
    if max_ep_len is None or max_ep_len > env.max_ep_len:
      max_ep_len = env.max_ep_len

    try:
      policy.reset()
    except:
      pass

    process_frame = utils.process_frame
    encode_frame = self.encoder.encode_frame

    raw_obs = process_frame(env.reset())

    obs = encode_frame(raw_obs)
    hidden_state = (np.zeros((1, env.rnn_size)), np.zeros((1, env.rnn_size)))
    obs = np.concatenate((obs, hidden_state[0][0, :], hidden_state[1][0, :]))

    done = False
    prev_obs = deepcopy(obs)
    prev_raw_obs = deepcopy(raw_obs)
    rollout = []
    for step_idx in range(max_ep_len):
      if done:
        break
      action = policy(obs)
      obs, r, done, info = env.step(action)
      obs = process_frame(obs)

      raw_obs = deepcopy(obs)

      obs = encode_frame(obs)
      hidden_state = self.compute_next_obs(
          prev_obs[np.newaxis, :self.env.n_z_dim],
          action[np.newaxis, :],
          init_state=hidden_state)['next_state']
      obs = np.concatenate((obs, hidden_state[0][0, :], hidden_state[1][0, :]))

      rollout.append((prev_raw_obs if store_raw_obs else obs, action, r,
                      raw_obs if store_raw_obs else obs, float(done), info))
      prev_obs = deepcopy(obs)
      prev_raw_obs = deepcopy(raw_obs)
      if render:
        env.render()
    return rollout


def load_wm_pretrained_rnn(encoder, sess, env):
  scope = 'mdn_rnn'
  dynamics_model = MDNRNNDynamicsModel(
      encoder,
      sess,
      env,
      learning_rate=0.0001,
      kl_tolerance=0.5,
      scope=scope,
      scope_file=os.path.join(utils.carracing_data_dir, 'dyn_scope.pkl'),
      tf_file=os.path.join(utils.carracing_data_dir, 'dyn.tf'))
  jsonfile = os.path.join(utils.wm_dir, 'rnn', 'rnn.json')
  utils.load_wm_pretrained_model(jsonfile, scope, sess)
  return dynamics_model
