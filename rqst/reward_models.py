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
"""Currently supported reward models:
 - Supervised learning with ground-truth rewards
 - Random Expert Distillation
 - Bradley-Terry, learning from pairwise prefs
Currently supported uncertainty models:
 - Deep ensemble
"""

from __future__ import division

import pickle
from collections import defaultdict, Counter
import random
import uuid

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy

from rqst.models import TFModel
from rqst import traj_opt
from rqst import utils


class RewardModel(TFModel):
  """Ensembled reward model
  Only difference between members of ensemble are random seeds used for training
   (same data, architecture, etc.)
  """

  def __init__(self,
               *args,
               n_rew_nets_in_ensemble=4,
               n_layers=1,
               layer_size=64,
               rew_func_input="s'",
               use_discrete_actions=False,
               use_discrete_rewards=False,
               data_file=None,
               **kwargs):

    super().__init__(*args, **kwargs)

    if use_discrete_actions and use_discrete_rewards:
      raise ValueError

    if rew_func_input not in ['s', 'sa', "s'"]:
      raise ValueError

    if self.env.name.endswith('bandit') and not use_discrete_actions:
      raise ValueError

    if use_discrete_actions and rew_func_input != 'sa':
      raise ValueError

    if self.env.name in ['pointmass', 'carracing'] and not use_discrete_rewards:
      raise ValueError

    self.n_rew_nets_in_ensemble = n_rew_nets_in_ensemble
    self.n_layers = n_layers
    self.layer_size = layer_size
    self.rew_func_input = rew_func_input
    self.use_discrete_actions = use_discrete_actions
    self.use_discrete_rewards = use_discrete_rewards
    self.data_file = data_file

    self.member_mask_ph = tf.placeholder(tf.float32,
                                         [1, self.n_rew_nets_in_ensemble])

    # for sketches
    self.obs_ph = tf.placeholder(tf.float32, [None, self.env.n_obs_dim])
    self.act_ph = tf.placeholder(tf.float32, [None, self.env.n_act_dim])
    self.rew_ph = tf.placeholder(tf.float32, [None])
    self.next_obs_ph = tf.placeholder(tf.float32, [None, self.env.n_obs_dim])
    self.sketch_bal_weights_ph = tf.placeholder(tf.float32, [None])

    self.sketch_loss = self.build_sketch_loss()

    # for demos
    self.demo_obs_ph = tf.placeholder(tf.float32, [None, self.env.n_obs_dim])
    self.demo_act_ph = tf.placeholder(tf.float32, [None, self.env.n_act_dim])
    self.demo_bal_weights_ph = tf.placeholder(tf.float32, [None])

    self.demo_loss = None  # may be defined in subclass __init__

    # for prefs
    self.pref_ph = tf.placeholder(tf.float32, [None])  # binary
    self.traj_ph = tf.placeholder(
        tf.float32,
        [None, None, self.env.n_obs_dim])  # [batch_size, traj_len, ...]
    self.ref_traj_ph = tf.placeholder(tf.float32,
                                      [None, None, self.env.n_obs_dim])
    self.act_seq_ph = tf.placeholder(
        tf.float32,
        [None, None, self.env.n_act_dim])  # [batch_size, traj_len - 1, ...]
    self.ref_act_seq_ph = tf.placeholder(tf.float32,
                                         [None, None, self.env.n_act_dim])
    self.mask_ph = tf.placeholder(tf.float32, [None, None])
    self.ref_mask_ph = tf.placeholder(tf.float32, [None, None])

    self.pref_loss = self.build_pref_loss()

    self.loss = None
    self.update_op = None

    self.rew = None
    self.rew_uncertainty = None
    self.rew_ens_preds = None

    self.raw = None
    self.raw_ens_preds = None

    self.demo_data = None
    self.sketch_data = None

    self.votes = None

  def build_sketch_loss(self):
    if self.use_discrete_rewards:
      masks = [
          tf.equal(self.rew_ph, rew_class) for rew_class in self.env.rew_classes
      ]
      labels = tf.cast(tf.stack(masks, axis=1), tf.float32)

      ens = self.build_raw_ensemble(self.obs_ph, self.act_ph, self.next_obs_ph)
      logits = tf.reduce_sum(
          tf.expand_dims(self.member_mask_ph, 2) * ens, axis=1)

      xent = utils.tf_xent(labels, logits)
      unweighted_loss = xent
    else:
      ens = self.build_rew_ensemble(self.obs_ph, self.act_ph, self.next_obs_ph)
      rew_pred_of_member = tf.reduce_sum(self.member_mask_ph * ens, axis=1)
      sq_err = (self.rew_ph - rew_pred_of_member)**2
      unweighted_loss = sq_err
    sketch_loss = tf.reduce_sum(
        self.sketch_bal_weights_ph * unweighted_loss) / tf.reduce_sum(
            self.sketch_bal_weights_ph)
    return sketch_loss

  def build_pref_loss(self):
    traj_shape = tf.shape(self.traj_ph)
    tf_batch_size = traj_shape[0]
    traj_len = traj_shape[1]

    flat_obs_shape = [tf_batch_size * (traj_len - 1), self.env.n_obs_dim]
    flat_act_shape = [tf_batch_size * (traj_len - 1), self.env.n_act_dim]

    def flatten(traj_ph, act_seq_ph):
      obses = tf.reshape(traj_ph[:, :-1, :], flat_obs_shape)
      acts = tf.reshape(act_seq_ph, flat_act_shape)
      next_obses = tf.reshape(traj_ph[:, 1:, :], flat_obs_shape)
      return obses, acts, next_obses

    flat_rew_of_traj = self.build_rew(*flatten(self.traj_ph, self.act_seq_ph))
    flat_rew_of_ref_traj = self.build_rew(
        *flatten(self.ref_traj_ph, self.ref_act_seq_ph))

    unflat_shape = [tf_batch_size, traj_len - 1]
    rew_of_traj = tf.reduce_sum(
        tf.reshape(flat_rew_of_traj, unflat_shape) * self.mask_ph, axis=1)
    rew_of_ref_traj = tf.reduce_sum(
        tf.reshape(flat_rew_of_ref_traj, unflat_shape) * self.ref_mask_ph,
        axis=1)

    z = tf.math.reduce_logsumexp(
        tf.stack([rew_of_traj, rew_of_ref_traj], axis=1), axis=1)
    log_pref_prob = rew_of_traj - z
    log_unpref_prob = rew_of_ref_traj - z

    pref_loss = -tf.reduce_mean(self.pref_ph * log_pref_prob +
                                (1 - self.pref_ph) * log_unpref_prob)
    return pref_loss

  def build_rew_ensemble(self, obs, act, next_obs):
    ens = [
        self.build_rew_net_member(obs, act, next_obs, member_idx)
        for member_idx in range(self.n_rew_nets_in_ensemble)
    ]
    ens = tf.stack(ens, axis=1)
    return ens

  def build_rew_mean(self, *args):
    return tf.reduce_mean(self.build_rew_ensemble(*args), axis=1)

  def build_rew_var(self, *args):
    return tf.math.reduce_variance(self.build_rew_ensemble(*args), axis=1)

  def build_rew(self, *args, **kwargs):
    return self.build_rew_mean(*args, **kwargs)

  def build_rew_uncertainty(self, *args, **kwargs):
    return self.build_rew_var(*args, **kwargs)

  def build_raw_ensemble(self, obs, act, next_obs):
    ens = [
        self.build_raw_net_member(obs, act, next_obs, member_idx)
        for member_idx in range(self.n_rew_nets_in_ensemble)
    ]
    ens = tf.stack(ens, axis=1)
    return ens

  def build_raw_mean(self, *args):
    return tf.reduce_mean(self.build_raw_ensemble(*args), axis=1)

  def build_clf_disag(self, *args):
    logits = self.build_raw_ensemble(*args)
    probs = tf.nn.softmax(logits, axis=2)
    return utils.ens_disag(probs)

  def build_raw(self, *args, **kwargs):
    return self.build_raw_mean(*args, **kwargs)

  def build_clf_uncertainty(self, *args, **kwargs):
    return self.build_clf_disag(*args, **kwargs)

  def build_uncertainty(self, *args, **kwargs):
    if self.use_discrete_rewards:
      return self.build_clf_uncertainty(*args, **kwargs)
    else:
      return self.build_rew_uncertainty(*args, **kwargs)

  def bal_weights_of_batch(self, batch_elts):
    batch_size = len(batch_elts)
    weights = np.ones(batch_size)
    idxes_of_elt = defaultdict(list)
    for idx, elt in enumerate(batch_elts):
      idxes_of_elt[elt].append(idx)
    n_classes = len(idxes_of_elt)
    for elt, idxes in idxes_of_elt.items():
      weights[idxes] = batch_size / (n_classes * len(idxes))
    return weights

  def format_batch(self, demo_batch, sketch_batch, pref_batch, member_mask):
    feed_dict = {self.member_mask_ph: member_mask[np.newaxis, :]}
    if demo_batch is not None:
      feed_dict.update({
          self.demo_obs_ph: demo_batch['obses'],
          self.demo_act_ph: demo_batch['actions']
      })
      if self.use_discrete_actions:
        feed_dict[self.demo_bal_weights_ph] = utils.bal_weights(
            np.argmax(demo_batch['actions'], axis=1))
      else:
        feed_dict[self.demo_bal_weights_ph] = np.ones(len(demo_batch['obses']))
    if sketch_batch is not None:
      feed_dict.update({
          self.obs_ph: sketch_batch['obses'],
          self.act_ph: sketch_batch['actions'],
          self.next_obs_ph: sketch_batch['next_obses'],
          self.rew_ph: sketch_batch['rews']
      })
      if self.use_discrete_rewards:
        feed_dict[self.sketch_bal_weights_ph] = utils.bal_weights(
            sketch_batch['rews'])
      else:
        feed_dict[self.sketch_bal_weights_ph] = np.ones(
            len(sketch_batch['obses']))
    if pref_batch is not None:
      feed_dict.update({
          self.ref_traj_ph: pref_batch['ref_trajs'],
          self.traj_ph: pref_batch['trajs'],
          self.ref_act_seq_ph: pref_batch['ref_act_seqs'],
          self.act_seq_ph: pref_batch['act_seqs'],
          self.pref_ph: pref_batch['prefs'],
          self.mask_ph: pref_batch['mask'],
          self.ref_mask_ph: pref_batch['ref_mask']
      })
    return feed_dict

  def train(self,
            demo_data=None,
            sketch_data=None,
            pref_data=None,
            demo_coeff=1.,
            sketch_coeff=1.,
            iterations=100000,
            ftol=1e-4,
            batch_size=512,
            learning_rate=1e-3,
            val_update_freq=100,
            verbose=False,
            warm_start=False):
    """
    Args:
     demo_data: output of a call to rqst.utils.split_rollouts
     sketch_data: output of a call to rqst.utils.split_rollouts
     pref_data: output of a call to rqst.utils.split_prefs
    """
    if demo_data is None and pref_data is None and sketch_data is None:
      raise ValueError

    self.demo_data = demo_data
    self.sketch_data = sketch_data

    self.build_ensemble_outputs()

    pref_loss = self.pref_loss if pref_data is not None and self.pref_loss is not None else 0
    demo_loss = self.demo_loss if demo_data is not None and self.demo_loss is not None else 0
    sketch_loss = self.sketch_loss if sketch_data is not None and self.sketch_loss is not None else 0

    if pref_loss == 0 and demo_loss == 0 and sketch_loss == 0:
      raise ValueError

    self.loss = demo_coeff * demo_loss + sketch_coeff * sketch_loss + pref_loss

    opt_scope = utils.opt_scope_of_obj(self)
    with tf.variable_scope(opt_scope, reuse=tf.AUTO_REUSE):
      self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    scopes = [opt_scope]
    if not warm_start:
      scopes.append(self.scope)
    utils.init_tf_vars(self.sess, scopes)

    sketch_data_keys = ['obses', 'actions', 'next_obses', 'rews']
    demo_data_keys = ['obses', 'actions']
    pref_data_keys = [
        'ref_trajs', 'ref_act_seqs', 'trajs', 'act_seqs', 'prefs', 'mask',
        'ref_mask'
    ]

    unif_member_mask = np.ones(
        self.n_rew_nets_in_ensemble) / self.n_rew_nets_in_ensemble
    val_losses = []

    if pref_data is None:
      val_pref_batch = None
    else:
      val_pref_batch = utils.sample_batch(
          size=len(pref_data['val_idxes']),
          data=pref_data,
          data_keys=pref_data_keys,
          idxes_key='val_idxes')

    if demo_data is None:
      val_demo_batch = None
    else:
      val_demo_batch = utils.sample_batch(
          size=len(demo_data['val_idxes']),
          data=demo_data,
          data_keys=demo_data_keys,
          idxes_key='val_idxes')

    if sketch_data is None:
      val_sketch_batch = None
    else:
      val_sketch_batch = utils.sample_batch(
          size=len(sketch_data['val_idxes']),
          data=sketch_data,
          data_keys=sketch_data_keys,
          idxes_key='val_idxes')

    pref_batch = None
    demo_batch = None
    sketch_batch = None

    member_masks = [
        utils.onehot_encode(member_idx, self.n_rew_nets_in_ensemble)
        for member_idx in range(self.n_rew_nets_in_ensemble)
    ]

    bootstrap_prob = 1.

    def bootstrap(train_idxes, mem_idx):
      guar_idxes = [
          x for i, x in enumerate(train_idxes)
          if i % self.n_rew_nets_in_ensemble == mem_idx
      ]
      nonguar_idxes = [
          x for i, x in enumerate(train_idxes)
          if i % self.n_rew_nets_in_ensemble != mem_idx
      ]
      n_train_per_mem = int(np.ceil(bootstrap_prob * len(nonguar_idxes)))
      return guar_idxes + random.sample(nonguar_idxes, n_train_per_mem)

    train_idxes_key_of_mem = []
    for mem_idx in range(self.n_rew_nets_in_ensemble):
      train_idxes_key = 'train_idxes_of_mem_%d' % mem_idx
      train_idxes_key_of_mem.append(train_idxes_key)
      if demo_data is not None:
        if self.use_discrete_actions:
          train_idxes_of_act_key = 'train_idxes_of_act_of_mem_%d' % mem_idx
          demo_data[train_idxes_of_act_key] = {}
          for c, idxes_of_c in demo_data['train_idxes_of_act'].items():
            demo_data[train_idxes_of_act_key][c] = bootstrap(
                idxes_of_c, mem_idx)
          demo_data[train_idxes_key] = sum(
              (v for v in demo_data[train_idxes_of_act_key].values()), [])
        else:
          demo_data[train_idxes_key] = bootstrap(demo_data['train_idxes'],
                                                 mem_idx)
      if pref_data is not None:
        pref_data[train_idxes_key] = bootstrap(pref_data['train_idxes'],
                                               mem_idx)
      if sketch_data is not None:
        if self.use_discrete_rewards:
          train_idxes_of_rew_class_key = 'train_idxes_of_rew_class_of_mem_%d' % mem_idx
          sketch_data[train_idxes_of_rew_class_key] = {}
          for c, idxes_of_c in sketch_data['train_idxes_of_rew_class'].items():
            sketch_data[train_idxes_of_rew_class_key][c] = bootstrap(
                idxes_of_c, mem_idx)
          sketch_data[train_idxes_key] = sum(
              (v for v in sketch_data[train_idxes_of_rew_class_key].values()),
              [])
        else:
          sketch_data[train_idxes_key] = bootstrap(sketch_data['train_idxes'],
                                                   mem_idx)

    if verbose:
      print('iters total_iters train_loss val_loss')

    #best_val_loss = None # uncomment to save model with lowest val loss
    for t in range(iterations):
      for mem_idx, member_mask in enumerate(member_masks):
        if demo_data is not None:
          if self.use_discrete_actions:
            class_idxes_key = 'train_idxes_of_act_of_mem_%d' % mem_idx
          else:
            class_idxes_key = None
          demo_batch = utils.sample_batch(
              size=batch_size,
              data=demo_data,
              data_keys=demo_data_keys,
              idxes_key=train_idxes_key_of_mem[mem_idx],
              class_idxes_key=class_idxes_key)

        if sketch_data is not None:
          if self.use_discrete_rewards:
            class_idxes_key = 'train_idxes_of_rew_class_of_mem_%d' % mem_idx
          else:
            class_idxes_key = None
          sketch_batch = utils.sample_batch(
              size=batch_size,
              data=sketch_data,
              data_keys=sketch_data_keys,
              idxes_key=train_idxes_key_of_mem[mem_idx],
              class_idxes_key=class_idxes_key)

        if pref_data is not None:
          pref_batch = utils.sample_batch(
              size=batch_size,
              data=pref_data,
              data_keys=pref_data_keys,
              idxes_key=train_idxes_key_of_mem[mem_idx])

        formatted_batch = self.format_batch(demo_batch, sketch_batch, pref_batch, member_mask)
        train_loss = self.compute_batch_loss(formatted_batch, update=True)

      if t % val_update_freq == 0:
        formatted_batch = self.format_batch(val_demo_batch, val_sketch_batch,
            val_pref_batch, unif_member_mask)
        val_loss = self.compute_batch_loss(formatted_batch, update=False)

        if verbose:
          print('%d %d %f %f' % (t, iterations, train_loss, val_loss))

        val_losses.append(val_loss)

        # uncomment to save model checkpoint if it achieves lower val loss
        #if best_val_loss is None or val_loss < best_val_loss:
        #  best_val_loss = val_loss
        #  self.save()

        if utils.converged(val_losses, ftol):
          break

    if verbose:
      plt.plot(val_losses)
      plt.show()

    #self.load() # uncomment to load model checkpoint with lowest val loss

  def compute_quant_of_transes(self, quant, obses, actions, next_obses):
    feed_dict = {
        self.obs_ph: obses,
        self.act_ph: actions,
        self.next_obs_ph: next_obses
    }
    feed_dict = {
        k: v
        for k, v in feed_dict.items()
        if v is not None and (type(v) != list or all(x is not None for x in v))
    }
    return self.sess.run(quant, feed_dict=feed_dict)

  def compute_raw_of_transes(self, *args):
    return self.compute_quant_of_transes(self.raw, *args)

  def compute_clf_uncertainty_of_transes(self, *args):
    return self.compute_quant_of_transes(self.clf_uncertainty, *args)

  def compute_raw_ens_preds_of_transes(self, *args):
    return self.compute_quant_of_transes(self.raw_ens_preds, *args)

  def compute_rew_of_transes(self, *args):
    return self.compute_quant_of_transes(self.rew, *args)

  def compute_rew_uncertainty_of_transes(self, *args):
    return self.compute_quant_of_transes(self.rew_uncertainty, *args)

  def compute_rew_ens_preds_of_transes(self, *args):
    return self.compute_quant_of_transes(self.rew_ens_preds, *args)

  def compute_uncertainty_of_transes(self, *args):
    if self.use_discrete_rewards:
      return self.compute_clf_uncertainty_of_transes(*args)
    else:
      return self.compute_rew_uncertainty_of_transes(*args)

  def save(self):
    super().save()
    if self.data_file is not None:
      with open(self.data_file, 'wb') as f:
        pickle.dump((self.demo_data, self.sketch_data), f,
                    pickle.HIGHEST_PROTOCOL)

  def load(self):
    super().load()
    self.build_ensemble_outputs()
    if self.data_file is not None:
      with open(self.data_file, 'rb') as f:
        self.demo_data, self.sketch_data = pickle.load(f)

  def init_tf_vars(self, *args, **kwargs):
    super().init_tf_vars(*args, **kwargs)
    self.build_ensemble_outputs()

  def build_ensemble_outputs(self):
    args = [self.obs_ph, self.act_ph, self.next_obs_ph]
    self.rew = self.build_rew(*args)
    self.rew_uncertainty = self.build_rew_uncertainty(*args)
    self.rew_ens_preds = self.build_rew_ensemble(*args)

    self.raw = self.build_raw(*args)
    self.clf_uncertainty = self.build_clf_uncertainty(*args)
    self.raw_ens_preds = self.build_raw_ensemble(*args)

  def build_rew_net_input(self, obs, act, next_obs):
    if self.rew_func_input == 's' or self.use_discrete_actions:
      assert obs is not None
      return obs
    elif self.rew_func_input == 'sa':
      assert act is not None
      return tf.concat([obs, act], axis=1)
    elif self.rew_func_input == "s'":
      assert next_obs is not None
      return next_obs
    else:
      raise ValueError

  def build_rew_net_args(self, obs, act, next_obs):
    rew_net_input = self.build_rew_net_input(obs, act, next_obs)
    kwargs = {
        'n_layers': self.n_layers,
        'size': self.layer_size,
        'activation': tf.nn.relu,
        'output_activation': None
    }
    return rew_net_input, kwargs

  def build_raw_net(self, obs, act, next_obs, scope):
    rew_net_input, kwargs = self.build_rew_net_args(obs, act, next_obs)
    if self.use_discrete_actions:
      n_outputs = self.env.n_act_dim
    elif self.use_discrete_rewards:
      n_outputs = len(self.env.rew_classes)
    else:
      n_outputs = 1
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      outputs = utils.build_mlp(rew_net_input, n_outputs, scope, **kwargs)
    return outputs

  def build_rew_net(self, obs, act, next_obs, scope):
    outputs = self.build_raw_net(obs, act, next_obs, scope)
    if self.use_discrete_actions:
      output = tf.reduce_sum(
          tf.nn.softmax(act, axis=1) * utils.tf_normalize_logits(outputs),
          axis=1)
    elif self.use_discrete_rewards:
      output = tf.reduce_sum(
          self.env.rew_classes[np.newaxis, :] * tf.nn.softmax(outputs, axis=1),
          axis=1)
    else:
      output = tf.squeeze(outputs, axis=[1])
    return output

  def scope_of_member(self, member_idx):
    return 'ens-mem-%d' % member_idx

  def build_vote_on_actions(self, obs):
    assert self.use_discrete_actions
    logits = self.build_raw_ensemble(obs, None, None)
    votes = tf.nn.softmax(logits, axis=2)
    return votes

  def vote_on_actions(self, obs):
    if self.votes is None:
      self.votes = self.build_vote_on_actions(self.obs_ph)
    return self.sess.run(self.votes, feed_dict={self.obs_ph: obs})

  def build_rew_net_member(self, obs, act, next_obs, member_idx):
    return self.build_rew_net(obs, act, next_obs,
                              self.scope_of_member(member_idx))

  def build_raw_net_member(self, obs, act, next_obs, member_idx):
    return self.build_raw_net(obs, act, next_obs,
                              self.scope_of_member(member_idx))

  def viz_learned_rew(self, unc_save_path=None, pred_save_path=None, pred_title=None):
    if self.sketch_data is not None and self.use_discrete_rewards:
      print('sketch rew class distrn: %s' %
            str(Counter(self.sketch_data['rews'])))
    if self.demo_data is not None and self.use_discrete_actions:
      print('demo act class distrn: %s' %
            str(Counter(np.argmax(self.demo_data['actions'], axis=1))))
    if self.env.name in ['bandit', 'pointmass']:
      res = 100
      lows = self.env.observation_space.low
      highs = self.env.observation_space.high
      ranges = highs - lows
      xs, ys = np.meshgrid(
          np.arange(lows[0], highs[0], ranges[0] / res),
          np.flip(np.arange(lows[1], highs[1], ranges[1] / res)))
      xs = xs.ravel()
      ys = ys.ravel()
      obses = np.array(list(zip(xs, ys)))
      feed_dict = {self.obs_ph: obses, self.next_obs_ph: obses}

      acts = None
      if self.env.name == 'bandit':
        acts = np.tile(np.array([0, 1]), obses.shape[0]).reshape(
            (obses.shape[0], self.env.n_act_dim))
        feed_dict[self.act_ph] = acts

      rew_uncs = self.compute_uncertainty_of_transes(obses, acts, obses)
      rew_means = self.compute_rew_of_transes(obses, acts, obses)

      plt.title('Reward Uncertainty')
      sc = plt.imshow(rew_uncs.reshape((res, res)), cmap=plt.cm.Blues)
      plt.axis('off')
      plt.colorbar(sc)
      if unc_save_path is not None:
        plt.savefig(
            unc_save_path, bbox_inches='tight', dpi=500, transparent=True)
      plt.show()

      if pred_title is None:
        pred_title = 'Predicted Reward'
      plt.title(pred_title)
      sc = plt.imshow(rew_means.reshape((res, res)), cmap=plt.cm.RdYlGn)
      plt.axis('off')
      plt.colorbar(sc)
      if self.env.rew_classes is not None:
        plt.clim(self.env.rew_classes[0], self.env.rew_classes[2])
      if pred_save_path is not None:
        plt.savefig(
            pred_save_path, bbox_inches='tight', dpi=500, transparent=True)
      plt.show()

      confs = []
      if self.sketch_data is not None:
        confs.append((self.sketch_data, 'Sketches', self.sketch_data['rews']))
      if self.demo_data is not None and self.env.name == 'bandit':
        confs.append((self.demo_data, 'Demos',
                      np.argmax(self.demo_data['actions'], axis=1)))

      if self.rew_func_input == "s'":
        data_key = 'next_obses'
      else:
        data_key = 'obses'

      for data, title, colors in confs:
        xs = data[data_key][:, 0]
        ys = data[data_key][:, 1]

        plt.title('Training %s' % title)
        plt.scatter(xs, ys, c=colors, alpha=0.5, linewidth=0)
        eps = 0.05
        plt.xlim([lows[0] - eps, highs[0] + eps])
        plt.ylim([lows[1] - eps, highs[1] + eps])
        if self.env.name == 'bandit':
          plt.grid()
        plt.show()

  def discretize_rewards(self, rews):
    assert self.env.rew_classes is not None
    rews = np.array(rews)
    return np.array([rews == r for r in self.env.rew_classes]).astype(float).T

  def discretize_actions(self, acts):
    acts = np.array(acts)
    return np.array([
        utils.onehot_encode(np.argmax(acts[i, :]), self.env.n_act_dim)
        for i in range(acts.shape[0])
    ]).astype(float)


class REDRewardModel(RewardModel):
  """Random Expert Distillation (https://arxiv.org/abs/1905.06750)"""

  def __init__(self, *args, **kwargs):
    self.sigma_one = kwargs.get('sigma_one', 1)

    super().__init__(*args, **kwargs)

    if self.rew_func_input not in ['s', 'sa']:
      raise ValueError

    ens = self.build_pred_ensemble(self.demo_obs_ph, self.demo_act_ph, None)
    target_pred_of_member = tf.reduce_sum(self.member_mask_ph * ens, axis=1)
    target = self.build_target_net(self.demo_obs_ph, self.demo_act_ph, None)
    err = (tf.stop_gradient(target) - target_pred_of_member)**2
    self.demo_loss = tf.reduce_sum(
        self.demo_bal_weights_ph * err) / tf.reduce_sum(
            self.demo_bal_weights_ph)

  def build_target_net(self, obs, act, next_obs):
    return self.build_rew_net(obs, act, next_obs, 'target')

  def build_pred_net_member(self, obs, act, next_obs, member_idx):
    return self.build_rew_net(obs, act, next_obs,
                              self.scope_of_member(member_idx))

  def build_pred_ensemble(self, obs, act, next_obs):
    ens = [
        self.build_pred_net_member(obs, act, next_obs, member_idx)
        for member_idx in range(self.n_rew_nets_in_ensemble)
    ]
    ens = tf.stack(ens, axis=1)
    return ens

  def build_rew_net_member(self, obs, act, next_obs, member_idx):
    target_pred = self.build_pred_net_member(obs, act, next_obs, member_idx)
    target = self.build_target_net(obs, act, next_obs)
    return tf.exp(-self.sigma_one * (target_pred - target)**2)


class BCRewardModel(RewardModel):
  """Behavioral cloning
  r(s,a) = log(p(a|s))
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    if self.rew_func_input not in ['s', 'sa']:
      raise ValueError

    if self.use_discrete_rewards or not self.use_discrete_actions:
      raise ValueError

    if not self.env.name.endswith('bandit'):
      raise ValueError

    ens = self.build_raw_ensemble(self.demo_obs_ph, self.demo_act_ph, None)
    raw_outputs_of_member = tf.reduce_sum(
        tf.expand_dims(self.member_mask_ph, 2) * ens, axis=1)
    if self.use_discrete_actions:
      demo_act_probs = tf.nn.softmax(self.demo_act_ph, axis=1)
      losses = utils.tf_xent(demo_act_probs, raw_outputs_of_member)
    else:
      losses = (self.demo_act_ph - raw_outputs_of_member)**2
    self.demo_loss = tf.reduce_sum(
        self.demo_bal_weights_ph * losses) / tf.reduce_sum(
            self.demo_bal_weights_ph)


def compute_rew_err_metrics(rollouts, reward_model):
  metrics = {}
  obses, acts, true_rews, next_obses = list(zip(*sum(rollouts, [])))[:4]
  ens_uncs = reward_model.compute_uncertainty_of_transes(
      obses, acts, next_obses)
  metrics['ens_unc'] = np.mean(ens_uncs)
  pred_rews = reward_model.compute_rew_of_transes(obses, acts, next_obses)
  try:
    corr = scipy.stats.spearmanr(true_rews, pred_rews)[0]
  except:
    corr = np.nan
  if reward_model.use_discrete_rewards or reward_model.use_discrete_actions:
    pred_logits = utils.normalize_logits(
        reward_model.compute_raw_of_transes(obses, acts, next_obses))
    if reward_model.use_discrete_rewards:
      labels = reward_model.discretize_rewards(true_rews)
      weights = utils.bal_weights(true_rews)
    elif reward_model.use_discrete_actions:
      labels = reward_model.discretize_actions(acts)
      weights = utils.bal_weights(np.argmax(acts, axis=1))
    else:
      weights = np.ones(len(pred_logits))
    xent = -(pred_logits * labels).sum(axis=1)
    metrics['xent'] = (xent * weights).sum() / weights.sum()
    ent = -(np.exp(pred_logits) * pred_logits).sum(axis=1)
    metrics['ent'] = (ent * weights).sum() / weights.sum()
    pred_disc_rews = np.argmax(pred_logits, axis=1)
    true_disc_rews = np.argmax(labels, axis=1)
    acc = (pred_disc_rews == true_disc_rews).astype(float)
    metrics['acc'] = (acc * weights).sum() / weights.sum()
    if reward_model.use_discrete_rewards:
      metrics['tpr'] = (
          (pred_disc_rews == 2) &
          (true_disc_rews == 2)).sum() / (true_disc_rews == 2).sum()
      metrics['tnr'] = (
          (pred_disc_rews == 0) &
          (true_disc_rews == 0)).sum() / (true_disc_rews == 0).sum()
      metrics['fpr'] = (
          (pred_disc_rews == 2) &
          (true_disc_rews != 2)).sum() / (pred_disc_rews == 2).sum()
      metrics['fnr'] = (
          (pred_disc_rews == 0) &
          (true_disc_rews != 0)).sum() / (pred_disc_rews == 0).sum()
    n_rew_classes = labels.shape[1]
    metrics['conf'] = np.array([[
        ((pred_disc_rews == i) & (true_disc_rews == j)).sum()
        for j in range(n_rew_classes)
    ]
                                for i in range(n_rew_classes)])
  else:
    metrics['err'] = np.mean((true_rews - pred_rews)**2) / np.var(true_rews)
  data = {'true_rews': true_rews, 'pred_rews': pred_rews}
  return metrics, data


def evaluate_reward_model(sess,
                          env,
                          trans_env,
                          reward_model,
                          dynamics_model,
                          offpol_eval_rollouts=None,
                          n_eval_rollouts=10,
                          imitator=None,
                          imitation_kwargs={}):

  if imitator is None:
    imitator = traj_opt.make_imitation_policy(sess, env, reward_model,
                                              dynamics_model,
                                              **imitation_kwargs)

  onpol_eval = utils.evaluate_policy(
      sess,
      env,
      trans_env,
      dynamics_model,
      imitator,
      n_eval_rollouts=n_eval_rollouts,
      dream=False)

  perf = onpol_eval['perf']
  onpol_eval_rollouts = onpol_eval['rollouts']

  rollouts = offpol_eval_rollouts if offpol_eval_rollouts is not None else onpol_eval_rollouts
  if rollouts != []:
    rew_err_metrics, rew_pred_data = compute_rew_err_metrics(
        rollouts, reward_model)
    perf.update(rew_err_metrics)
  else:
    rew_pred_data = None

  return {
      'perf': perf,
      'rollouts': onpol_eval_rollouts,
      'rew_pred_data': rew_pred_data
  }


def synth_demo(traj, demo_policy, proc_obses=None):
  traj = traj[:-1]
  obses = proc_obses(traj) if proc_obses is not None else traj
  return [(orig_obs, demo_policy(obs), None, None, None, None)
          for orig_obs, obs in zip(traj, obses)]


def synth_sketch(traj, act_seq, reward_func):
  sketch = reward_func(traj[:-1], act_seq, traj[1:])
  return [(s, a, r, ns, None, None)
          for s, a, r, ns in zip(traj[:-1], act_seq, sketch, traj[1:])]


def synth_pref(ref_traj, ref_act_seq, traj, act_seq, reward_func):
  # sample preference from bradley-terry model using true reward
  rew_of_ref = sum(reward_func(ref_traj[:-1], ref_act_seq, ref_traj[1:]))
  rew = sum(reward_func(traj[:-1], act_seq, traj[1:]))
  logits = np.array([rew_of_ref, rew])  # unnormalized logits
  return np.argmax(logits)  # + np.random.gumbel(0, 1, 2)) # uncomment to use gumbel-max trick


def autolabel_prefs(rollouts, env, segment_len=None):
  if segment_len is None:
    segment_len = env.max_ep_len + 1

  if segment_len > env.max_ep_len + 1:
    segment_len = env.max_ep_len + 1

  pref_logs = {
      'ref_trajs': [],
      'ref_act_seqs': [],
      'trajs': [],
      'act_seqs': [],
      'prefs': []
  }

  for i, ref_rollout in enumerate(rollouts):
    ref_traj = utils.traj_of_rollout(ref_rollout)
    ref_act_seq = utils.act_seq_of_rollout(ref_rollout)
    curr_ref_seg_len = min(segment_len, len(ref_traj))
    for j in range(len(ref_traj) - curr_ref_seg_len + 1):
      for rollout in rollouts[i + 1:]:
        traj = utils.traj_of_rollout(rollout)
        act_seq = utils.act_seq_of_rollout(rollout)
        curr_seg_len = min(segment_len, len(traj))
        for k in range(len(traj) - curr_seg_len + 1):
          ref_seg = ref_traj[j:j + curr_ref_seg_len]
          ref_act_seg = ref_act_seq[j:j + curr_ref_seg_len - 1]
          seg = traj[k:k + curr_seg_len]
          act_seg = act_seq[k:k + curr_seg_len - 1]
          pref_logs['ref_trajs'].append(ref_seg)
          pref_logs['ref_act_seqs'].append(ref_act_seg)
          pref_logs['trajs'].append(seg)
          pref_logs['act_seqs'].append(act_seg)
          pref_logs['prefs'].append(
              synth_pref(ref_seg, ref_act_seg, seg, act_seg, env.reward_func))

  return pref_logs
