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
"""Currently supported trajectory optimization algorithms:
 - Gradient descent
 - L-BFGS
 - Stochastic search over discrete set of policy rollouts
"""

from __future__ import division

import collections
import random
import uuid
import time

from matplotlib import pyplot as plt
import scipy
import tensorflow as tf
import numpy as np

from rqst import envs
from rqst import utils


class GDTrajOptimizer(object):

  def __init__(self,
               sess,
               env,
               reward_model,
               dynamics_model,
               traj_len=100,
               n_trajs=2,
               prior_coeff=1000.,
               diversity_coeff=10.,
               query_loss_opt='pref_uncertainty',
               opt_init_obs=False,
               opt_act_seq=True,
               join_trajs_at_init_state=False,
               traj_scope=None,
               shoot_steps=1,
               learning_rate=1e-4,
               query_type='sketch',
               using_mixact=False):

    if query_type not in ['sketch', 'pref', 'demo']:
      raise ValueError

    if traj_scope is None:
      traj_scope = str(uuid.uuid4())

    if traj_len < 2:
      raise ValueError

    if traj_len > env.max_ep_len + 1:
      traj_len = env.max_ep_len + 1

    if query_loss_opt == 'pref_uncertainty' and n_trajs != 2:
      raise ValueError

    if prior_coeff == np.inf:
      prior_coeff = 0.
      shoot_steps = None
      using_mixact = False

    self.sess = sess
    self.env = env
    self.reward_model = reward_model
    self.dynamics_model = dynamics_model
    self.traj_scope = traj_scope
    self.opt_init_obs = opt_init_obs
    self.opt_act_seq = opt_act_seq
    self.traj_len = traj_len
    self.query_type = query_type
    self.n_trajs = n_trajs
    self.query_loss_opt = query_loss_opt
    self.diversity_coeff = diversity_coeff
    self.prior_coeff = prior_coeff
    self.learning_rate = learning_rate
    self.join_trajs_at_init_state = join_trajs_at_init_state

    using_rnn_dyn = utils.isinstance(dynamics_model, 'MDNRNNDynamicsModel')
    self.obs_dim = self.env.n_z_dim if using_rnn_dyn else self.env.n_obs_dim

    if using_mixact and not (using_rnn_dyn and shoot_steps is None):
      raise ValueError

    self.init_obs_ph = tf.placeholder(tf.float32, [self.env.n_obs_dim])
    init_obs_ph = self.init_obs_ph[:self.obs_dim]
    if using_rnn_dyn:
      if opt_init_obs:
        init_c_ph = tf.zeros(self.env.rnn_size)
        init_h_ph = tf.zeros(self.env.rnn_size)
      else:
        init_c_ph = self.init_obs_ph[self.env.n_z_dim:self.env.n_z_dim +
                                     self.env.rnn_size]
        init_h_ph = self.init_obs_ph[-self.env.rnn_size:]
      init_hidden_state = tf.nn.rnn_cell.LSTMStateTuple(
          c=init_c_ph, h=init_h_ph)

      def proc_init(init_ph):
        init = tf.tile(init_ph, tf.expand_dims(tf.constant(self.n_trajs), 0))
        init = tf.reshape(init, [self.n_trajs, self.env.rnn_size])
        return init

      init_hidden_states = tf.nn.rnn_cell.LSTMStateTuple(
          c=proc_init(init_c_ph), h=proc_init(init_h_ph))
    else:
      init_hidden_state = None
      init_hidden_states = None

    with tf.variable_scope(self.traj_scope, reuse=tf.AUTO_REUSE) as scope_obj:
      obs_var = tf.get_variable(
          'obs_var',
          shape=[self.n_trajs, self.obs_dim],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer)

      traj_var = tf.get_variable(
          'traj_var',
          shape=[self.n_trajs, self.traj_len - 1, self.obs_dim],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer)

      if using_mixact:
        mixact_seq_var = tf.get_variable(
            'mixact_seq_var',
            shape=[
                self.n_trajs, self.traj_len - 1, self.env.n_z_dim,
                self.dynamics_model.num_mixture
            ],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer)

      if self.opt_act_seq:
        act_seq_var = tf.get_variable(
            'act_seq_var',
            shape=[self.n_trajs, self.traj_len - 1, self.env.n_act_dim],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer)
      else:
        self.act_seq_ph = tf.placeholder(
            tf.float32, [self.n_trajs, traj_len - 1, self.env.n_act_dim])
        act_seq_var = self.act_seq_ph

    self.init_traj_ph = tf.placeholder(tf.float32, [traj_len - 1, self.obs_dim])
    self.init_act_seq_ph = tf.placeholder(tf.float32,
                                          [traj_len - 1, self.env.n_act_dim])

    self.assign_init_traj = tf.assign(
        traj_var,
        tf.reshape(
            tf.tile(
                utils.unnormalize_obs(self.init_traj_ph, self.env),
                [self.n_trajs, 1]),
            [self.n_trajs, self.traj_len - 1, self.obs_dim]),
        validate_shape=True)

    if self.opt_act_seq:
      self.assign_init_act_seq = tf.assign(
          act_seq_var,
          tf.reshape(
              tf.tile(
                  utils.unnormalize_act(self.init_act_seq_ph, self.env),
                  [self.n_trajs, 1]),
              [self.n_trajs, self.traj_len - 1, self.env.n_act_dim]),
          validate_shape=True)

    obs_var = utils.normalize_obs(obs_var, self.env)
    traj_var = utils.normalize_obs(traj_var, self.env)
    act_seq_var = utils.normalize_act(act_seq_var, self.env)

    init_obses = []
    for traj_idx in range(self.n_trajs):
      if not self.opt_init_obs:
        init_obs = init_obs_ph
      else:
        if (self.join_trajs_at_init_state and
            traj_idx == 0) or (not self.join_trajs_at_init_state):
          init_obs = obs_var[traj_idx, :]
        else:
          init_obs = init_obses[0]
      init_obses.append(init_obs)
    init_obses = tf.stack(init_obses, axis=0)

    if shoot_steps is None:
      self.trajs = self.dynamics_model.trajs_of_act_seqs(
          init_obses,
          act_seq_var,
          traj_len,
          init_state=init_hidden_states,
          mixact_seqs=(mixact_seq_var if using_mixact else None))['trajs']
    elif shoot_steps > 1:
      n_shots = traj_len // shoot_steps
      trajs = []
      hidden_state = init_hidden_states
      for shot_idx in range(n_shots):
        t_start = shot_idx * shoot_steps
        if shot_idx == n_shots - 1:
          t_end = traj_len - 1
        else:
          t_end = min(t_start + shoot_steps, traj_len) - 1

        if shot_idx == 0:
          free_obses = init_obses
        else:
          free_obses = traj_var[:, t_start, :]

        shot = self.dynamics_model.trajs_of_act_seqs(
            free_obses,
            act_seq_var[:, t_start:t_end, :],
            t_end - t_start,
            init_state=hidden_state,
            mixact_seqs=(mixact_seq_var[:, t_start:t_end, :, :]
                         if using_mixact else None))
        hidden_state = shot.get('hidden_state', None)

        trajs.append(tf.expand_dims(free_obses, 1))
        trajs.append(shot['trajs'])
      self.trajs = tf.concat(trajs, axis=1)
    elif shoot_steps == 1:
      self.trajs = tf.concat([tf.expand_dims(init_obses, 1), traj_var], axis=1)

    self.trajs = tf.unstack(self.trajs, axis=0)
    self.act_seqs = tf.unstack(act_seq_var, axis=0)

    diversity_losses = []
    for i, ref_traj in enumerate(self.trajs):
      for traj in self.trajs[i + 1:]:
        diversity_losses.append(-self.traj_dist(ref_traj, traj))
    self.diversity_loss = 0. if diversity_losses == [] else tf.reduce_mean(
        diversity_losses)

    self.prior_loss = -tf.reduce_mean(
        tf.stack([
            self.dynamics_model.log_prob_rollout(
                traj,
                act_seq,
                init_state=init_hidden_state,
                mixact_seq=(mixact_seq_var[i, :, :, :]
                            if using_mixact else None))
            for i, (traj, act_seq) in enumerate(zip(self.trajs, self.act_seqs))
        ]))

    if using_rnn_dyn:
      # concatenate obses with hidden states
      self.trajs = [
          self.dynamics_model.rnn_encode_traj(
              traj, act_seq, init_state=init_hidden_state)
          for traj, act_seq in zip(self.trajs, self.act_seqs)
      ]

    if self.env.name == 'clfbandit' and self.query_loss_opt == 'pref_uncertainty':
      self.query_loss = self.max_imi_pol_uncertainty_query_loss(
          self.trajs, self.act_seqs)
    self.query_loss = eval('self.%s_query_loss(self.trajs, self.act_seqs)' %
                           self.query_loss_opt)

    self.loss = self.query_loss
    if self.diversity_coeff > 0:
      self.loss += self.diversity_coeff * self.diversity_loss
    if self.prior_coeff > 0:
      self.loss += self.prior_coeff * self.prior_loss

    # do this here instead of in run() for efficiency
    self.opt_scope = utils.opt_scope_of_obj(self)
    with tf.variable_scope(self.opt_scope, reuse=tf.AUTO_REUSE):
      var_list = utils.get_tf_vars_in_scope(self.traj_scope)
      try:
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss, var_list=var_list)
      except:
        self.update_op = None

      self.lbfgs_optimizer = tf.contrib.opt.ScipyOptimizerInterface(
          self.loss,
          method='L-BFGS-B',
          options={'ftol': 1e-9, 'disp': True},
          var_list=var_list
          )

  def traj_dist(self, ref_traj, traj):
    dist = tf.norm(ref_traj - traj, ord='euclidean', axis=1)
    dist_mult = 1. / tf.sqrt(tf.cast(ref_traj.shape[1], tf.float32))
    return -tf.reduce_mean(tf.exp(-dist_mult * dist))

  def unif_query_loss(self, *args):
    return 0.

  def max_imi_pol_uncertainty_query_loss(self, trajs, act_seqs):
    if not (self.reward_model.use_discrete_actions and
            utils.isinstance(self.reward_model, 'BCRewardModel')):
      raise ValueError

    obses = tf.concat(trajs, axis=0)
    votes = self.reward_model.build_vote_on_actions(obses)
    return -tf.reduce_mean(utils.ens_disag(votes))

  def max_nov_query_loss(self, trajs, act_seqs):
    if self.query_type == 'demo':
      data = self.reward_model.demo_data
    elif self.query_type == 'sketch':
      data = self.reward_model.sketch_data
    else:
      raise ValueError

    if data is None:
      raise ValueError

    if self.reward_model.rew_func_input == "s'":
      data_key = 'next_obses'
    else:
      data_key = 'obses'

    ref_obses = data[data_key]

    if self.env.name in ['pointmass', 'carracing']:
      max_ref_obses = 1000
      if len(ref_obses) > max_ref_obses:
        idxes = random.sample(list(range(len(ref_obses))), max_ref_obses)
        ref_obses = ref_obses[idxes, :]

    obses = tf.expand_dims(tf.concat(trajs, axis=0), 0)
    ref_obses = ref_obses[:, np.newaxis, :]

    if utils.isinstance(self.dynamics_model, 'MDNRNNDynamicsModel'):
      ref_obses = ref_obses[:, :, :self.env.n_z_dim]
      obses = obses[:, :, :self.env.n_z_dim]

    dist = tf.norm(ref_obses - obses, ord='euclidean', axis=2)
    dist_mult = 1. / tf.sqrt(tf.cast(ref_obses.shape[2], tf.float32))
    loss = tf.reduce_mean(tf.exp(-dist_mult * dist))
    return loss

  def max_rew_query_loss(self, trajs, act_seqs):
    return -tf.reduce_mean(
        tf.stack([
            self.reward_model.build_rew(traj[:-1, :], act_seq, traj[1:, :])
            for traj, act_seq in zip(trajs, act_seqs)
        ]))

  def min_rew_query_loss(self, *args, **kwargs):
    return -self.max_rew_query_loss(*args, **kwargs)

  def rew_uncertainty_query_loss(self, trajs, act_seqs):
    return -tf.reduce_mean(
        tf.stack([
            self.reward_model.build_uncertainty(traj[:-1, :], act_seq,
                                                traj[1:, :])
            for traj, act_seq in zip(trajs, act_seqs)
        ]))

  def pref_uncertainty_query_loss(self, trajs, act_seqs):
    if len(trajs) < 2:
      raise ValueError
    rew_preds = [
        tf.reduce_mean(
            self.reward_model.build_rew_ensemble(traj[:-1, :], act_seq,
                                                 traj[1:, :]),
            axis=0) for traj, act_seq in zip(trajs, act_seqs)
    ]
    rew_preds = tf.stack(
        rew_preds, axis=1)  # rew_preds[member_idx][traj_idx] = predicted reward
    pref_log_prob_distrns = rew_preds - tf.math.reduce_logsumexp(
        rew_preds, axis=1, keep_dims=True)
    # ... = member_idx's log-prob of traj_idx being preferred over all other trajs
    pref_log_prob_distrns = tf.expand_dims(pref_log_prob_distrns, 0)
    pref_prob_distrns = tf.exp(pref_log_prob_distrns)
    return -tf.reduce_mean(utils.ens_disag(pref_prob_distrns))

  def run(self, **kwargs):

    if self.env.name == 'carracing' and kwargs.get('init_act_seq',
                                                   None) is None:
      make_init_act_seq = lambda act: np.array(
          [act for _ in range(self.traj_len - 1)])
      steer_mag = 1.
      acc_mag = 0.2
      init_acts = [[0., 0., 0.], [0., acc_mag, 0.], [0., 0., acc_mag],
                   [steer_mag, acc_mag, 0.], [-steer_mag, acc_mag, 0.],
                   [steer_mag, 0., acc_mag], [-steer_mag, 0., acc_mag],
                   [steer_mag, 0., 0.], [-steer_mag, 0., 0.]]
      init_act_seqs = [make_init_act_seq(init_act) for init_act in init_acts]
      init_act_seqs.append(None)
      best_eval = None
      for init_act_seq in init_act_seqs:
        data = self._run(init_act_seq=init_act_seq, **kwargs)
        if best_eval is None or data['loss'] < best_eval['loss']:
          best_eval = data
      return best_eval
    elif self.env.name == ('clfbandit' and
                           self.query_loss_opt == 'pref_uncertainty'):
      data = self._run(**kwargs)
      assert self.reward_model.use_discrete_actions
      assert self.join_trajs_at_init_state
      votes = self.reward_model.vote_on_actions(data['traj'][0])[0, :, :]
      votes = np.argmax(votes, axis=1)
      n_votes_of_act = collections.Counter(votes)
      actions = sorted(
          list(n_votes_of_act.items()), key=lambda x: x[1], reverse=True)[:2]
      if len(actions) == 1:
        actions.append(actions[0])
      data['act_seq'] = [
          utils.onehot_encode(action, self.env.n_act_dim)[np.newaxis, :] *
          utils.inf for action, _ in actions
      ]
      return data
    else:
      return self._run(**kwargs)

  def _run(
      self,
      init_obs=None,
      act_seq=None,
      iterations=10000,
      ftol=1e-6,
      min_iters=2,
      verbose=False,
      warm_start=False,
      init_with_lbfgs=False,
      init_act_seq=None,
      init_traj=None,
  ):

    if (init_obs is not None) == self.opt_init_obs:
      raise ValueError

    if (act_seq is not None) == self.opt_act_seq:
      raise ValueError

    if act_seq is not None and init_act_seq is not None:
      raise ValueError

    if init_act_seq is not None and warm_start:
      raise ValueError

    if self.query_loss_opt == 'unif':
      if self.env.name == 'clfbandit':
        std = np.exp(-self.prior_coeff)

        def rand_traj():
          obs = np.random.normal(0, std, self.env.n_z_dim)[np.newaxis, :]
          next_obs = self.env.absorbing_state[np.newaxis, :]
          return np.concatenate((obs, next_obs), axis=0)

        trajs_eval = [rand_traj() for _ in range(self.n_trajs)]
        act_seqs_eval = [
            [self.env.action_space.sample()] for _ in range(self.n_trajs)
        ]
      elif self.env.name == 'pointmass':
        unif_env = envs.make_pointmass_env()
        random_policy = utils.make_random_policy(unif_env)
        unif_rollouts = [
            utils.run_ep(random_policy, unif_env, max_ep_len=1)
            for _ in range(self.n_trajs)
        ]
        trajs_eval = [
            utils.traj_of_rollout(rollout) for rollout in unif_rollouts
        ]
        act_seqs_eval = [
            utils.act_seq_of_rollout(rollout) for rollout in unif_rollouts
        ]
      else:
        raise ValueError
      loss_eval = 0.
      return {'traj': trajs_eval, 'act_seq': act_seqs_eval, 'loss': loss_eval}

    scopes = [self.opt_scope]
    if not warm_start:
      scopes.append(self.traj_scope)
    utils.init_tf_vars(self.sess, scopes, use_cache=True)

    feed_dict = {}
    assign_ops = []
    if init_act_seq is not None:
      feed_dict[self.init_act_seq_ph] = init_act_seq
      assign_ops.append(self.assign_init_act_seq)
    if init_traj is not None:
      self.obs_dim = (
          self.env.n_z_dim
          if self.env.name == 'carracing' else self.env.n_obs_dim)
      feed_dict[self.init_traj_ph] = init_traj[1:, :self.obs_dim]
      assign_ops.append(self.assign_init_traj)
    if assign_ops != []:
      self.sess.run(assign_ops, feed_dict=feed_dict)

    feed_dict = {}
    if init_obs is not None:
      feed_dict[self.init_obs_ph] = init_obs() if callable(
          init_obs) else init_obs
    if act_seq is not None:
      feed_dict[self.act_seq_ph] = act_seq

    if verbose:
      print('iters loss')

    if init_with_lbfgs:
      self.lbfgs_optimizer.minimize(self.sess, feed_dict=feed_dict)

    loss_evals = []
    loss_eval, trajs_eval, act_seqs_eval = self.sess.run(
        [self.loss, self.trajs, self.act_seqs], feed_dict=feed_dict)
    best_eval = {
        'traj': trajs_eval,
        'act_seq': act_seqs_eval,
        'loss': loss_eval
    }
    #start_time = time.time() # uncomment for profiling
    for t in range(iterations):
      loss_eval, trajs_eval, act_seqs_eval, _ = self.sess.run(
          [self.loss, self.trajs, self.act_seqs, self.update_op],
          feed_dict=feed_dict)

      if verbose:
        print('%d %f' % (t, loss_eval))

      loss_evals.append(loss_eval)

      if loss_eval < best_eval['loss']:
        best_eval = {
            'traj': trajs_eval,
            'act_seq': act_seqs_eval,
            'loss': loss_eval
        }

      if ftol is not None and utils.converged(
          loss_evals, ftol, min_iters=min_iters):
        break
    # uncomment for profiling
    #print('call to update_op: %0.3f' % ((time.time() - start_time) / t))
    #print('iterations: %d' % t)

    if verbose:
      plt.plot(loss_evals)
      plt.show()

    return best_eval


class MPCAgent(object):
  """Model-predictive control with a gradient descent planner
  Replan every n_blind_steps steps
  """

  def __init__(self,
               sess,
               env,
               reward_model,
               dynamics_model,
               plan_horizon=10,
               n_blind_steps=1,
               test_mode=False,
               query_loss_opt='max_rew'):

    if n_blind_steps > plan_horizon:
      raise ValueError

    self.env = env
    self.dynamics_model = dynamics_model
    self.plan_horizon = plan_horizon
    self.test_mode = test_mode
    self.n_blind_steps = n_blind_steps

    if not self.test_mode and self.env.name == 'carracing':
      prior_coeff = 0.
      shoot_steps = None
    else:
      prior_coeff = 10000.
      shoot_steps = 1

    self.traj_optimizer = GDTrajOptimizer(
        sess,
        self.env,
        reward_model,
        self.dynamics_model,
        traj_len=(self.plan_horizon + 1),
        n_trajs=1,
        prior_coeff=prior_coeff,
        diversity_coeff=0.,
        query_loss_opt=query_loss_opt,
        opt_init_obs=False,
        learning_rate=1e-2,
        shoot_steps=shoot_steps)

    self.reset()

  def reset(self):
    self.steps = 0
    self.plan = None

  def __call__(self, obs):
    if self.steps % self.n_blind_steps == 0:
      #start_time = time.time() # uncomment for profiling
      if self.env.name == 'carracing':
        iterations = 0
      elif self.test_mode:
        iterations = 1
      else:
        iterations = 3000
      data = self.traj_optimizer.run(
          iterations=iterations,
          ftol=1e-4,
          verbose=False,
          init_obs=obs,
          warm_start=(self.plan is not None and self.env.name == 'pointmass'))
      #print('call to mpc policy: %0.3f' % (time.time() - start_time)) # uncomment for profiling
      self.plan = data['act_seq'][0][:self.n_blind_steps]
      self.steps = 0
    self.steps += 1
    return self.plan[self.steps - 1]


class MyopicAgent(object):

  def __init__(self, reward_model):
    self.reward_model = reward_model

  def __call__(self, obs):
    votes = self.reward_model.vote_on_actions(obs[np.newaxis, :])[0, :, :]
    vote_distrn = np.mean(votes, axis=0)
    return np.log(1e-9 + vote_distrn)


def make_imitation_policy(sess,
                          env,
                          reward_model,
                          dynamics_model,
                          query_loss_opt='max_rew',
                          *args,
                          **kwargs):
  if utils.isinstance(reward_model,
                      'BCRewardModel') and reward_model.use_discrete_actions:
    assert query_loss_opt == 'max_rew'
    return MyopicAgent(reward_model)
  else:
    return MPCAgent(
        sess, env, reward_model, dynamics_model, query_loss_opt=query_loss_opt, **kwargs)


class StochTrajOptimizer(object):

  def __init__(self,
               sess,
               env,
               reward_model,
               dynamics_model,
               traj_len=100,
               rollout_len=None,
               query_loss_opt='pref_uncertainty',
               imitation_kwargs={},
               use_rand_policy=False,
               query_type='sketch',
               guided_search=False):

    if query_type not in ['pref', 'sketch', 'demo']:
      raise ValueError

    if traj_len > env.max_ep_len + 1:
      traj_len = env.max_ep_len + 1

    if rollout_len is None:
      rollout_len = env.max_ep_len

    if traj_len > rollout_len + 1:
      raise ValueError

    if use_rand_policy and guided_search:
      raise ValueError

    if guided_search and query_loss_opt == 'pref_uncertainty':
      raise ValueError

    self.query_loss = eval('self.%s_query_loss' % query_loss_opt)

    self.sess = sess
    self.query_type = query_type
    self.env = env
    self.reward_model = reward_model
    self.dynamics_model = dynamics_model
    self.traj_len = traj_len
    self.rollout_len = rollout_len
    self.use_rand_policy = use_rand_policy
    self.guided_search = guided_search
    self.imitation_kwargs = imitation_kwargs
    self.query_loss_opt = query_loss_opt

    if self.use_rand_policy:
      self.imitator = utils.make_random_policy(self.env)
    else:
      query_loss_opt_for_pol = (
          query_loss_opt if self.guided_search else 'max_rew')
      if 'plan_horizon' not in self.imitation_kwargs:
        self.imitation_kwargs['plan_horizon'] = self.rollout_len
      self.imitator = make_imitation_policy(
          self.sess,
          self.env,
          self.reward_model,
          self.dynamics_model,
          query_loss_opt=query_loss_opt_for_pol,
          **self.imitation_kwargs)

  def unif_query_loss(self, traj, act_seq):
    return np.random.random()

  def max_imi_pol_uncertainty_query_loss(self, traj, act_seq):
    if not (self.reward_model.use_discrete_actions and
            utils.isinstance(self.reward_model, 'BCRewardModel')):
      raise ValueError

    votes = self.reward_model.vote_on_actions(traj)
    return -np.mean(utils.np_ens_disag(votes))

  def max_nov_query_loss(self, traj, act_seq):
    if self.query_type == 'demo':
      data = self.reward_model.demo_data
    elif self.query_type == 'sketch':
      data = self.reward_model.sketch_data
    else:
      raise ValueError

    if data is None:
      raise ValueError

    if self.reward_model.rew_func_input == "s'":
      data_key = 'next_obses'
    else:
      data_key = 'obses'

    ref_obses = data[data_key]
    ref_obses = ref_obses[:, np.newaxis, :]
    obses = traj[np.newaxis, :, :]

    if utils.isinstance(self.dynamics_model, 'MDNRNNDynamicsModel'):
      ref_obses = ref_obses[:, :, :self.env.n_z_dim]
      obses = obses[:, :, :self.env.n_z_dim]

    dist = np.linalg.norm(ref_obses - obses, axis=2)
    dist_mult = 1. / np.sqrt(ref_obses.shape[2])
    loss = np.mean(np.exp(-dist_mult * dist))
    return loss

  def max_rew_query_loss(self, traj, act_seq):
    return -np.mean(
        self.reward_model.compute_rew_of_transes(traj[:-1, :], act_seq,
                                                 traj[1:, :]))

  def min_rew_query_loss(self, *args, **kwargs):
    return -self.max_rew_query_loss(*args, **kwargs)

  def rew_uncertainty_query_loss(self, traj, act_seq):
    return -sum(
        self.reward_model.compute_uncertainty_of_transes(
            traj[:-1, :], act_seq, traj[1:, :]))

  def pref_uncertainty_query_loss(self, trajs, act_seqs):
    if len(trajs) < 2:
      raise ValueError

    rew_preds = []
    for traj, act_seq in zip(trajs, act_seqs):
      rew_preds.append(
          self.reward_model.compute_rew_ens_preds_of_transes(
              traj[:-1, :], act_seq, traj[1:, :]).mean(axis=0))
    rew_preds = np.array(
        rew_preds)  # rew_preds[traj_idx][member_idx] = predicted reward
    pref_log_prob_distrns = rew_preds - scipy.misc.logsumexp(
        rew_preds, axis=0
    )  # ... = member_idx's prob of traj_idx being preferred over all other trajs
    pref_prob_distrns = np.exp(pref_log_prob_distrns)
    pref_prob_distrns = np.swapaxes(pref_prob_distrns, 0, 1)
    pref_prob_distrns = pref_prob_distrns[np.newaxis, :, :]
    return -np.mean(utils.np_ens_disag(pref_prob_distrns))

  def run(self,
          n_trajs=10,
          act_seq=None,
          n_samples=None,
          init_obs=None,
          verbose=False):
    """
    Args:
     n_samples: num rollouts to sample from env
     n_trajs: num trajs to return
      n_trajs <= n_samples
    """
    if n_samples is None:
      n_samples = n_trajs

    if self.query_loss_opt == 'pref_uncertainty' and not (n_trajs == 2 and
                                                          n_samples >= 2):
      raise ValueError

    if n_trajs > n_samples:
      raise ValueError

    clfbandit_pref = (
        self.query_loss_opt == 'pref_uncertainty' and
        self.env.name == 'clfbandit')

    if init_obs is None:
      # rollout in real world
      rollouts = [
          self.dynamics_model.run_ep(
              self.imitator,
              self.env,
              max_ep_len=self.rollout_len,
              store_raw_obs=False) for _ in range(n_samples)
      ]
      trajs = [utils.traj_of_rollout(rollout) for rollout in rollouts]
      act_seqs = [utils.act_seq_of_rollout(rollout) for rollout in rollouts]
      if clfbandit_pref:
        n_classes = self.env.n_act_dim
        if act_seq is None:
          new_trajs = []
          act_seqs = []
          for i in range(n_classes):
            act_seq = utils.onehot_encode(i,
                                          n_classes)[np.newaxis, :] * utils.inf
            new_trajs.extend(trajs)
            act_seqs.extend([act_seq] * len(trajs))
          trajs = new_trajs
        else:
          act_seqs = [act_seq] * len(trajs)
    else:
      if self.env.name.endswith('bandit'):
        raise ValueError

      # rollout in dynamics model
      dream_env = utils.DreamEnv(self.env, self.dynamics_model)
      dream_rollouts = [
          utils.run_ep(self.imitator, dream_env, max_ep_len=self.rollout_len)
          for _ in range(n_samples)
      ]
      trajs = [utils.traj_of_rollout(rollout) for rollout in dream_rollouts]
      act_seqs = [
          utils.act_seq_of_rollout(rollout) for rollout in dream_rollouts
      ]

    trajs, act_seqs = utils.segment_trajs(
        trajs, act_seqs, seg_len=self.traj_len)

    if self.query_loss_opt != 'pref_uncertainty':
      # find best trajs
      idxes = [(i, self.query_loss(traj, act_seq))
               for i, (traj, act_seq) in enumerate(zip(trajs, act_seqs))]
      sorted_idxes = sorted(idxes, key=lambda x: x[1])
      best_trajs = [trajs[i] for i, _ in sorted_idxes[:n_trajs]]
      best_act_seqs = [act_seqs[i] for i, _ in sorted_idxes[:n_trajs]]
      best_loss = sum(x[1] for x in sorted_idxes[:n_trajs])
    else:
      # find best traj pair
      best_ref_idx = None
      best_idx = None
      best_loss = None
      for ref_idx, (ref_traj, ref_act_seq) in enumerate(zip(trajs, act_seqs)):
        for idx, (traj, act_seq) in enumerate(
            zip(trajs[ref_idx + 1:], act_seqs[ref_idx + 1:])):
          if clfbandit_pref and (ref_traj[0] != traj[0]).any():
            continue
          loss = self.query_loss([ref_traj, traj], [ref_act_seq, act_seq])
          if best_loss is None or loss < best_loss:
            best_loss = loss
            best_ref_idx = ref_idx
            best_idx = idx + ref_idx + 1
      best_trajs = [trajs[best_ref_idx], trajs[best_idx]]
      best_act_seqs = [act_seqs[best_ref_idx], act_seqs[best_idx]]

    return {'traj': best_trajs, 'act_seq': best_act_seqs, 'loss': best_loss}
