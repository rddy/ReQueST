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

from __future__ import division

from copy import deepcopy
import pickle
import time
import os

import tensorflow as tf
import numpy as np

from rqst.reward_models import REDRewardModel, BCRewardModel
from rqst.traj_opt import GDTrajOptimizer, StochTrajOptimizer
from rqst.dynamics_models import MDNRNNDynamicsModel
from rqst import reward_models
from rqst import traj_opt
from rqst import utils

default_reward_train_kwargs = {
    'demo_coeff': 1.,
    'sketch_coeff': 1.,
    'iterations': 1,
    'ftol': 1e-4,
    'batch_size': 512,
    'learning_rate': 1e-3,
    'val_update_freq': 100,
    'verbose': False
}

default_dynamics_train_kwargs = {
    'iterations': 1,
    'batch_size': 512,
    'learning_rate': 1e-3,
    'ftol': 1e-4,
    'val_update_freq': 100,
    'verbose': False
}

default_gd_traj_opt_init_kwargs = {
    'traj_len': 100,
    'n_trajs': 2,
    'prior_coeff': 1000.,
    'diversity_coeff': 10.,
    'query_loss_opt': 'pref_uncertainty',
    'opt_init_obs': False,
    'join_trajs_at_init_state': False
}

default_gd_traj_opt_run_kwargs = {
    'iterations': 1,
    'ftol': 1e-4,
    'learning_rate': 1e-2,
    'verbose': False,
    'warm_start': False
}

default_stoch_traj_opt_init_kwargs = {
    'traj_len': 100,
    'rollout_len': 100,
    'query_loss_opt': 'pref_uncertainty'
}

default_stoch_traj_opt_run_kwargs = {
    'n_traj': 2,
    'n_samples': 2,
    'init_obs': None,
    'verbose': False
}

default_traj_opt_cls = GDTrajOptimizer
default_traj_opt_run_kwargs = default_gd_traj_opt_run_kwargs
default_traj_opt_init_kwargs = default_gd_traj_opt_init_kwargs

default_imitation_kwargs = {'plan_horizon': 10, 'n_blind_steps': 1}

default_eval_kwargs = {'n_eval_rollouts': 10}


class InteractiveRewardOptimizer(object):

  def __init__(self, sess, env, trans_env, reward_model, dynamics_model):

    self.sess = sess
    self.env = env
    self.trans_env = trans_env
    self.reward_model = reward_model
    self.dynamics_model = dynamics_model

  def run(self,
          demo_rollouts=None,
          sketch_rollouts=None,
          pref_logs=None,
          rollouts_for_dyn=[],
          reward_train_kwargs=None,
          dynamics_train_kwargs=None,
          traj_opt_cls=None,
          traj_opt_run_kwargs=None,
          traj_opt_init_kwargs=None,
          imitation_kwargs=None,
          eval_kwargs=None,
          init_train_dyn=False,
          init_train_rew=False,
          n_imitation_rollouts_per_dyn_update=1,
          n_queries=1000,
          reward_update_freq=1,
          reward_eval_freq=1,
          dyn_update_freq=1,
          verbose=True,
          warm_start_rew=False,
          query_type='pref',
          callback=None):

    if query_type not in ['pref', 'sketch', 'demo']:
      raise ValueError

    if query_type == 'demo' and not any(
        utils.isinstance(self.reward_model, reward_model_cls_name)
        for reward_model_cls_name in ['REDRewardModel', 'BCRewardModel']):
      raise ValueError

    if reward_train_kwargs is None:
      reward_train_kwargs = default_reward_train_kwargs

    if dynamics_train_kwargs is None:
      dynamics_train_kwargs = default_dynamics_train_kwargs

    if traj_opt_cls is None:
      traj_opt_cls = default_traj_opt_cls

    if traj_opt_run_kwargs is None:
      traj_opt_run_kwargs = default_traj_opt_run_kwargs

    if traj_opt_init_kwargs is None:
      traj_opt_init_kwargs = default_traj_opt_init_kwargs

    if type(traj_opt_init_kwargs) != type(traj_opt_run_kwargs):
      raise ValueError

    if type(traj_opt_init_kwargs
           ) == dict and 'opt_init_obs' in traj_opt_init_kwargs and (
               not traj_opt_init_kwargs['opt_init_obs']
           ) and 'init_obs' not in traj_opt_run_kwargs:
      raise ValueError

    if imitation_kwargs is None:
      imitation_kwargs = default_imitation_kwargs

    if eval_kwargs is None:
      eval_kwargs = default_eval_kwargs
    eval_kwargs['imitation_kwargs'] = imitation_kwargs

    if type(traj_opt_init_kwargs) == list:
      for i in range(len(traj_opt_init_kwargs)):
        traj_opt_init_kwargs[i]['query_type'] = query_type
    else:
      traj_opt_init_kwargs['query_type'] = query_type

    if verbose:
      print('initializing reward and dynamics models...')

    demo_rollouts = deepcopy(demo_rollouts)
    sketch_rollouts = deepcopy(sketch_rollouts)
    pref_logs = deepcopy(pref_logs)

    def proc_rollouts(rollouts, traj_len=None):
      if rollouts is None:
        return None
      else:
        # for padding
        max_len = max(len(rollout) for rollout in rollouts)
        max_len = min(
            max_len if traj_len is None else max(traj_len - 1, max_len),
            self.env.max_ep_len)
        return utils.split_rollouts(utils.vectorize_rollouts(rollouts, max_len))

    proc_pref_logs = lambda pref_logs: None if pref_logs is None else utils.split_prefs(
        pref_logs)

    demo_data = proc_rollouts(demo_rollouts)
    sketch_data = proc_rollouts(sketch_rollouts)
    pref_data = proc_pref_logs(pref_logs)

    if init_train_rew:
      self.reward_model.train(
          demo_data=demo_data,
          sketch_data=sketch_data,
          pref_data=pref_data,
          **reward_train_kwargs)
    else:
      self.reward_model.init_tf_vars()

    using_rnn_dyn = utils.isinstance(self.dynamics_model, 'MDNRNNDynamicsModel')
    proc_dyn_rollouts = lambda rollouts: utils.split_rollouts(
        utils.vectorize_rollouts(
            rollouts, self.env.max_ep_len, preserve_trajs=using_rnn_dyn))

    if init_train_dyn:
      self.dynamics_model.train(
          proc_dyn_rollouts(rollouts_for_dyn), **dynamics_train_kwargs)

    if pref_logs is None and query_type == 'pref':
      pref_logs = {
          'ref_trajs': [],
          'trajs': [],
          'ref_act_seqs': [],
          'act_seqs': [],
          'prefs': []
      }
    if sketch_rollouts is None:
      sketch_rollouts = []
    if demo_rollouts is None:
      demo_rollouts = []

    make_traj_optimizer = lambda kwargs: traj_opt_cls(
        self.sess, self.env, self.reward_model, self.dynamics_model, **kwargs)

    if type(traj_opt_init_kwargs) != list:
      traj_opt_run_kwargs = [traj_opt_run_kwargs]
      traj_opt_init_kwargs = [traj_opt_init_kwargs]

    imitator = traj_opt.make_imitation_policy(self.sess, self.env,
                                              self.reward_model,
                                              self.dynamics_model,
                                              **imitation_kwargs)

    def update_rew_perf(rew_perf_evals, n_queries_made):
      rew_eval = reward_models.evaluate_reward_model(
          self.sess,
          self.env,
          self.trans_env,
          self.reward_model,
          self.dynamics_model,
          imitator=imitator,
          **eval_kwargs)

      rew_perf = rew_eval['perf']
      rew_perf['n_queries'] = n_queries_made
      rew_perf['n_real_rollouts'] = len(
          rollouts_for_dyn) + n_real_rollouts_from_traj_opt

      if verbose:
        print(
            '\n'.join(['%s: %s' % (k, str(v)) for k, v in rew_perf.items()]),
            flush=True)
        # uncomment to plot learned rewards
        #utils.viz_rew_eval(rew_eval, self.env, encoder=self.dynamics_model.encoder)

      if rew_perf_evals == {}:
        rew_perf_evals = {k: [] for k in rew_perf}
      for k, v in rew_perf.items():
        rew_perf_evals[k].append(v)

      return rew_perf_evals

    if verbose:
      print('initializing traj optimizers...')

    traj_optimizers = [
        make_traj_optimizer(kwargs) for kwargs in traj_opt_init_kwargs
    ]

    if query_type == 'demo' and self.dynamics_model.encoder is not None:
      proc_obses = self.dynamics_model.encoder.decode_batch_latents
    else:
      proc_obses = None

    if verbose:
      print('evaluating reward model...')

    n_queries_made = 0
    n_real_rollouts_from_traj_opt = 0
    rew_perf_evals = update_rew_perf({}, n_queries_made)

    if verbose:
      print('')

    iter_idx = 0

    if type(traj_opt_run_kwargs[0]['init_obs']) == list:
      init_obses = traj_opt_run_kwargs[0]['init_obs']
    else:
      init_obses = None
      traj_opt_run_kwargs_update = {}

    while n_queries_made < n_queries:
      start_time = time.time()
      if verbose:
        print('iter %d' % iter_idx)
        print('synthesizing queries...')

      query_trajs = []
      query_act_seqs = []
      if init_obses is not None:
        traj_opt_run_kwargs_update = {
            'init_obs': init_obses[iter_idx % len(init_obses)]
        }
      for traj_optimizer, kwargs in zip(traj_optimizers, traj_opt_run_kwargs):
        kwargs.update(traj_opt_run_kwargs_update)
        data = traj_optimizer.run(**kwargs)
        query_trajs.extend(data['traj'])
        query_act_seqs.extend(data['act_seq'])

      if utils.isinstance(
          traj_optimizers[0],
          'StochTrajOptimizer') and traj_opt_run_kwargs[0]['init_obs'] is None:
        assert len(traj_opt_run_kwargs) == 1
        n_real_rollouts_from_traj_opt += traj_opt_run_kwargs[0]['n_samples']

      if verbose:
        print('eliciting feedback...')

      if query_type == 'pref':
        if len(query_trajs) != 2:
          raise ValueError
        pref = reward_models.synth_pref(query_trajs[0], query_act_seqs[0],
                                        query_trajs[1], query_act_seqs[1],
                                        self.env.reward_func)
        pref_logs['ref_trajs'].append(query_trajs[0])
        pref_logs['ref_act_seqs'].append(query_act_seqs[0])
        pref_logs['trajs'].append(query_trajs[1])
        pref_logs['act_seqs'].append(query_act_seqs[1])
        pref_logs['prefs'].append(pref)
        n_queries_made += 1
      elif query_type == 'sketch':
        sketches = [
            reward_models.synth_sketch(traj, act_seq, self.env.reward_func)
            for traj, act_seq in zip(query_trajs, query_act_seqs)
        ]
        sketch_rollouts.extend(sketches)
        n_queries_made += sum(len(sketch) for sketch in sketches)
      elif query_type == 'demo':
        demos = [
            reward_models.synth_demo(
                traj, self.env.expert_policy, proc_obses=proc_obses)
            for traj in query_trajs
        ]
        demo_rollouts.extend(demos)
        n_queries_made += sum(len(demo) for demo in demos)

      if verbose:
        query_data = {
            'demo_rollouts': demo_rollouts,
            'sketch_rollouts': sketch_rollouts,
            'pref_logs': pref_logs
        }
        utils.viz_query_data(
            query_data, self.env, encoder=self.dynamics_model.encoder)

      update_reward = iter_idx % reward_update_freq == 0
      if update_reward:
        if verbose:
          print('updating reward model...')

        if query_type == 'pref':
          pref_data = proc_pref_logs(pref_logs)
        elif query_type == 'demo':
          demo_data = proc_rollouts(
              demo_rollouts, traj_len=traj_optimizers[0].traj_len)
        elif query_type == 'sketch':
          sketch_data = proc_rollouts(
              sketch_rollouts, traj_len=traj_optimizers[0].traj_len)

        self.reward_model.train(
            demo_data=demo_data,
            sketch_data=sketch_data,
            pref_data=pref_data,
            warm_start=warm_start_rew,
            **reward_train_kwargs)

        if verbose:
          self.reward_model.viz_learned_rew()

      update_dynamics = dyn_update_freq is not None and iter_idx % dyn_update_freq == 0
      if update_dynamics:
        if verbose:
          print('updating dynamics model...')

        rollouts_for_dyn += [
            self.dynamics_model.run_ep(
                imitator, self.env, store_raw_obs=using_rnn_dyn)
            for _ in range(n_imitation_rollouts_per_dyn_update)
        ]
        self.dynamics_model.train(
            proc_dyn_rollouts(rollouts_for_dyn), **dynamics_train_kwargs)

      if (iter_idx + 1) % reward_eval_freq == 0:
        if verbose:
          print('evaluating reward model...')

        rew_perf_evals = update_rew_perf(rew_perf_evals, n_queries_made)

        if callback is not None:
          result = (rew_perf_evals, None)
          callback(result)

      if verbose:
        print('time elapsed: %f' % (time.time() - start_time), flush=True)
        print('')

      iter_idx += 1

    query_data = {
        'demo_rollouts': demo_rollouts,
        'sketch_rollouts': sketch_rollouts,
        'pref_logs': pref_logs
    }

    return rew_perf_evals, query_data

  def save(self):
    self.reward_model.save()
    self.dynamics_model.save()

  def load(self):
    self.reward_model.load()
    self.dynamics_model.load()
