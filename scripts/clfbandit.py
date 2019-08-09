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
"""Run experiments on MNIST classification domain"""

from __future__ import division

from copy import deepcopy
import pickle
import random
import uuid
import sys
import os

import tensorflow as tf
import numpy as np

from rqst.traj_opt import GDTrajOptimizer, StochTrajOptimizer
from rqst.reward_models import RewardModel, BCRewardModel
from rqst.reward_opt import InteractiveRewardOptimizer
from rqst.dynamics_models import ObsPriorModel
from rqst.encoder_models import VAEModel, IdenModel, MNISTVAEModel
from rqst import reward_models
from rqst import utils
from rqst import envs

from matplotlib import pyplot as plt
import matplotlib as mpl

import warnings
warnings.filterwarnings('ignore')

sess = utils.make_tf_session(gpu_mode=False)

env = envs.make_clfbandit_env()
trans_env = envs.make_clfbandit_trans_env(env)

encoder = MNISTVAEModel(
    sess,
    env,
    kl_tolerance=utils.inf,
    #scope=str(uuid.uuid4()),
    scope_file=os.path.join(utils.clfbandit_data_dir, 'enc_scope.pkl'),
    tf_file=os.path.join(utils.clfbandit_data_dir, 'enc.tf'))

encoder.load()

env.set_expert_policy(encoder)
trans_env.set_expert_policy(encoder)

with open(os.path.join(utils.clfbandit_data_dir, 'demo_rollouts.pkl'),
          'rb') as f:
  demo_rollouts = pickle.load(f)

with open(os.path.join(utils.clfbandit_data_dir, 'aug_rollouts.pkl'),
          'rb') as f:
  aug_rollouts = pickle.load(f)

dynamics_model = ObsPriorModel(sess, env, encoder=encoder)

sketch_rollouts_for_reward_model = None
pref_logs_for_reward_model = None

offpol_eval_rollouts = random.sample(aug_rollouts, 1000)

demo_rollouts_for_reward_model = demo_rollouts

reward_model = BCRewardModel(
    sess,
    env,
    n_rew_nets_in_ensemble=4,
    n_layers=1,
    layer_size=128,
    scope=str(uuid.uuid4()),
    scope_file=os.path.join(utils.clfbandit_data_dir, 'bc_rew_scope.pkl'),
    tf_file=os.path.join(utils.clfbandit_data_dir, 'bc_rew.tf'),
    rew_func_input='sa',
    use_discrete_actions=True)

rew_optimizer = InteractiveRewardOptimizer(sess, env, trans_env, reward_model,
                                           dynamics_model)

reward_train_kwargs = {
    'demo_coeff': 1.,
    'sketch_coeff': 1.,
    'iterations': 5000,
    'ftol': 1e-4,
    'batch_size': 32,
    'learning_rate': 1e-2,
    'val_update_freq': 100,
    'verbose': False
}

dynamics_train_kwargs = {}

imitation_kwargs = {}

eval_kwargs = {
    'n_eval_rollouts': 100,
    'offpol_eval_rollouts': offpol_eval_rollouts
}

sing_gd_traj_opt_init_kwargs = {
    'traj_len': 2,
    'n_trajs': 1,
    'prior_coeff': 0.,
    'diversity_coeff': 0.,
    'query_loss_opt': 'unif',
    'opt_init_obs': True,
    'opt_act_seq': True,
    'learning_rate': 1e-2,
    'join_trajs_at_init_state': True,
    'shoot_steps': None
}

sing_gd_traj_opt_run_kwargs = {
    'init_obs': None,
    'iterations': 5000,
    'ftol': 1e-4,
    'verbose': False,
    'warm_start': False
}

query_loss_opts = ['max_imi_pol_uncertainty', 'max_nov']
prior_coeffs = [1e-1, 1e-2]

traj_opt_init_kwargs = []
for query_loss_opt, prior_coeff in zip(query_loss_opts, prior_coeffs):
  kwargs = deepcopy(sing_gd_traj_opt_init_kwargs)
  kwargs['query_loss_opt'] = query_loss_opt
  kwargs['prior_coeff'] = prior_coeff
  traj_opt_init_kwargs.append(kwargs)
gd_traj_opt_init_kwargs = traj_opt_init_kwargs
gd_traj_opt_run_kwargs = [sing_gd_traj_opt_run_kwargs
                         ] * len(gd_traj_opt_init_kwargs)

stoch_traj_opt_init_kwargs = {
    'traj_len': 2,
    'rollout_len': 1,
    'query_loss_opt': 'unif',
    'use_rand_policy': False
}

stoch_traj_opt_run_kwargs = {
    'n_trajs': 10,
    'n_samples': 10,
    'init_obs': None,
    'verbose': False
}

rew_opt_kwargs = {
    'demo_rollouts': demo_rollouts_for_reward_model,
    'sketch_rollouts': sketch_rollouts_for_reward_model,
    'pref_logs': pref_logs_for_reward_model,
    'rollouts_for_dyn': [],  #aug_rollouts,
    'reward_train_kwargs': reward_train_kwargs,
    'dynamics_train_kwargs': dynamics_train_kwargs,
    'imitation_kwargs': imitation_kwargs,
    'eval_kwargs': eval_kwargs,
    'init_train_dyn': False,
    'init_train_rew': True,
    'n_imitation_rollouts_per_dyn_update': 1,
    'n_queries': 2000,
    'reward_update_freq': 5,
    'reward_eval_freq': 5,
    'dyn_update_freq': None,
    'verbose': False,
    'warm_start_rew': False,
    'query_type': 'demo'
}


def make_eval_func(conf_key):

  def eval_func(conf_val, callback=None):
    if conf_key == 'prior_coeff':
      traj_opt_init_kwargs = deepcopy(gd_traj_opt_init_kwargs)
      for i in range(len(traj_opt_init_kwargs)):
        traj_opt_init_kwargs[i][conf_key] = conf_val[i]
      traj_opt_run_kwargs = gd_traj_opt_run_kwargs
    elif conf_key == 'query_loss_opt':
      idxes = [
          i for i, kwargs in enumerate(gd_traj_opt_init_kwargs)
          if kwargs['query_loss_opt'] != conf_val
      ]
      traj_opt_init_kwargs = [gd_traj_opt_init_kwargs[i] for i in idxes]
      traj_opt_run_kwargs = [gd_traj_opt_run_kwargs[i] for i in idxes]
    return rew_optimizer.run(
        traj_opt_cls=GDTrajOptimizer,
        traj_opt_run_kwargs=traj_opt_run_kwargs,
        traj_opt_init_kwargs=traj_opt_init_kwargs,
        callback=callback,
        **rew_opt_kwargs)

  return eval_func


eval_prior_coeff = make_eval_func('prior_coeff')
eval_query_loss = make_eval_func('query_loss_opt')

confs = []

n_trials = 3

prior_coeffs = list(zip([0, 1e-2, 1e-1, 1], [0, 1e-3, 1e-2, 1e-1]))

for prior_coeff in prior_coeffs:
  for i in range(n_trials):
    func = lambda callback, conf: eval_prior_coeff(conf[0], callback=callback)
    conf = (prior_coeff, i)
    confs.append((conf, func))

query_loss_opts = ['max_imi_pol_uncertainty', 'max_nov']

for query_loss_opt in query_loss_opts:
  for i in range(n_trials):
    func = lambda callback, conf: eval_query_loss(conf[0], callback=callback)
    conf = (query_loss_opt, i)
    confs.append((conf, func))

compute_stoch_eval = lambda callback, conf: rew_optimizer.run(
    traj_opt_cls=StochTrajOptimizer,
    traj_opt_run_kwargs=stoch_traj_opt_run_kwargs,
    traj_opt_init_kwargs=stoch_traj_opt_init_kwargs,
    callback=callback,
    **rew_opt_kwargs)

for i in range(n_trials):
  conf = ('stoch', i)
  func = compute_stoch_eval
  confs.append((conf, func))

compute_unif_eval = lambda callback, conf: rew_optimizer.run(
    traj_opt_cls=GDTrajOptimizer,
    traj_opt_run_kwargs=sing_gd_traj_opt_run_kwargs,
    traj_opt_init_kwargs=sing_gd_traj_opt_init_kwargs,
    callback=callback,
    **rew_opt_kwargs)

for i in range(n_trials):
  conf = ('unif', i)
  func = compute_unif_eval
  confs.append((conf, func))

prior_coeff = (1e-1, 1e-2)
for i in range(n_trials):
  func = lambda callback, conf: eval_prior_coeff(conf[0], callback=callback)
  conf = (prior_coeff, i)
  confs.append((conf, func))

if __name__ == '__main__':
  #print(len(confs))
  #raise ValueError

  conf_idx = int(sys.argv[1])
  conf, func = confs[conf_idx]
  print('conf: %s' % str(conf), flush=True)

  def callback(result):
    with open(
        os.path.join(utils.clfbandit_data_dir, 'cloud', '%d.pkl' % conf_idx),
        'wb') as f:
      pickle.dump((conf, result), f, pickle.HIGHEST_PROTOCOL)

  result = func(callback, conf)
  callback(result)
