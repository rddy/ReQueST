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
"""Smoke test on 2D bandit domain"""

from __future__ import division

import numpy as np
import os
import uuid

import logging
logging.getLogger().setLevel(logging.INFO)

from rqst import envs
from rqst import reward_models
from rqst import utils
from rqst.dynamics_models import DynamicsModel
from rqst.reward_models import BCRewardModel
from rqst.reward_models import RewardModel
from rqst.reward_opt import InteractiveRewardOptimizer
from rqst.traj_opt import GDTrajOptimizer
from rqst.traj_opt import StochTrajOptimizer

test_data_dir = os.path.join(utils.bandit_data_dir, 'test')
if not os.path.exists(test_data_dir):
  os.makedirs(test_data_dir)

n_demo_rollouts = 10
n_aug_rollouts = 10


def main():
  sess = utils.make_tf_session(gpu_mode=False)

  env = envs.make_bandit_env()
  trans_env = envs.make_bandit_trans_env(env)
  expert_policy = env.make_expert_policy()
  random_policy = utils.make_random_policy(env)

  default_init_obs = env.default_init_obs

  utils.run_ep(expert_policy, env)
  utils.run_ep(random_policy, env)
  utils.run_ep(expert_policy, trans_env)
  utils.run_ep(random_policy, trans_env)

  logging.info('envs and policies OK')

  demo_rollouts = [
      utils.run_ep(expert_policy, env) for _ in range(n_demo_rollouts)
  ]

  aug_rollouts = demo_rollouts + [
      utils.run_ep(random_policy, env) for _ in range(n_aug_rollouts)
  ]

  demo_data = utils.split_rollouts(
      utils.vectorize_rollouts(demo_rollouts, env.max_ep_len))
  aug_data = utils.split_rollouts(
      utils.vectorize_rollouts(aug_rollouts, env.max_ep_len))

  unused_demo_traj_data = utils.split_rollouts(
      utils.vectorize_rollouts(
          demo_rollouts, env.max_ep_len, preserve_trajs=True))
  unused_aug_traj_data = utils.split_rollouts(
      utils.vectorize_rollouts(
          aug_rollouts, env.max_ep_len, preserve_trajs=True))

  logging.info('data collection OK')

  dynamics_model = DynamicsModel(sess, env)

  dynamics_model.train(
      aug_data,
      iterations=1,
      ftol=1e-4,
      learning_rate=1e-3,
      batch_size=4,
      val_update_freq=1,
      verbose=False)

  dynamics_model.save()

  dynamics_model.load()

  logging.info('dynamics model OK')

  demo_data_for_reward_model = demo_data
  demo_rollouts_for_reward_model = demo_rollouts

  sketch_data_for_reward_model = aug_data
  sketch_rollouts_for_reward_model = aug_rollouts

  autolabels = reward_models.autolabel_prefs(
      aug_rollouts, env, segment_len=env.max_ep_len + 1)

  pref_logs_for_reward_model = autolabels
  pref_data_for_reward_model = utils.split_prefs(autolabels)

  logging.info('autolabels OK')

  reward_model = RewardModel(
      sess,
      env,
      n_rew_nets_in_ensemble=4,
      n_layers=1,
      layer_size=64,
      scope=str(uuid.uuid4()),
      scope_file=os.path.join(test_data_dir, 'rew_scope.pkl'),
      tf_file=os.path.join(test_data_dir, 'rew.tf'),
      rew_func_input='sa',
      use_discrete_actions=True)

  reward_model = BCRewardModel(
      sess,
      env,
      n_rew_nets_in_ensemble=4,
      n_layers=1,
      layer_size=64,
      scope=str(uuid.uuid4()),
      scope_file=os.path.join(test_data_dir, 'bc_rew_scope.pkl'),
      tf_file=os.path.join(test_data_dir, 'bc_rew.tf'),
      rew_func_input='sa',
      use_discrete_actions=True)

  for demo_data in [None, demo_data_for_reward_model]:
    for sketch_data in [None, sketch_data_for_reward_model]:
      for pref_data in [None, pref_data_for_reward_model]:
        if pref_data is None and sketch_data is None:
          continue
        reward_model.train(
            demo_data=demo_data,
            sketch_data=sketch_data,
            pref_data=pref_data,
            demo_coeff=1.,
            sketch_coeff=1.,
            iterations=1,
            ftol=1e-4,
            batch_size=4,
            learning_rate=1e-3,
            val_update_freq=1,
            verbose=False)

  reward_model.save()

  reward_model.load()

  logging.info('reward models OK')

  for query_loss_opt in [
      'pref_uncertainty', 'rew_uncertainty', 'max_rew', 'min_rew', 'max_nov',
      'max_imi_pol_uncertainty'
  ]:
    for init_obs in [None, default_init_obs]:
      for join_trajs_at_init_state in [True, False]:
        for query_type in ['pref', 'demo', 'sketch']:
          if query_type == 'pref' and query_loss_opt == 'max_nov':
            continue

          for shoot_steps in [1, 2]:
            if shoot_steps > 1 and np.array(init_obs == default_init_obs).all():
              continue
            traj_optimizer = GDTrajOptimizer(
                sess,
                env,
                reward_model,
                dynamics_model,
                traj_len=2,
                n_trajs=2,
                prior_coeff=1.,
                diversity_coeff=0.,
                query_loss_opt=query_loss_opt,
                opt_init_obs=(init_obs is None),
                join_trajs_at_init_state=join_trajs_at_init_state,
                shoot_steps=shoot_steps,
                learning_rate=1e-2,
                query_type=query_type)

            traj_optimizer.run(
                init_obs=init_obs,
                iterations=1,
                ftol=1e-4,
                verbose=False,
            )

  logging.info('grad descent traj opt OK')

  imitation_kwargs = {'plan_horizon': 10, 'n_blind_steps': 2, 'test_mode': True}

  for n_eval_rollouts in [0, 1]:
    reward_models.evaluate_reward_model(
        sess,
        env,
        trans_env,
        reward_model,
        dynamics_model,
        offpol_eval_rollouts=sketch_rollouts_for_reward_model,
        n_eval_rollouts=n_eval_rollouts,
        imitation_kwargs=imitation_kwargs)

  logging.info('reward eval OK')

  for query_loss_opt in [
      'pref_uncertainty', 'rew_uncertainty', 'max_rew', 'min_rew', 'max_nov',
      'max_imi_pol_uncertainty', 'unif'
  ]:
    for use_rand_policy in [False, True]:
      traj_optimizer = StochTrajOptimizer(
          sess,
          env,
          reward_model,
          dynamics_model,
          traj_len=2,
          rollout_len=2,
          query_loss_opt=query_loss_opt,
          use_rand_policy=use_rand_policy)

      traj_optimizer.run(n_trajs=2, n_samples=2, init_obs=None, verbose=False)

  logging.info('stoch traj opt OK')

  reward_model = RewardModel(
      sess,
      env,
      n_rew_nets_in_ensemble=4,
      n_layers=1,
      layer_size=64,
      scope=str(uuid.uuid4()),
      scope_file=os.path.join(test_data_dir, 'rew_scope.pkl'),
      tf_file=os.path.join(test_data_dir, 'rew.tf'),
      rew_func_input='sa',
      use_discrete_actions=True)

  dynamics_model = DynamicsModel(sess, env)

  rew_optimizer = InteractiveRewardOptimizer(sess, env, trans_env, reward_model,
                                             dynamics_model)

  bc_reward_model = BCRewardModel(
      sess,
      env,
      n_rew_nets_in_ensemble=4,
      n_layers=1,
      layer_size=64,
      scope=str(uuid.uuid4()),
      scope_file=os.path.join(test_data_dir, 'rew_scope.pkl'),
      tf_file=os.path.join(test_data_dir, 'rew.tf'),
      rew_func_input='sa',
      use_discrete_actions=True)

  bc_rew_optimizer = InteractiveRewardOptimizer(sess, env, trans_env,
                                                bc_reward_model, dynamics_model)

  reward_train_kwargs = {
      'demo_coeff': 1.,
      'sketch_coeff': 1.,
      'iterations': 1,
      'ftol': 1e-4,
      'batch_size': 4,
      'learning_rate': 1e-3,
      'val_update_freq': 1,
      'verbose': False
  }

  dynamics_train_kwargs = {}

  gd_traj_opt_init_kwargs = {
      'traj_len': 2,
      'n_trajs': 2,
      'prior_coeff': 1.,
      'diversity_coeff': 1.,
      'query_loss_opt': 'pref_uncertainty',
      'opt_init_obs': False,
      'learning_rate': 1e-2,
      'join_trajs_at_init_state': False
  }

  gd_traj_opt_run_kwargs = {
      'init_obs': default_init_obs,
      'iterations': 1,
      'ftol': 1e-4,
      'verbose': False,
  }

  unused_stoch_traj_opt_init_kwargs = {
      'traj_len': 2,
      'rollout_len': 2,
      'query_loss_opt': 'pref_uncertainty'
  }

  unused_stoch_traj_opt_run_kwargs = {
      'n_samples': 2,
      'init_obs': None,
      'verbose': False
  }

  imitation_kwargs = {'plan_horizon': 2, 'n_blind_steps': 2}

  eval_kwargs = {'n_eval_rollouts': 1}

  for init_train in [True, False]:
    for query_type in ['pref', 'sketch', 'demo']:
      local_rew_optimizer = bc_rew_optimizer if query_type == 'demo' else rew_optimizer
      local_rew_optimizer.run(
          demo_rollouts=demo_rollouts_for_reward_model,
          sketch_rollouts=sketch_rollouts_for_reward_model,
          pref_logs=pref_logs_for_reward_model,
          rollouts_for_dyn=aug_rollouts,
          reward_train_kwargs=reward_train_kwargs,
          dynamics_train_kwargs=dynamics_train_kwargs,
          traj_opt_cls=GDTrajOptimizer,
          traj_opt_run_kwargs=gd_traj_opt_run_kwargs,
          traj_opt_init_kwargs=gd_traj_opt_init_kwargs,
          imitation_kwargs=imitation_kwargs,
          eval_kwargs=eval_kwargs,
          init_train_dyn=init_train,
          init_train_rew=init_train,
          n_imitation_rollouts_per_dyn_update=1,
          n_queries=1,
          reward_update_freq=1,
          reward_eval_freq=1,
          dyn_update_freq=1,
          verbose=False,
          query_type=query_type)

  rew_optimizer.save()

  rew_optimizer.load()

  logging.info('rqst OK')


if __name__ == '__main__':
  main()
