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
"""Smoke test on Car Racing domain"""

from __future__ import division

import os
import uuid

import logging
logging.getLogger().setLevel(logging.INFO)

import numpy as np
from rqst import envs
from rqst import reward_models
from rqst import utils
from rqst.dynamics_models import AbsorptionModel
from rqst.dynamics_models import load_wm_pretrained_rnn
from rqst.dynamics_models import MDNRNNDynamicsModel
from rqst.encoder_models import load_wm_pretrained_vae
from rqst.encoder_models import VAEModel
from rqst.reward_opt import InteractiveRewardOptimizer
from rqst.traj_opt import GDTrajOptimizer
from rqst.traj_opt import StochTrajOptimizer

test_data_dir = os.path.join(utils.carracing_data_dir, 'test')
if not os.path.exists(test_data_dir):
  os.makedirs(test_data_dir)

n_demo_rollouts = 3
n_aug_rollouts = 3


def main():
  sess = utils.make_tf_session(gpu_mode=False)

  env = envs.make_carracing_env(sess)
  trans_env = envs.make_carracing_trans_env(sess)
  random_policy = utils.make_random_policy(env)

  utils.run_ep(random_policy, env, max_ep_len=3, render=False)
  trans_rollout = utils.run_ep(
      random_policy, trans_env, max_ep_len=3, render=False)

  logging.info('envs and policies OK')

  raw_demo_rollouts = [
      utils.run_ep(random_policy, env, max_ep_len=3, render=False)
      for _ in range(n_demo_rollouts)
  ]
  raw_aug_rollouts = [
      utils.run_ep(random_policy, env, max_ep_len=3, render=False)
      for _ in range(n_aug_rollouts)
  ]
  raw_aug_rollouts += raw_demo_rollouts

  raw_aug_obses = []
  for rollout in raw_aug_rollouts:
    for x in rollout:
      raw_aug_obses.append(x[0])
  raw_aug_obses = np.array(raw_aug_obses)
  raw_aug_obs_data = utils.split_rollouts({'obses': raw_aug_obses})

  logging.info('data collection OK')

  encoder = VAEModel(
      sess,
      env,
      learning_rate=0.0001,
      kl_tolerance=0.5,
      scope=str(uuid.uuid4()),
      scope_file=os.path.join(test_data_dir, 'enc_scope.pkl'),
      tf_file=os.path.join(test_data_dir, 'enc.tf'))

  encoder.train(
      raw_aug_obs_data,
      iterations=1,
      ftol=1e-4,
      learning_rate=1e-3,
      val_update_freq=1,
      verbose=False)

  encoder = load_wm_pretrained_vae(sess, env)

  encoder.save()

  encoder.load()

  obs = raw_aug_rollouts[0][0][0]
  latent = encoder.encode_frame(obs)
  unused_recon = encoder.decode_latent(latent)

  logging.info('encoder OK')

  raw_aug_traj_data = utils.split_rollouts(
      utils.vectorize_rollouts(
          raw_aug_rollouts, env.max_ep_len, preserve_trajs=True))

  abs_model = AbsorptionModel(
      sess,
      env,
      n_layers=1,
      layer_size=32,
      scope=str(uuid.uuid4()),
      scope_file=os.path.join(test_data_dir, 'abs_scope.pkl'),
      tf_file=os.path.join(test_data_dir, 'abs.tf'))

  dynamics_model = MDNRNNDynamicsModel(
      encoder,
      sess,
      env,
      scope=str(uuid.uuid4()),
      tf_file=os.path.join(test_data_dir, 'dyn.tf'),
      scope_file=os.path.join(test_data_dir, 'dyn_scope.pkl'),
      abs_model=abs_model)

  dynamics_model.train(
      raw_aug_traj_data,
      iterations=1,
      learning_rate=1e-3,
      ftol=1e-4,
      batch_size=2,
      val_update_freq=1,
      verbose=False)

  dynamics_model = load_wm_pretrained_rnn(encoder, sess, env)

  dynamics_model.save()

  dynamics_model.load()

  demo_traj_data = utils.rnn_encode_rollouts(raw_demo_rollouts, env, encoder,
                                             dynamics_model)
  aug_traj_data = utils.rnn_encode_rollouts(raw_aug_rollouts, env, encoder,
                                            dynamics_model)
  demo_rollouts = utils.rollouts_of_traj_data(demo_traj_data)
  aug_rollouts = utils.rollouts_of_traj_data(aug_traj_data)
  demo_data = utils.split_rollouts(utils.flatten_traj_data(demo_traj_data))
  aug_data = utils.split_rollouts(utils.flatten_traj_data(aug_traj_data))

  env.default_init_obs = aug_rollouts[0][0][0]

  trans_rollouts = utils.rollouts_of_traj_data(
      utils.rnn_encode_rollouts([trans_rollout], trans_env, encoder,
                                dynamics_model))
  trans_env.default_init_obs = trans_rollouts[0][0][0]

  logging.info('mdnrnn dynamics OK')

  demo_data_for_reward_model = demo_data
  demo_rollouts_for_reward_model = demo_rollouts

  sketch_data_for_reward_model = aug_data
  sketch_rollouts_for_reward_model = aug_rollouts

  reward_init_kwargs = {
      'n_rew_nets_in_ensemble': 2,
      'n_layers': 1,
      'layer_size': 32,
      'scope': str(uuid.uuid4()),
      'scope_file': os.path.join(test_data_dir, 'true_rew_scope.pkl'),
      'tf_file': os.path.join(test_data_dir, 'true_rew.tf'),
      'rew_func_input': "s'",
      'use_discrete_rewards': True
  }

  reward_train_kwargs = {
      'demo_coeff': 1.,
      'sketch_coeff': 1.,
      'iterations': 1,
      'ftol': 1e-4,
      'batch_size': 2,
      'learning_rate': 1e-3,
      'val_update_freq': 1,
      'verbose': False
  }

  data = envs.make_carracing_rew(
      sess,
      env,
      sketch_data=sketch_data_for_reward_model,
      reward_init_kwargs=reward_init_kwargs,
      reward_train_kwargs=reward_train_kwargs)
  env.__dict__.update(data)
  trans_env.__dict__.update(data)

  autolabels = reward_models.autolabel_prefs(
      aug_rollouts, env, segment_len=env.max_ep_len + 1)

  pref_logs_for_reward_model = autolabels
  pref_data_for_reward_model = utils.split_prefs(autolabels)

  logging.info('autolabels OK')

  for rew_func_input in ['s', 'sa', "s'"]:
    reward_model = reward_models.RewardModel(
        sess,
        env,
        n_rew_nets_in_ensemble=2,
        n_layers=1,
        layer_size=32,
        scope=str(uuid.uuid4()),
        scope_file=os.path.join(test_data_dir, 'rew_scope.pkl'),
        tf_file=os.path.join(test_data_dir, 'rew.tf'),
        rew_func_input=rew_func_input,
        use_discrete_rewards=True)

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
            batch_size=2,
            learning_rate=1e-3,
            val_update_freq=1,
            verbose=False)

  reward_model.save()

  reward_model.load()

  logging.info('reward models OK')

  for query_loss_opt in [
      'pref_uncertainty', 'rew_uncertainty', 'max_rew', 'min_rew', 'max_nov'
  ]:
    for init_obs in [None, env.default_init_obs]:
      for join_trajs_at_init_state in [True, False]:
        for shoot_steps in [1, 2]:
          if (shoot_steps > 1 and
              np.array(init_obs == env.default_init_obs).all()):
            continue
          traj_opt = GDTrajOptimizer(
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
              learning_rate=1e-2)

          traj_opt.run(
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
      'unif'
  ]:
    for use_rand_policy in [False, True]:
      traj_opt = StochTrajOptimizer(
          sess,
          env,
          reward_model,
          dynamics_model,
          traj_len=2,
          rollout_len=2,
          query_loss_opt=query_loss_opt,
          use_rand_policy=use_rand_policy)

      for init_obs in [None, env.default_init_obs]:
        traj_opt.run(n_trajs=2, n_samples=2, init_obs=init_obs, verbose=False)

  logging.info('stoch traj opt OK')

  reward_model = reward_models.RewardModel(
      sess,
      env,
      n_rew_nets_in_ensemble=2,
      n_layers=1,
      layer_size=32,
      scope=str(uuid.uuid4()),
      scope_file=os.path.join(test_data_dir, 'rew_scope.pkl'),
      tf_file=os.path.join(test_data_dir, 'rew.tf'),
      rew_func_input="s'",
      use_discrete_rewards=True)

  rew_optimizer = InteractiveRewardOptimizer(sess, env, trans_env, reward_model,
                                             dynamics_model)

  reward_train_kwargs = {
      'demo_coeff': 1.,
      'sketch_coeff': 1.,
      'iterations': 1,
      'ftol': 1e-4,
      'batch_size': 2,
      'learning_rate': 1e-3,
      'val_update_freq': 1,
      'verbose': False
  }

  dynamics_train_kwargs = {
      'iterations': 1,
      'batch_size': 2,
      'learning_rate': 1e-3,
      'ftol': 1e-4,
      'val_update_freq': 1,
      'verbose': False
  }

  gd_traj_opt_init_kwargs = {
      'traj_len': env.max_ep_len,
      'n_trajs': 2,
      'prior_coeff': 1.,
      'diversity_coeff': 1.,
      'query_loss_opt': 'pref_uncertainty',
      'opt_init_obs': False,
      'learning_rate': 1e-2,
      'join_trajs_at_init_state': False
  }

  gd_traj_opt_run_kwargs = {
      'init_obs': env.default_init_obs,
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

  eval_kwargs = {'n_eval_rollouts': 1}

  for init_train in [True, False]:
    for query_type in ['pref', 'sketch']:
      rew_optimizer.run(
          demo_rollouts=demo_rollouts_for_reward_model,
          sketch_rollouts=sketch_rollouts_for_reward_model,
          pref_logs=pref_logs_for_reward_model,
          rollouts_for_dyn=raw_aug_rollouts,
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
