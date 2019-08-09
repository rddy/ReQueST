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
"""Smoke test on MNIST classification domain"""

from __future__ import division

import os

import logging
logging.getLogger().setLevel(logging.INFO)

from rqst import envs
from rqst import utils

test_data_dir = os.path.join(utils.clfbandit_data_dir, 'test')
if not os.path.exists(test_data_dir):
  os.makedirs(test_data_dir)

n_demo_rollouts = 10
n_aug_rollouts = 10


def main():
  unused_sess = utils.make_tf_session(gpu_mode=False)

  env = envs.make_clfbandit_env(verbose=True)
  env.expert_policy = env.make_expert_policy()
  env.random_policy = utils.make_random_policy(env)
  trans_env = envs.make_clfbandit_trans_env(env)

  utils.run_ep(env.expert_policy, env)
  utils.run_ep(env.random_policy, env)
  utils.run_ep(trans_env.expert_policy, trans_env)
  utils.run_ep(trans_env.random_policy, trans_env)

  logging.info('OK')


if __name__ == '__main__':
  main()
