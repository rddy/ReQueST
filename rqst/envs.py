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
"""Four domains for evaluation:
 - 2D contextual bandit
 - MNIST classification
 - 2D navigation
 - Car Racing
"""

from __future__ import division

from copy import deepcopy
import pickle
import types
import uuid
import os
import sys

from sklearn import neighbors
from gym.envs.box2d.car_dynamics import Car
from gym import spaces
import numpy as np
import gym
import tensorflow as tf
import scipy

from rqst.reward_models import RewardModel
from rqst import utils

sys.path.append(utils.wm_dir)
import model as carracing_model


class Env(gym.Env):

  def make_noisy_expert_policy(self, eps=0.5):
    return utils.make_noisy_expert_policy(
        self.make_expert_policy(), self.action_space, eps=eps)

  def prob_succ(self, obses):
    return np.zeros(len(obses))

  def prob_crash(self, obses):
    return np.zeros(len(obses))


class BanditEnv(Env):
  """Contextual bandit with 2D features and binary actions"""

  def __init__(self, domain_shift=False):

    self.domain_shift = domain_shift
    self.n_obs_dim = 2
    self.n_act_dim = 2
    self.observation_space = spaces.Box(-np.ones(self.n_obs_dim),
                                        np.ones(self.n_obs_dim))
    self.action_space = spaces.Box(-utils.inf * np.ones(self.n_act_dim),
                                   utils.inf * np.ones(self.n_act_dim))
    self.name = 'bandit'
    self.max_ep_len = 1
    self.expert_policy = self.make_expert_policy()
    self.subopt_policy = self.make_noisy_expert_policy(eps=0.5)
    self.default_init_obs = np.zeros(self.n_obs_dim)
    self.rew_classes = None
    self.only_terminal_reward = True

    self.absorbing_state = np.zeros(self.n_obs_dim)
    self.raw_absorbing_state = np.zeros(self.n_obs_dim)

    self.w = np.array([0., 0.5])

  def sample_context(self):
    context = np.random.random(self.n_obs_dim)
    if self.domain_shift:
      context = context * 2 - 1
    return context

  def reward_func(self, obses, acts, next_obses, eps=1e-6):
    act_labels = np.array(
        [self.expert_policy(obses[i, :]) for i in range(obses.shape[0])])
    act_labels -= scipy.misc.logsumexp(act_labels, axis=1, keepdims=True)
    acts -= scipy.misc.logsumexp(acts, axis=1, keepdims=True)
    xent = -(acts * np.exp(act_labels)).sum(axis=1)
    return -xent

  def step(self, action):
    action_label = self.expert_policy(self.prev_obs)
    succ = np.argmax(action) == np.argmax(action_label)

    obs = self.raw_absorbing_state
    r = self.reward_func(self.prev_obs[np.newaxis, :], action[np.newaxis, :],
                         obs[np.newaxis, :])[0]
    done = True
    info = {'succ': succ, 'crash': False}

    return obs, r, done, info

  def reset(self):
    self.prev_obs = self.sample_context()
    return self.prev_obs

  def make_expert_policy(self):

    def policy(obs):
      a = 1 if self.w.dot(obs) > 0 else 0
      return utils.onehot_encode(a, self.n_act_dim) * utils.inf

    return policy


class CLFBanditEnv(BanditEnv):
  """Turn supervised classification problem into contextual bandit"""

  def __init__(self, dataset, domain_shift=False):

    self.dataset = dataset
    self.domain_shift = domain_shift

    self.n_z_dim = self.dataset['n_z_dim']
    self.n_obs_dim = self.n_z_dim
    self.n_act_dim = self.dataset['n_classes']
    self.observation_space = spaces.Box(self.dataset['feat_lows'],
                                        self.dataset['feat_highs'])
    self.action_space = spaces.Box(-utils.inf * np.ones(self.n_act_dim),
                                   utils.inf * np.ones(self.n_act_dim))
    self.name = 'clfbandit'
    self.max_ep_len = 1
    self.make_expert_policy = lambda: self.dataset['make_expert_policy'](
        self.dataset)

    self.expert_policy = None # initializing here is expensive, so just initialize as needed
    self.subopt_policy = None

    self.default_init_obs = None
    self.rew_classes = None
    self.only_terminal_reward = True

    self.raw_absorbing_state = np.zeros(self.dataset['feat_shape'])
    self.absorbing_state = np.zeros(self.n_obs_dim)

  def sample_context(self):
    idxes = self.dataset['val_idxes'] if self.domain_shift else self.dataset[
        'train_idxes']
    idx = np.random.choice(idxes)
    return self.dataset['feats'][idx, :]

  def set_expert_policy(self, encoder):
    self.expert_policy = self.dataset['make_expert_policy'](self.dataset,
                                                            encoder)
    self.subopt_policy = None # initializing here is expensive, so just initialize as needed


class PointMassEnv(Env):
  """Navigate to target while avoiding trap
  Not allowed to go out of bounds (unit square)
  Episode terminates upon reaching target or trap
  Continuous actions (velocity vector)
  2D position observations
  """

  def __init__(self,
               max_ep_len=1000,
               goal_dist_thresh=0.15,
               trap_dist_thresh=0.15,
               succ_rew_bonus=1.,
               crash_rew_penalty=-10.,
               max_speed=0.01,
               goal=None,
               trap=None,
               init_pos=None):
    """
    Args:
     init_pos: a np.array with dimensions (2)
      None -> random initialization for each episode
    """

    if goal is None:
      goal = np.array([0.5, 0.5])

    if trap is None:
      trap = np.array([0.25, 0.25])

    self.max_ep_len = max_ep_len
    self.goal_dist_thresh = goal_dist_thresh
    self.trap_dist_thresh = trap_dist_thresh
    self.succ_rew_bonus = succ_rew_bonus
    self.crash_rew_penalty = crash_rew_penalty
    self.max_speed = max_speed
    self.init_pos = init_pos
    self.goal = goal
    self.trap = trap

    # non-overlapping target/trap
    if np.linalg.norm(self.goal - self.trap) < 2 * self.goal_dist_thresh:
      raise ValueError

    self.n_act_dim = 2  # angle, speed
    self.n_obs_dim = 2  # position
    self.observation_space = spaces.Box(
        np.zeros(self.n_obs_dim), np.ones(self.n_obs_dim))
    self.action_space = spaces.Box(
        np.zeros(2), np.array([2 * np.pi, self.max_speed]))
    self.name = 'pointmass'
    self.expert_policy = self.make_expert_policy()
    self.subopt_policy = self.make_noisy_expert_policy(eps=0.75)
    self.default_init_obs = init_pos if init_pos is not None else np.zeros(2)
    self.rew_classes = np.array(
        [self.crash_rew_penalty, 0., self.succ_rew_bonus])
    self.only_terminal_reward = True

    self.pos = None

  def prob_succ(self, obses):
    at_goal = np.linalg.norm(obses - self.goal, axis=1) <= self.goal_dist_thresh
    return at_goal.astype(float)

  def prob_crash(self, obses):
    at_trap = np.linalg.norm(obses - self.trap, axis=1) <= self.trap_dist_thresh
    return at_trap.astype(float)

  def reward_func(self, obses, acts, next_obses):
    r = self.succ_rew_bonus * self.prob_succ(next_obses)
    r += self.crash_rew_penalty * self.prob_crash(next_obses)
    return r

  def obs(self):
    return self.pos

  def cart_to_polar(self, v):
    return np.array([np.arctan2(v[1], v[0]), np.linalg.norm(v)])

  def normalize_ang(self, a):
    return (2 * np.pi - abs(a) % (2 * np.pi)) if a < 0 else (abs(a) %
                                                             (2 * np.pi))

  def normalize_polar(self, v):
    return np.array([self.normalize_ang(v[0]), min(v[1], self.max_speed)])

  def polar_to_cart(self, v):
    return v[1] * np.array([np.cos(v[0]), np.sin(v[0])])

  def step(self, action):
    action = self.polar_to_cart(self.normalize_polar(action))
    if (self.pos + action >= 0).all() and (self.pos + action <
                                           1).all():  # stay in bounds
      self.pos += action

    self.succ = np.linalg.norm(self.pos - self.goal) <= self.goal_dist_thresh
    self.crash = np.linalg.norm(self.pos - self.trap) <= self.trap_dist_thresh

    self.timestep += 1

    obs = self.obs()
    r = self.reward_func(self.prev_obs[np.newaxis, :], action[np.newaxis, :],
                         obs[np.newaxis, :])[0]
    done = self.succ or self.crash
    info = {'goal': self.goal, 'succ': self.succ, 'crash': self.crash}

    self.prev_obs = obs

    return obs, r, done, info

  def reset(self):
    self.pos = np.random.random(2) if self.init_pos is None else deepcopy(
        self.init_pos)
    self.prev_obs = self.obs()
    self.timestep = 0
    return self.prev_obs

  def make_expert_policy(self, noise=0, safety_margin=0.05):
    """Expert goes directly to target, swings around trap if necessary"""

    def policy(obs):
      u = self.goal - obs
      w = self.cart_to_polar(u)
      v = self.trap - obs
      p = v.dot(u)
      x = obs + u / np.linalg.norm(u) * p
      if p > 0 and np.linalg.norm(
          v) < self.trap_dist_thresh + safety_margin and np.linalg.norm(
              x - self.trap) < self.trap_dist_thresh + safety_margin:
        w[0] = self.cart_to_polar(v)[0] + 0.5 * np.pi
      w[0] += np.random.normal(0, 1) * noise
      return w

    return policy


def make_bandit_env():
  return BanditEnv(domain_shift=False)


def make_bandit_trans_env(env):
  return BanditEnv(domain_shift=True)


def make_mnist_expert_policy(dataset, encoder=None):
  obses = dataset['feats']
  if encoder is not None:
    obses = encoder.encode_batch_frames(obses)
  else:
    obses = obses.reshape((obses.shape[0], obses.shape[1] * obses.shape[2]))

  clf_path = os.path.join(utils.clfbandit_data_dir,
                          'clf-%d.pkl' % obses.shape[1])
  if os.path.exists(clf_path):
    with open(clf_path, 'rb') as f:
      clf = pickle.load(f)
  else:
    clf = neighbors.KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=1)
    clf.fit(obses, dataset['labels'])
    with open(clf_path, 'wb') as f:
      pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

  log_proba_of_obs = {
      tuple(obses[i].ravel()):
      (utils.onehot_encode(label, dataset['n_classes']) * utils.inf)
      for i, label in enumerate(dataset['labels'])
  }

  def policy(obs):
    if len(obs.shape) == 3 and encoder is not None:
      obs = encoder.encode_frame(obs)
    flat_obs = obs.ravel()
    try:
      return log_proba_of_obs[tuple(flat_obs)]
    except KeyError:
      return np.log(1e-9 + clf.predict_proba(flat_obs[np.newaxis, :])[0, :])

  return policy


def make_mnist_dataset(verbose=False):
  load = lambda fname: np.load(os.path.join(utils.mnist_dir, fname))

  def load_imgs(X):
    X = X.T
    d = int(np.sqrt(X.shape[1]))
    X = X.reshape((X.shape[0], d, d))
    return X

  load_labels = lambda X: X.T.ravel().astype(int)
  X = load('mnist.npz')
  train_imgs = load_imgs(X['train'])
  train_labels = load_labels(X['train_labels'])
  test_imgs = load_imgs(X['test'])
  test_labels = load_labels(X['test_labels'])

  imgs = np.concatenate((train_imgs, test_imgs), axis=0)
  labels = np.concatenate((train_labels, test_labels))

  n_classes = len(np.unique(labels))

  feats = imgs[:, :, :, np.newaxis] / 255.

  feat_shape = feats.shape[1:]
  feat_lows = np.zeros(feat_shape)
  feat_highs = np.ones(feat_shape)

  # uncomment to do a train-val split with no domain shift
  #train_idxes = list(range(train_labels.size))
  #val_idxes = list(range(train_labels.size, train_labels.size + test_labels.size))

  # domain shift
  train_label_distrn = np.zeros(n_classes)
  n_src_classes = n_classes // 2
  train_label_distrn[:n_src_classes] = 0.
  train_label_distrn[n_src_classes:] = 1.

  train_idxes = []
  val_idxes = []

  idxes = np.arange(0, train_labels.size + test_labels.size, 1)
  idxes_of_label = [[] for _ in range(n_classes)]
  # assume np.unique(labels) = [0, 1, 2, ..., n_classes-1]
  # assume labels is balanced
  for idx, label in enumerate(labels):
    idxes_of_label[label].append(idx)
  for label, frac in enumerate(train_label_distrn):
    label_idxes = idxes_of_label[label]
    n_train_ex_with_label = int(frac * len(label_idxes))
    train_idxes.extend(label_idxes[:n_train_ex_with_label])
    val_idxes.extend(label_idxes[n_train_ex_with_label:])

  if verbose:
    train_distrn = np.zeros(n_classes)
    val_distrn = np.zeros(n_classes)
    for idx in train_idxes:
      train_distrn[labels[idx]] += 1
    for idx in val_idxes:
      val_distrn[labels[idx]] += 1
    train_distrn /= train_distrn.sum()
    val_distrn /= val_distrn.sum()
    print('Training label distribution: ' + str(train_distrn))
    print('Validation label distribution: ' + str(val_distrn))
    print('Number of training examples: %d' % len(train_idxes))
    print('Number of validation examples: %d' % len(val_idxes))

  dataset = {
      'feat_shape': feat_shape,
      'n_z_dim': 8,
      'n_classes': n_classes,
      'feat_lows': feat_lows,
      'feat_highs': feat_highs,
      'feats': feats,
      'labels': labels,
      'train_idxes': train_idxes,
      'val_idxes': val_idxes,
      'make_expert_policy': make_mnist_expert_policy
  }

  return dataset


def make_clfbandit_env(**make_dataset_kwargs):
  dataset = make_mnist_dataset(**make_dataset_kwargs)
  return CLFBanditEnv(dataset, domain_shift=False)


def make_clfbandit_trans_env(env):
  trans_env = deepcopy(env)
  trans_env.domain_shift = True
  return trans_env


def make_pointmass_env(goal=None, trap=None, init_pos=None):

  if goal is None:
    goal = np.array([0.3, 0.3])

  if trap is None:
    trap = np.array([0.7, 0.7])

  env = PointMassEnv(goal=goal, trap=trap, init_pos=init_pos)
  return env


def make_pointmass_trans_env(env, trans_init_pos=None):
  """make transfer env for reward_models.evaluate_reward_model
  """
  if trans_init_pos is None:
    trans_init_pos = np.array([0.99, 0.99])

  trans_env = PointMassEnv(
      goal=env.goal, trap=env.trap, init_pos=trans_init_pos)
  return trans_env

def make_carracing_rew(sess,
                       env,
                       sketch_data=None,
                       reward_init_kwargs=None,
                       reward_train_kwargs=None):
  """carracing doesn't have a reward function that can be evaluated on encoded frames,
   so we train a ground-truth reward model by doing supervised learning on ground-truth rewards
  Args:
   sketch_data:
    None -> try to load trained reward model
    else -> train reward model on sketch_data
  """
  if reward_init_kwargs is None:
    reward_init_kwargs = {
        'n_rew_nets_in_ensemble':
            4,
        'n_layers':
            1,
        'layer_size':
            256,
        'scope':
            str(uuid.uuid4()) if sketch_data is not None else None,
        'scope_file':
            os.path.join(utils.carracing_data_dir, 'true_rew_scope.pkl'),
        'tf_file':
            os.path.join(utils.carracing_data_dir, 'true_rew.tf'),
        'rew_func_input':
            "s'",
        'use_discrete_rewards':
            True
    }

  if reward_train_kwargs is None:
    reward_train_kwargs = {
        'demo_coeff': 1,
        'sketch_coeff': 1,
        'iterations': 5000,
        'ftol': 1e-4,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'val_update_freq': 100,
        'verbose': True
    }

  true_reward_model = RewardModel(sess, env, **reward_init_kwargs)

  if sketch_data is not None:
    true_reward_model.train(
        demo_data=None,
        sketch_data=sketch_data,
        pref_data=None,
        **reward_train_kwargs)
    true_reward_model.save()
  else:
    true_reward_model.load()

  # reward_func(traj, act_seq) = array of rewards (one rew per timestep)
  prob_succ = lambda obses: utils.normalize_logits(
      true_reward_model.compute_raw_of_transes(None, None, obses))[:, 2]
  prob_crash = lambda obses: utils.normalize_logits(
      true_reward_model.compute_raw_of_transes(None, None, obses))[:, 0]

  def reward_func(*args):
    raws = true_reward_model.compute_raw_of_transes(*args)
    disc_rews = np.argmax(raws, axis=1)
    return [env.rew_classes[r] for r in disc_rews]

  return {
      'prob_succ': prob_succ,
      'prob_crash': prob_crash,
      'reward_func': reward_func
  }

def make_carracing_env(sess,
                       load_reward=False,
                       n_z_dim=32,
                       rnn_size=256,
                       succ_rew_bonus=10.,
                       crash_rew_penalty=-1.):
  """
  Args:
   n_z_dim: number of latent features for encoded frames
   rnn_size: size of hidden layer in mdn-rnn dynamics model
  """

  env = gym.make('CarRacing-v0')
  env.n_act_dim = 3  # steer, gas, brake
  env.max_ep_len = 1000
  env.name = 'carracing'
  env.default_init_obs = None
  env.succ_rew_bonus = succ_rew_bonus
  env.crash_rew_penalty = crash_rew_penalty
  env.rew_classes = np.array([env.crash_rew_penalty, 0., env.succ_rew_bonus])
  env.only_terminal_reward = False

  env.n_z_dim = n_z_dim
  env.rnn_size = rnn_size
  # concatenate latents with hidden states from mdnrnn dynamics model
  env.n_obs_dim = n_z_dim + 2 * rnn_size
  if load_reward:
    data = make_carracing_rew(sess, env)
    env.__dict__.update(data)

  filename = os.path.join(utils.wm_dir, 'log', 'carracing.cma.16.64.best.json')
  expert_model = carracing_model.make_model()
  expert_model.load_model(filename)

  def encode_obs(obs):
    if len(obs.shape) == 3:
      return expert_model.encode_obs(obs)[0]
    else:
      expert_model.state = tf.nn.rnn_cell.LSTMStateTuple(
          c=obs[env.n_z_dim:env.n_z_dim + env.rnn_size][np.newaxis, :],
          h=obs[-env.rnn_size:][np.newaxis, :])
      return obs[:env.n_z_dim]

  env.expert_policy = lambda obs: expert_model.get_action(encode_obs(obs))
  env.subopt_policy = utils.LaggyPolicy(env.expert_policy, lag_prob=0.75)

  return env

def make_carracing_trans_env(*args, init_rotation=None, **kwargs):
  """Make transfer env for rqst.reward_models.evaluate_reward_model
  """
  if init_rotation is None:
    init_rotation = 0.25 * np.pi

  trans_env = make_carracing_env(*args, **kwargs)

  # monkeypatch init rotation
  trans_env.orig_reset = trans_env.reset

  def reset(self, *args, **kwargs):
    self.orig_reset(*args, **kwargs)
    # rotate init car state
    self.car = Car(self.world, self.track[0][1] + init_rotation,
                   *self.track[0][2:4])
    return self.step(None)[0]

  trans_env.reset = types.MethodType(reset, trans_env)

  trans_env.default_init_obs = None

  return trans_env
