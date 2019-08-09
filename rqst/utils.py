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
"""Utility functions for TF and data collection/formatting"""

from __future__ import division

from copy import deepcopy
import collections
import json
import os
import random
import uuid

import gym
from IPython.core.display import display
from IPython.core.display import HTML
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf

inf = 999.

home_dir = os.path.join('/home', 'sid')
data_dir = os.path.join(home_dir, 'ReQueST', 'data')
carracing_data_dir = os.path.join(data_dir, 'carracing')
pointmass_data_dir = os.path.join(data_dir, 'pointmass')
bandit_data_dir = os.path.join(data_dir, 'bandit')
clfbandit_data_dir = os.path.join(data_dir, 'clfbandit')

# https://github.com/lucastheis/deepbelief/blob/master/data/mnist.npz
mnist_dir = os.path.join(home_dir, 'mnist')

# https://github.com/hardmaru/WorldModelsExperiments/tree/master/carracing
wm_dir = os.path.join(home_dir, 'WorldModelsExperiments', 'carracing')

for path in [
    carracing_data_dir, pointmass_data_dir, bandit_data_dir, clfbandit_data_dir
]:
  if not os.path.exists(path):
    os.makedirs(path)


def make_tf_session(gpu_mode=False):
  if not gpu_mode:
    kwargs = {'config': tf.ConfigProto(device_count={'GPU': 0})}
  else:
    kwargs = {}
  sess = tf.InteractiveSession(**kwargs)
  return sess


def get_tf_vars_in_scope(scope):
  return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


tf_init_vars_cache = {}


def init_tf_vars(sess, scopes=None, use_cache=True):
  """Initialize TF variables"""
  if scopes is None:
    sess.run(tf.global_variables_initializer())
  else:
    global tf_init_vars_cache
    init_ops = []
    for scope in scopes:
      if not use_cache or scope not in tf_init_vars_cache:
        tf_init_vars_cache[scope] = tf.initializers.variables(
            get_tf_vars_in_scope(scope))
      init_ops.append(tf_init_vars_cache[scope])
    sess.run(init_ops)


def save_tf_vars(sess, scope, save_path):
  """Save TF variables"""
  saver = tf.train.Saver(
      [v for v in tf.global_variables() if v.name.startswith(scope + '/')])
  saver.save(sess, save_path=save_path)


def load_tf_vars(sess, scope, load_path):
  """Load TF variables"""
  saver = tf.train.Saver(
      [v for v in tf.global_variables() if v.name.startswith(scope + '/')])
  saver.restore(sess, load_path)


def tf_logit(p):
  """Logit function"""
  return tf.log(p / (1. - p))


def tf_lognormal(y, mean, logstd):
  """Log-normal distribution"""
  log_sqrt_two_pi = np.log(np.sqrt(2.0 * np.pi))
  return -0.5 * ((y - mean) / tf.exp(logstd))**2 - logstd - log_sqrt_two_pi


def tf_clip(grad, grad_clip):
  """Gradient clipping"""
  if grad is None:
    return None
  return tf.clip_by_value(grad, -grad_clip, grad_clip)


def tf_normalize_logits(logits):
  """Normalize logits"""
  return logits - tf.reduce_logsumexp(logits, axis=1, keep_dims=True)


def tf_xent(labels, logits):
  """Cross entropy"""
  return -tf.reduce_sum(labels * tf_normalize_logits(logits), axis=1)


def normalize_logits(logits):
  """Logit normalization"""
  return logits - scipy.misc.logsumexp(logits, axis=1, keepdims=True)


def ens_disag(probs):
  """Represent ensemble disagreement"""
  mean = tf.reduce_mean(probs, axis=1, keep_dims=True)
  kl = tf.reduce_sum(
      probs * (tf.log(1e-9 + probs) - tf.log(1e-9 + mean)), axis=2)
  return tf.reduce_mean(kl, axis=1)


def np_ens_disag(probs):
  """Numpy version of ensemble disagreement"""
  mean = np.mean(probs, axis=1, keepdims=True)
  kl = np.sum(probs * (np.log(1e-9 + probs) - np.log(1e-9 + mean)), axis=2)
  return np.mean(kl, axis=1)


def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=1,
              size=256,
              activation=tf.nn.relu,
              output_activation=tf.nn.softmax):
  """Build MLP model"""
  out = input_placeholder
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    for _ in range(n_layers):
      out = tf.layers.dense(out, size, activation=activation)
    out = tf.layers.dense(out, output_size, activation=output_activation)
  return out


def onehot_encode(i, n):
  x = np.zeros(n)
  x[i] = 1
  return x


def process_frame(frame):
  """First preprocessing step for raw observations from the carracing env"""
  # frame = image with shape (96, 96, 3) returns (64, 64, 3) image
  obs = frame[0:84, :, :].astype(np.float) / 255.0
  obs = scipy.misc.imresize(obs, (64, 64))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs


def map_frames(rollouts, f, batch=False):
  """Apply f to all observations in rollouts"""
  if not batch:
    return [[(f(s), a, r, f(ns), d, i)
             for s, a, r, ns, d, i in rollout]
            for rollout in rollouts]

  obses = []
  for rollout in rollouts:
    for s, a, r, ns, d, i in rollout:
      obses.append(s)
  obses = np.array(obses)

  next_obses = []
  for rollout in rollouts:
    for s, a, r, ns, d, i in rollout:
      next_obses.append(ns)
  next_obses = np.array(next_obses)

  mapped_obses = f(obses)
  mapped_next_obses = f(next_obses)
  mapped_rollouts = deepcopy(rollouts)
  flat_idx = 0
  for rollout_idx, rollout in enumerate(mapped_rollouts):
    for t, (s, a, r, ns, d, i) in enumerate(rollout):
      mapped_rollouts[rollout_idx][t] = (mapped_obses[flat_idx], a, r,
                                         mapped_next_obses[flat_idx], d, i)
      flat_idx += 1
  return mapped_rollouts


def run_ep(policy,
           env,
           max_ep_len=None,
           proc_obs=(lambda x: x),
           render=False,
           **unused_kwargs):
  """Run episode"""
  if env.name == 'carracing' and not isinstance(env, 'DreamEnv'):
    proc_obs = process_frame

  if max_ep_len is None or max_ep_len > env.max_ep_len:
    max_ep_len = env.max_ep_len

  try:
    policy.reset()
  except:
    pass

  obs = proc_obs(env.reset())
  done = False
  prev_obs = deepcopy(obs)
  rollout = []
  for _ in range(max_ep_len):
    if done:
      break
    action = policy(prev_obs)
    obs, r, done, info = env.step(action)
    obs = proc_obs(obs)
    rollout.append(deepcopy((prev_obs, action, r, obs, float(done), info)))
    prev_obs = deepcopy(obs)
    if render:
      try:
        env.render()
      except NotImplementedError:
        pass
  return rollout


def vectorize_rollouts(rollouts, max_ep_len, preserve_trajs=False):
  """Unzips rollouts into separate arrays for obses, actions, etc."""
  # preserve_trajs = False -> flatten rollouts, lose episode structure
  data = {'obses': [], 'actions': [], 'rews': [], 'next_obses': [], 'dones': []}
  for rollout in rollouts:
    more_obses, more_actions, more_rews, more_next_obses, more_dones = [
        list(x) for x in zip(*rollout[:max_ep_len])
    ][:5]
    if preserve_trajs:
      more_obses = pad(np.array(more_obses), max_ep_len)
      more_actions = pad(np.array(more_actions), max_ep_len)
      more_rews = pad(np.array(more_rews), max_ep_len)
      more_next_obses = pad(np.array(more_next_obses), max_ep_len)
      more_dones = pad(np.array(more_dones), max_ep_len)
    data['obses'].append(more_obses)
    data['actions'].append(more_actions)
    data['rews'].append(more_rews)
    data['next_obses'].append(more_next_obses)
    data['dones'].append(more_dones)

  if not preserve_trajs:
    data = {k: sum(v, []) for k, v in data.items()}

  data = {k: np.array(v) for k, v in data.items()}

  if preserve_trajs:
    data['traj_lens'] = np.array([
        len(rollout[:max_ep_len]) + 1 for rollout in rollouts
    ])  # remember where padding begins

  idxes = list(range(len(data['obses'])))
  random.shuffle(idxes)
  data = {k: v[idxes] for k, v in data.items()}

  return data


def rollouts_of_traj_data(traj_data):
  """Inverse of vectorize_rollouts(, preserve_trajs=True)"""
  rollouts = []
  for i in range(traj_data['obses'].shape[0]):
    ep_len = traj_data['traj_lens'][i] - 1
    rollout = list(
        zip(traj_data['obses'][i, :ep_len], traj_data['actions'][i, :ep_len],
            traj_data['rews'][i, :ep_len], traj_data['next_obses'][i, :ep_len],
            traj_data['dones'][i, :ep_len], [{}] * ep_len))
    rollouts.append(rollout)
  return rollouts


def flatten_traj_data(traj_data):
  """Flattens a trajectory"""

  def _flatten(v):
    """Flatten dims"""
    try:
      n_dims = len(v.shape)
      if n_dims > 3:
        raise ValueError
      elif n_dims == 3:
        return v.reshape((v.shape[0] * v.shape[1], v.shape[2]))
      elif n_dims == 2:
        return v.ravel()
      else:
        return v
    except:
      return v

  return {k: _flatten(v) for k, v in traj_data.items()}


def traj_of_rollout(rollout):
  """Extract trajectory (observations) from rollout (observations and actions)"""
  traj = [x[0] for x in rollout]
  last_obs = rollout[-1][3]
  if last_obs is not None:
    traj.append(last_obs)
  return np.array(traj)


def act_seq_of_rollout(rollout):
  return np.array([x[1] for x in rollout])


def segment_trajs(trajs, act_seqs, seg_len=2):
  if seg_len < 2:
    raise ValueError

  segs = []
  act_segs = []
  for traj, act_seq in zip(trajs, act_seqs):
    curr_seg_len = min(seg_len, len(traj))
    for i in range(len(traj) - curr_seg_len + 1):
      segs.append(traj[i:i + curr_seg_len])
      act_segs.append(act_seq[i:i + curr_seg_len - 1])
  return segs, act_segs


def split_rollouts(rollouts, train_frac=0.9):
  """Train-test split
  Useful for sample_batch
  """
  idxes = list(range(rollouts['obses'].shape[0]))
  random.shuffle(idxes)
  n_train_examples = int(train_frac * len(idxes))
  train_idxes = idxes[:n_train_examples]
  val_idxes = idxes[n_train_examples:]

  rews = rollouts.get('rews', None)
  if rews is not None and len(rews.shape) != 2 and all(
      r is not None for r in rews):

    def proc_idxes(idxes):
      idxes_of_rew_class = collections.defaultdict(list)
      for idx in idxes:
        idxes_of_rew_class[float(rews[idx])].append(idx)
      idxes_of_rew_class = dict(idxes_of_rew_class)
      return idxes_of_rew_class

    train_idxes_of_rew_class = proc_idxes(train_idxes)
  else:
    train_idxes_of_rew_class = None

  if 'actions' in rollouts:
    actions = rollouts['actions']

    def proc_idxes(idxes):
      idxes_of_act = collections.defaultdict(list)
      for idx in idxes:
        idxes_of_act[float(np.argmax(actions[idx]))].append(idx)
      idxes_of_act = dict(idxes_of_act)
      return idxes_of_act

    train_idxes_of_act = proc_idxes(train_idxes)
  else:
    train_idxes_of_act = None

  rollouts.update({
      'train_idxes': train_idxes,
      'val_idxes': val_idxes,
      'train_idxes_of_rew_class': train_idxes_of_rew_class,
      'train_idxes_of_act': train_idxes_of_act
  })
  return rollouts


def pad(arr, max_len):
  """Zero-padding
  Useful for vectorize_rollouts(, preserve_trajs=True) and split_prefs"""
  n = arr.shape[0]
  if n > max_len:
    raise ValueError
  elif n == max_len:
    return arr
  else:
    shape = [max_len - n]
    shape.extend(arr.shape[1:])
    padding = np.zeros(shape)
    return np.concatenate((arr, padding), axis=0)


def build_mask(trajs, max_len):
  """Mask for traj data with variable-length trajs
  Assumes trajs are unpadded
  """
  return np.array([
      (np.arange(0, max_len, 1) < len(traj)).astype(np.float) for traj in trajs
  ])


def split_prefs(pref_logs, train_frac=0.9):
  """Pad, mask, train-test split pref data
  Analogous to split_rollouts
  """
  max_traj_len = max(
      len(traj) for traj in (pref_logs['trajs'] + pref_logs['ref_trajs']))
  data = {
      k: np.stack([pad(traj, max_traj_len) for traj in pref_logs[k]
                  ], axis=0) for k in ['trajs', 'ref_trajs']
  }
  data.update({
      k: np.stack([pad(act_seq, max_traj_len - 1) for act_seq in pref_logs[k]],
                  axis=0) for k in ['act_seqs', 'ref_act_seqs']
  })
  data['mask'] = build_mask(pref_logs['trajs'], max_traj_len - 1)
  data['ref_mask'] = build_mask(pref_logs['ref_trajs'], max_traj_len - 1)
  data['prefs'] = np.array(pref_logs['prefs'])
  idxes = list(range(data['ref_trajs'].shape[0]))
  random.shuffle(idxes)
  n_train_examples = int(train_frac * len(idxes))
  train_idxes = idxes[:n_train_examples]
  val_idxes = idxes[n_train_examples:]
  data.update({'train_idxes': train_idxes, 'val_idxes': val_idxes})
  return data


def elts_at_idxes(x, idxes):
  if type(x) == list:
    return [x[i] for i in idxes]
  else:
    return x[idxes]


def sample_batch(size, data, data_keys, idxes_key, class_idxes_key=None):
  if size < len(data[idxes_key]):
    if class_idxes_key is None:
      idxes = random.sample(data[idxes_key], size)
    else:
      # sample class-balanced batch
      idxes = []
      idxes_of_class = data[class_idxes_key]
      n_classes = len(idxes_of_class)
      for c, idxes_of_c in idxes_of_class.items():
        k = int(np.ceil(size / n_classes))
        if k > len(idxes_of_c):
          idxes_of_c_samp = idxes_of_c
        else:
          idxes_of_c_samp = random.sample(idxes_of_c, k)
        idxes.extend(idxes_of_c_samp)
      if len(idxes) > size:
        random.shuffle(idxes)
        idxes = idxes[:size]
  else:
    idxes = data[idxes_key]
  batch = {k: elts_at_idxes(data[k], idxes) for k in data_keys}
  return batch


def make_random_policy(env):
  policy = lambda _: env.action_space.sample()
  return policy


def plot_trajs(trajs, env, encoder=None, save_path=None):
  if env.name == 'pointmass':
    plt.scatter([env.goal[0]], [env.goal[1]],
                marker='o',
                facecolors='none',
                linestyle='--',
                edgecolor='green',
                linewidth=1,
                s=4000,
                alpha=1)

    plt.scatter([env.trap[0]], [env.trap[1]],
                marker='o',
                facecolors='none',
                linestyle='--',
                edgecolor='red',
                linewidth=1,
                s=4000,
                alpha=1)

    for traj in trajs:
      if type(traj) == list:
        x, y = list(zip(*traj))
      else:
        x = traj[:, 0]
        y = traj[:, 1]
      plt.scatter(
          x, y, c=range(len(x)), cmap=plt.cm.Blues, alpha=0.75, linewidth=0)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    if save_path is not None:
      plt.savefig(save_path, bbox_inches='tight', dpi=500, transparent=True)
    plt.show()
  elif env.name == 'bandit':
    obses = np.array(trajs)[:, 0, :]
    plt.scatter(obses[:, 0], obses[:, 1], linewidth=0)
    lows = env.observation_space.low
    highs = env.observation_space.high
    eps = 0.05
    plt.xlim([lows[0] - eps, highs[0] + eps])
    plt.ylim([lows[1] - eps, highs[1] + eps])
    plt.grid()
    plt.show()
  elif env.name == 'clfbandit':
    max_trajs_shown = 3
    trajs = trajs[:max_trajs_shown]
    obses = np.array([traj[0] for traj in trajs])
    if encoder is not None:
      obses = encoder.decode_batch_latents(obses)
    obses = obses[:, :, :, 0]
    for i in range(obses.shape[0]):
      plt.imshow(obses[i, :, :], cmap=plt.cm.binary)
      plt.grid()
      plt.xticks([])
      plt.yticks([])
      plt.show()
  elif env.name == 'carracing':
    for traj_eval in trajs:
      if encoder is not None:
        traj_eval = np.array(traj_eval)[:, :env.n_z_dim]
        frames = encoder.decode_batch_latents(traj_eval)
      else:
        frames = np.array(traj_eval)

      fig = plt.figure()
      ax = plt.axes()

      plt.axis('off')
      im = plt.imshow(frames[0], interpolation='none')
      plt.close()

      def init():
        im.set_data(frames[0])
        return [im]

      def animate(i):
        im.set_array(frames[i])
        return [im]

      anim = animation.FuncAnimation(
          fig,
          animate,
          init_func=init,
          frames=len(frames),
          interval=20,
          blit=True)

      display(HTML(anim.to_html5_video()))


def viz_rew_eval(rew_eval, env, encoder=None):
  """
  Args:
   rew_eval: output of a call to rqst.reward_models.evaluate_reward_model
  """
  if rew_eval.get('rew_pred_data', None) is not None:
    xs = rew_eval['rew_pred_data']['true_rews']
    ys = rew_eval['rew_pred_data']['pred_rews']
    plt.xlabel('True Reward')
    plt.ylabel('Predicted Reward')
    plt.scatter(xs, ys, linewidth=0, alpha=0.5)
    plt.show()

  if rew_eval.get('rollouts', None) is not None:
    rollouts = rew_eval['rollouts']
    if rollouts != []:
      trajs = [traj_of_rollout(rollout) for rollout in rollouts]
      plot_trajs(trajs, env, encoder=encoder)


def viz_query_data(query_data, env, encoder=None):
  if env.name in ['bandit', 'pointmass']:
    confs = []
    if query_data['demo_rollouts'] != []:
      confs.append(('demo_rollouts', 'Demo'))
    if query_data['sketch_rollouts'] != []:
      confs.append(('sketch_rollouts', 'Sketch'))

    for data_key, title in confs:
      obses = np.array([
          obs for rollout in query_data[data_key]
          for obs in traj_of_rollout(rollout)
      ])
      xs = obses[:, 0]
      ys = obses[:, 1]

      plt.title('%s Queries' % title)
      plt.scatter(
          xs,
          ys,
          c=list(range(len(xs))),
          alpha=0.5,
          linewidth=0,
          cmap=plt.cm.cool)

      lows = env.observation_space.low
      highs = env.observation_space.high
      eps = 0.05
      plt.xlim([lows[0] - eps, highs[0] + eps])
      plt.ylim([lows[1] - eps, highs[1] + eps])
      if env.name == 'bandit':
        plt.grid()
      plt.show()
  elif env.name == 'clfbandit':
    if query_data['pref_logs'] is not None:
      print('latest pref query:')
      plot_trajs(
          query_data['pref_logs']['ref_trajs'][-1:], env, encoder=encoder)
      print(
          np.argmax(query_data['pref_logs']['ref_act_seqs'][-1]),
          np.argmax(query_data['pref_logs']['act_seqs'][-1]))
    if query_data['demo_rollouts'] != []:
      print('latest demo queries:')
      plot_trajs([traj_of_rollout(r) for r in query_data['demo_rollouts'][-3:]],
                 env,
                 encoder=encoder)
    if query_data['sketch_rollouts'] != []:
      print('latest sketch queries:')
      plot_trajs(
          [traj_of_rollout(r) for r in query_data['sketch_rollouts'][-3:]],
          env,
          encoder=encoder)
  elif env.name == 'carracing':
    if query_data['pref_logs'] is not None:
      print('latest pref query:')
      plot_trajs([
          query_data['pref_logs']['ref_trajs'][-1],
          query_data['pref_logs']['trajs'][-1]
      ],
                 env,
                 encoder=encoder)
    if query_data['demo_rollouts'] != []:
      print('latest demo queries:')
      plot_trajs([traj_of_rollout(r) for r in query_data['demo_rollouts'][-4:]],
                 env,
                 encoder=encoder)
    if query_data['sketch_rollouts'] != []:
      print('latest sketch queries:')
      plot_trajs(
          [traj_of_rollout(r) for r in query_data['sketch_rollouts'][-4:]],
          env,
          encoder=encoder)


col_means = lambda x: np.nanmean(x, axis=0)
col_stderrs = lambda x: np.nanstd(
    x, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(x), axis=0))
err_bar_mins = lambda x: col_means(x) - col_stderrs(x)
err_bar_maxs = lambda x: col_means(x) + col_stderrs(x)


def make_perf_mat(perf_evals, y_key):
  n = len(perf_evals[0][y_key])
  max_len = max(len(perf_eval[y_key]) for perf_eval in perf_evals)

  def pad(lst, n):
    if len(lst) < n:
      lst += [np.nan] * (n - len(lst))
    return lst

  return np.array([pad(perf_eval[y_key], max_len) for perf_eval in perf_evals])


def smooth(xs, win=10):
  psums = np.concatenate((np.zeros(1), np.cumsum(xs)))
  rtn = (psums[win:] - psums[:-win]) / win
  rtn[0] = xs[0]
  return rtn


def plot_perf_evals(perf_evals,
                    x_key,
                    y_key,
                    label='',
                    smooth_win=None,
                    color=None):
  y_mat = make_perf_mat(perf_evals, y_key)
  y_mins = err_bar_mins(y_mat)
  y_maxs = err_bar_maxs(y_mat)
  y_means = col_means(y_mat)

  if smooth_win is not None:
    y_mins = smooth(y_mins, win=smooth_win)
    y_maxs = smooth(y_maxs, win=smooth_win)
    y_means = smooth(y_means, win=smooth_win)

  xs = max([perf_eval[x_key] for perf_eval in perf_evals], key=lambda x: len(x))
  xs = xs[:len(y_means)]

  kwargs = {}
  if color is not None:
    kwargs['color'] = color

  plt.fill_between(
      xs,
      y_mins,
      y_maxs,
      where=y_maxs >= y_mins,
      interpolate=True,
      label=label,
      alpha=0.5,
      **kwargs)
  plt.plot(xs, y_means, **kwargs)


def compute_perf_metrics(rollouts, env):
  metrics = {}
  metrics['rew'] = np.mean(
      [sum(r for s, a, r, ns, d, i in rollout) for rollout in rollouts])
  for key in ['succ', 'crash']:
    if env.only_terminal_reward:
      inds = [
          1 if rollout[-1][-1].get(key, False) else 0 for rollout in rollouts
      ]
    else:
      inds = [
          1 if x[-1].get(key, False) else 0
          for rollout in rollouts
          for x in rollout
      ]
    metrics[key] = np.mean(inds)
  metrics['rolloutlen'] = np.mean([len(rollout) for rollout in rollouts])
  return metrics


def evaluate_policy(sess,
                    env,
                    trans_env,
                    dynamics_model,
                    policy,
                    n_eval_rollouts=10,
                    dream=False):

  run_ep_func = run_ep if dream else dynamics_model.run_ep

  if dream:
    env = DreamEnv(env, dynamics_model)
  eval_rollouts = [
      run_ep_func(policy, env, render=False) for _ in range(n_eval_rollouts)
  ]
  perf = compute_perf_metrics(eval_rollouts, env)

  # transfer env
  if trans_env is not None:
    if dream:
      trans_env = DreamEnv(trans_env, dynamics_model)
    trans_rollouts = [
        run_ep_func(policy, trans_env, render=False)
        for _ in range(n_eval_rollouts)
    ]
    trans_perf = compute_perf_metrics(trans_rollouts, trans_env)
    perf.update({('trans_%s' % k): v for k, v in trans_perf.items()})
  else:
    trans_rollouts = []

  return {
      'perf': perf,
      'rollouts': eval_rollouts + trans_rollouts,
  }


def make_noisy_expert_policy(expert_policy, action_space, eps=0.5):

  def policy(obs):
    if np.random.random() < eps:
      return action_space.sample()
    else:
      return expert_policy(obs)

  return policy


class LaggyPolicy(object):

  def __init__(self, policy, lag_prob=0.5):
    self.policy = policy
    self.lag_prob = lag_prob
    self.last_action = None

  def reset(self):
    pass

  def __call__(self, obs):
    if self.last_action is None or np.random.random() > self.lag_prob:
      self.last_action = self.policy(obs)
    return self.last_action


def converged(val_losses, ftol, min_iters=2, eps=1e-9):
  return len(val_losses) >= max(2, min_iters) and (
      val_losses[-1] == np.nan or abs(val_losses[-1] - val_losses[-2]) /
      (eps + abs(val_losses[-2])) < ftol)


def isinstance(obj, cls_name):
  return obj.__class__.__name__ == cls_name


def load_wm_pretrained_model(jsonfile, scope, sess):
  with open(jsonfile, 'r') as f:
    params = json.load(f)

  t_vars = tf.trainable_variables(scope=scope)
  idx = 0
  for var in t_vars:
    pshape = tuple(var.get_shape().as_list())
    p = np.array(params[idx])
    assert pshape == p.shape, 'inconsistent shape'
    pl = tf.placeholder(tf.float32, pshape, var.name[:-2] + '_placeholder')
    assign_op = var.assign(pl)
    sess.run(assign_op, feed_dict={pl.name: p / 10000.})
    idx += 1


def unnormalize_obs(obs, env, eps=1e-9):
  if env.name in ['pointmass', 'bandit']:
    obs_low = env.observation_space.low - eps
    obs_range = env.observation_space.high + eps - obs_low
    p = (obs - obs_low) / obs_range
    return tf_logit(p)
  elif env.name in ['clfbandit', 'carracing']:
    return obs  # latents are unbounded
  else:
    raise ValueError


def normalize_obs(obs, env, eps=1e-2):
  if env.name in ['pointmass', 'bandit']:
    obs_low = env.observation_space.low - eps
    obs_range = env.observation_space.high + eps - obs_low
    return obs_low + tf.math.sigmoid(obs) * obs_range
  elif env.name in ['clfbandit', 'carracing']:
    return obs  # latents are unbounded
  else:
    raise ValueError


def unnormalize_act(act, env, eps=1e-2):
  if env.name in ['bandit', 'clfbandit']:
    return act
  else:
    act_low = env.action_space.low - eps
    act_range = env.action_space.high + eps - act_low
    p = (act - act_low) / act_range
    return tf_logit(p)


def normalize_act(act, env, eps=1e-2):
  if env.name in ['bandit', 'clfbandit']:
    return act
  else:
    act_low = env.action_space.low - eps
    act_range = env.action_space.high + eps - act_low
    return act_low + tf.math.sigmoid(act) * act_range


class LSTMCellWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell):

  def __init__(self, *args, **kwargs):
    super(LSTMCellWrapper, self).__init__(*args, **kwargs)
    self._inner_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(*args, **kwargs)

  @property
  def state_size(self):
    return self._inner_cell.state_size

  @property
  def output_size(self):
    return (self._inner_cell.state_size, self._inner_cell.output_size)

  def call(self, input, *args, **kwargs):
    output, next_state = self._inner_cell(input, *args, **kwargs)
    emit_output = (next_state, output)
    return emit_output, next_state


def rnn_encode_rollouts(raw_rollouts, env, encoder, dynamics_model):
  enc_rollouts = map_frames(
      raw_rollouts, encoder.encode_batch_frames, batch=True)
  enc_traj_data = split_rollouts(
      vectorize_rollouts(enc_rollouts, env.max_ep_len, preserve_trajs=True))
  return dynamics_model.rnn_encode_rollouts(enc_traj_data)


class DreamEnv(gym.Env):

  def __init__(self, env, dynamics_model):
    super(DreamEnv, self).__init__()
    self.__dict__.update(env.__dict__)
    self.env = env
    self.dynamics_model = dynamics_model

    self.using_rnn_dyn = isinstance(self.dynamics_model, 'MDNRNNDynamicsModel')

    self.curr_obs = None
    self.curr_hidden_state = None

  def reset(self):
    self.curr_obs = self.env.default_init_obs.ravel()
    if self.using_rnn_dyn:
      c = self.curr_obs[self.env.n_z_dim:self.env.n_z_dim +
                        self.dynamics_model.rnn_size]
      h = self.curr_obs[-self.env.rnn_size:]
      self.curr_hidden_state = (c[np.newaxis, :], h[np.newaxis, :])
    return self.curr_obs

  def step(self, act):
    act = act.ravel()
    obs = self.curr_obs
    if self.using_rnn_dyn:
      obs = obs[:self.env.n_z_dim]

    data = self.dynamics_model.compute_next_obs(
        obs[np.newaxis, :],
        act[np.newaxis, :],
        init_state=self.curr_hidden_state)

    prev_obs = deepcopy(self.curr_obs)
    self.curr_obs = data['next_obs'].ravel()
    if self.using_rnn_dyn:
      self.curr_hidden_state = data['next_state']
    r = self.env.reward_func(prev_obs[np.newaxis, :], act[np.newaxis, :],
                             self.curr_obs[np.newaxis, :])
    done = False
    prob_succ = self.env.prob_succ(self.curr_obs[np.newaxis, :])[0]
    prob_crash = self.env.prob_crash(self.curr_obs[np.newaxis, :])[0]
    info = {'succ': prob_succ, 'crash': prob_crash}

    return self.curr_obs, r, done, info


def rollout_in_dream(policy,
                     env,
                     dynamics_model,
                     init_obs=None,
                     max_ep_len=None):
  dream_env = DreamEnv(env, dynamics_model)
  if init_obs is not None:
    dream_env.env.default_init_obs = init_obs
  rollout = run_ep(policy, dream_env, max_ep_len=max_ep_len)
  init_act_seq = act_seq_of_rollout(rollout)
  init_traj = traj_of_rollout(rollout)
  return init_traj, init_act_seq


def bal_weights(elts):
  size = len(elts)
  weights = np.ones(size)
  idxes_of_elt = collections.defaultdict(list)
  for idx, elt in enumerate(elts):
    idxes_of_elt[elt].append(idx)
  n_classes = len(idxes_of_elt)
  for elt, idxes in idxes_of_elt.items():
    weights[idxes] = size / (n_classes * len(idxes))
  return weights


def opt_scope_of_obj(obj):
  return str(uuid.uuid4())
