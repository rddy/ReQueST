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

import pickle
import uuid
import os

from matplotlib import pyplot as plt
import tensorflow as tf

from rqst import utils


class TFModel(object):

  def __init__(self,
               sess,
               env,
               scope_file=None,
               tf_file=None,
               scope=None,
               *args,
               **kwargs):

    # scope vs. scope in scope_file
    if scope is None:
      if scope_file is not None and os.path.exists(scope_file):
        with open(scope_file, 'rb') as f:
          scope = pickle.load(f)
      else:
        scope = str(uuid.uuid4())

    self.env = env
    self.sess = sess
    self.tf_file = tf_file
    self.scope_file = scope_file
    self.scope = scope

    self.loss = None

  def save(self):
    if self.scope_file is None:
      return

    with open(self.scope_file, 'wb') as f:
      pickle.dump(self.scope, f, pickle.HIGHEST_PROTOCOL)

    utils.save_tf_vars(self.sess, self.scope, self.tf_file)

  def load(self):
    if self.scope_file is None:
      return

    with open(self.scope_file, 'rb') as f:
      self.scope = pickle.load(f)

    utils.load_tf_vars(self.sess, self.scope, self.tf_file)

  def init_tf_vars(self):
    utils.init_tf_vars(self.sess, [self.scope])

  def compute_batch_loss(self, feed_dict, update=True):
    if update:
      loss_eval, _ = self.sess.run([self.loss, self.update_op],
                                   feed_dict=feed_dict)
    else:
      loss_eval = self.sess.run(self.loss, feed_dict=feed_dict)
    return loss_eval

  def train(self,
            data,
            iterations=100000,
            ftol=1e-4,
            batch_size=32,
            learning_rate=1e-3,
            val_update_freq=100,
            verbose=False):

    if self.loss is None:
      return

    opt_scope = utils.opt_scope_of_obj(self)
    with tf.variable_scope(opt_scope, reuse=tf.AUTO_REUSE):
      self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    utils.init_tf_vars(self.sess, [self.scope, opt_scope])

    val_losses = []
    val_batch = utils.sample_batch(
        size=len(data['val_idxes']),
        data=data,
        data_keys=self.data_keys,
        idxes_key='val_idxes')

    if verbose:
      print('iters total_iters train_loss val_loss')

    for t in range(iterations):
      batch = utils.sample_batch(
          size=batch_size,
          data=data,
          data_keys=self.data_keys,
          idxes_key='train_idxes')

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
