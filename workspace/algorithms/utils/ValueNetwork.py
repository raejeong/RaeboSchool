import numpy as np 
import tensorflow as tf 
import importlib
from algorithms.utils.utils import *
import itertools

class ValueNetwork:
  """
  Value Network for Estimating the Return for Policy Gradient

  Note:
    Simple value network to estimate the return

  Args: 
    env: OpenAI environment object
    
    sess: TensorFlow Session object
        
    network_params (dict): Parameters to define the network used
      'network_type' (string): Defines the type of the network 'fully_connected_network' 
      'network_size' (string): Defines the size of the network 'small/medium/large'
    
    algorithm_params (dict): Parameters specific to the algorithm 
      'gamma' (double): Discount factor for MDP
      'learning_rate' (double): Learning rate for network gradient update
      'target_update_rate' (double): Rate to perform the soft target network update

  """
  def __init__(self,
               env,
               sess,
               network_params = {'network_type':'fully_connected_network',
                                 'network_size':'large'},
               algorithm_params = {'gamma':0.99,
                                   'learning_rate':1e-3,
                                   'target_update_rate':0.001}):

    # Tensorflow Session
    self.sess = sess
    
    # OpenAI Environment
    self.env = env
    
    # Hyper Parameters
    self.network_params = network_params
    self.algorithm_params = algorithm_params
    
    # Importing the desired network architecture
    self.value_network_class = importlib.import_module("architectures." + self.network_params['network_type'])

    # Getting the shape of observation space and action space of the environment
    self.observation_shape = self.env.observation_space.shape[0]
    self.action_shape = self.env.action_space.shape[0]

    ##### Placeholders #####

    # placeholder for learning rate for the optimizer
    self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    
    # Placeholder for observations and next observations
    self.observations = tf.placeholder(tf.float32, shape=[None, self.observation_shape], name="observations")

    # Placeholder for observations and next observations
    self.target_observations = tf.placeholder(tf.float32, shape=[None, self.observation_shape], name="target_observations")
            
    # Placeholder for the returns
    self.returns = tf.placeholder(tf.float32, shape=[None, 1], name="returns")
    
    ##### Network #####
    
    # Current Value network, outputs value current observation
    self.current_value_network = self.value_network_class.Network(self.sess, self.observations, 1, "current_value_network", self.network_params['network_size'])
    self.current_value_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_value_network')

    # Target Q network, outputs value of target observation
    self.target_value_network = self.value_network_class.Network(self.sess, self.target_observations, 1, "target_value_network", self.network_params['network_size'])
    self.target_value_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_value_network')

    # Best Value network, outputs value of observation
    self.best_value_network = self.value_network_class.Network(self.sess, self.observations, 1, "best_value_network", self.network_params['network_size'])
    self.best_value_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='best_value_network')

    ##### Loss #####
    
    # Compute and log loss for the value network
    self.value_network_loss = tf.reduce_mean(tf.squared_difference(self.current_value_network.out, self.returns))
    self.summary = tf.summary.scalar('value_network_loss', self.value_network_loss)

    ##### Optimization #####
    
    # Optimizers for the network
    self.value_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    
    # Training operation for the network
    self.train_value_network = self.value_network_optimizer.minimize(self.value_network_loss)
    
    ##### Target Update #####
    
    # Soft target update operations for the Q network
    # self.target_update = []
    # tau = self.algorithm_params['target_update_rate']
    # for var, var_target in zip(sorted(self.current_value_network_vars,key=lambda v: v.name),sorted(self.target_value_network_vars, key=lambda v: v.name)):
    #  self.target_update.append(var_target.assign((1. - tau) * var_target + tau * var))
    # self.target_update = tf.group(*self.target_update)

    self.target_update = []
    for var, var_target in zip(sorted(self.current_value_network_vars,key=lambda v: v.name),sorted(self.target_value_network_vars, key=lambda v: v.name)):
      self.target_update.append(var_target.assign(var))
    self.target_update = tf.group(*self.target_update)

    self.backup_op = []
    for var, var_target in zip(sorted(self.target_value_network_vars,key=lambda v: v.name),sorted(self.best_value_network_vars, key=lambda v: v.name)):
      self.backup_op.append(var_target.assign(var))
    self.backup_op = tf.group(*self.backup_op)
    
    self.restore_op = []
    for var, var_target in zip(sorted(self.best_value_network_vars,key=lambda v: v.name),sorted(self.target_value_network_vars, key=lambda v: v.name)):
      self.restore_op.append(var_target.assign(var))
    for var, var_target in zip(sorted(self.best_value_network_vars,key=lambda v: v.name),sorted(self.current_value_network_vars, key=lambda v: v.name)):
      self.restore_op.append(var_target.assign(var))
    self.restore_op = tf.group(*self.restore_op)
    
    ##### Logging #####

    # Log useful information
    # self.summary = tf.summary.merge_all()

    # Initialize all tf variables
    # self.sess.run(tf.global_variables_initializer())

# Compute best value of an observation
  def compute_value_best(self, observations):
    values = self.sess.run([self.best_value_network.out],{self.observations: observations})
    return values

    # Compute value of an observation
  def compute_value(self, observations):
    values = self.sess.run([self.target_value_network.out],{self.target_observations: observations})
    return values
  
  # Target_update the target network
  def soft_target_update(self):
    _ = self.sess.run([self.target_update],{})

  # Restore the last best network
  def restore(self):
    _ = self.sess.run([self.restore_op],{})

  # Backup the target network
  def backup(self):
    _ = self.sess.run([self.backup_op],{})

  # Train the Q network from the given batches
  def train(self, batch_size, observations_batch, returns_batch, learning_rate):
    self.algorithm_params['learning_rate'] = learning_rate
    summaries = []
    losses = []
    stats = {}
    for i in range(100):
      mini_batch_idx = np.random.choice(batch_size, 128)
      observations_mini_batch = observations_batch[mini_batch_idx,:]
      returns_mini_batch = returns_batch[mini_batch_idx,:]

      summary, value_network_loss, _ = self.sess.run([self.summary, self.value_network_loss, self.train_value_network], {self.observations:observations_mini_batch, self.returns:returns_mini_batch, self.learning_rate:self.algorithm_params['learning_rate']})
      summaries.append(summary)
      losses.append(value_network_loss)

    for i in range(100):
      mini_batch_idx = np.random.choice(batch_size, 128)
      observations_mini_batch = observations_batch[mini_batch_idx,:]
      returns_mini_batch = returns_batch[mini_batch_idx,:]

      summary, value_network_loss, _ = self.sess.run([self.summary, self.value_network_loss, self.train_value_network], {self.observations:observations_mini_batch, self.returns:returns_mini_batch, self.learning_rate:self.algorithm_params['learning_rate']})
      summaries.append(summary)
      losses.append(value_network_loss)
      if np.mean(np.array(losses)) < 100:
        break
    
    # for i in range(50):
    #   # Training with batch
    #   summary, value_network_loss, _ = self.sess.run([self.summary, self.value_network_loss, self.train_value_network], {self.observations:observations_batch, self.returns:returns_batch, self.learning_rate:self.algorithm_params['learning_rate']})
    #   summaries.append(summary)
    #   losses.append(value_network_loss)
    # for i in range(5000):
    # #   # Training with batch
    #   summary, value_network_loss, _ = self.sess.run([self.summary, self.value_network_loss, self.train_value_network], {self.observations:observations_batch, self.returns:returns_batch, self.learning_rate:self.algorithm_params['learning_rate']})
    #   summaries.append(summary)
    #   losses.append(value_network_loss)
    # #   # mini_batch_idx = np.random.choice(batch_size, 128)
    # #   # observations_mini_batch = observations_batch[mini_batch_idx,:]
    # #   # returns_mini_batch = returns_batch[mini_batch_idx,:]
    #   # summary, value_network_loss, _ = self.sess.run([self.summary, self.value_network_loss, self.train_value_network], {self.observations:observations_mini_batch, self.returns:returns_mini_batch, self.learning_rate:self.algorithm_params['learning_rate']})
    #   # summaries.append(summary)
    #   # losses.append(value_network_loss)
    #   if np.mean(np.array(losses)) < 200:
    #     break

    stats['value_network_loss'] = np.mean(np.array(losses))
    self.soft_target_update()

    return [summaries, stats]

  # Train the Q network from the given batches
  def train_once(self, batch_size, observations_batch, returns_batch, learning_rate):
    self.algorithm_params['learning_rate'] = learning_rate
    summaries = []
    losses = []
    stats = {}    
    for i in range(1):
      # Training with batch
      summary, value_network_loss, _ = self.sess.run([self.summary, self.value_network_loss, self.train_value_network], {self.observations:observations_batch, self.returns:returns_batch, self.learning_rate:self.algorithm_params['learning_rate']})
      summaries.append(summary)
      losses.append(value_network_loss)

    stats['value_network_loss'] = np.mean(np.array(losses))
    self.soft_target_update()

    return [summaries, stats]