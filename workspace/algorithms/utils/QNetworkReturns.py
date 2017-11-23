import numpy as np 
import tensorflow as tf 
import importlib
from algorithms.utils.utils import *
import itertools

class QNetwork:
  """
  Q Network for Estimating the Return for Policy Gradient

  Note:
    Simple q network to estimate the return

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
    self.q_network_class = importlib.import_module("architectures." + self.network_params['network_type'])

    # Getting the shape of observation space and action space of the environment
    self.observation_shape = self.env.observation_space.shape[0]
    self.action_shape = self.env.action_space.shape[0]

    ##### Placeholders #####

    # placeholder for learning rate for the optimizer
    self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    
    # Placeholder for observations and next observations
    self.observations = tf.placeholder(tf.float32, shape=[None, self.observation_shape], name="observations")

    # Placeholder for actions
    self.actions = tf.placeholder(tf.float32, shape=[None, self.action_shape], name="actions")
            
    # Placeholder for the returns
    self.returns = tf.placeholder(tf.float32, shape=[None, 1], name="returns")
    
    ##### Network #####

    # state action pair for both current Q network and target Q network
    self.state_action_pairs = tf.concat([self.observations, self.actions],1)
    
    # Current q network, outputs q current observation
    self.current_q_network = self.q_network_class.Network(self.sess, self.state_action_pairs, 1, "current_q_network", self.network_params['network_size'])
    self.current_q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_q_network')

    # Target Q network, outputs q of target observation
    self.target_q_network = self.q_network_class.Network(self.sess, self.state_action_pairs, 1, "target_q_network", self.network_params['network_size'])
    self.target_q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_network')

    ##### Loss #####
    
    # Compute and log loss for the q network
    self.q_network_loss = tf.reduce_mean(tf.squared_difference(self.current_q_network.out, self.returns))
    self.summary = tf.summary.scalar('q_network_loss', self.q_network_loss)

    ##### Optimization #####
    
    # Optimizers for the network
    self.q_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    
    # Training operation for the network
    self.train_q_network = self.q_network_optimizer.minimize(self.q_network_loss)
    
    ##### Target Update #####
    
    # Soft target update operations for the Q network
    # self.target_update = []
    # tau = self.algorithm_params['target_update_rate']
    # for var, var_target in zip(sorted(self.current_q_network_vars,key=lambda v: v.name),sorted(self.target_q_network_vars, key=lambda v: v.name)):
    #  self.target_update.append(var_target.assign((1. - tau) * var_target + tau * var))
    # self.target_update = tf.group(*self.target_update)

    self.target_update = []
    for var, var_target in zip(sorted(self.current_q_network_vars,key=lambda v: v.name),sorted(self.target_q_network_vars, key=lambda v: v.name)):
      self.target_update.append(var_target.assign(var))
    self.target_update = tf.group(*self.target_update)

    ##### Logging #####

    # Log useful information
    # self.summary = tf.summary.merge_all()

    # Initialize all tf variables
    # self.sess.run(tf.global_variables_initializer())

  # Compute q of an observation
  def compute_q(self, observations, actions):
    qs = self.sess.run([self.target_q_network.out],{self.observations: observations, self.actions:actions})
    return qs
  
  # target_update the target network
  def soft_target_update(self):
    _ = self.sess.run([self.target_update],{})

  # Train the Q network from the given batches
  def train(self, observations_batch, returns_batch, actions_batch, learning_rate):
    self.algorithm_params['learning_rate'] = learning_rate
    summaries = []
    losses = []
    stats = {}
    for i in range(300):
      # Training with batch
      summary, q_network_loss, _ = self.sess.run([self.summary, self.q_network_loss, self.train_q_network], {self.observations:observations_batch, self.returns:returns_batch, self.actions:actions_batch, self.learning_rate:self.algorithm_params['learning_rate']})
      summaries.append(summary)
      losses.append(q_network_loss)
    stats['q_network_loss'] = np.mean(np.array(losses))
    self.soft_target_update()

    return [summaries, stats]