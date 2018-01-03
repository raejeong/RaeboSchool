import numpy as np 
import tensorflow as tf 
import importlib
from algorithms.utils.utils import *
import itertools

class QNetwork:
  """
  Q Network for Continous Environments

  Note:
    Implementation of Deep Q Network with Priotized Experience Replay and Double Q Learning, Unlike the conventional DQN, this takes in action and state as input and outputs the Q value
  
  Args: 
    env: OpenAI environment object
    
    sess: TensorFlow Session object
        
    network_params (dict): Parameters to define the network used
      'network_type' (string): Defines the type of the network 'fully_connected_network' 
      'network_size' (string): Defines the size of the network 'small/medium/large'
    
    algorithm_params (dict): Parameters specific to the algorithm 
      'gamma' (double): Discount factor for MDP
      'learning_rate' (double): Learning rate for network gradient update
      'PER_size' (int): Experience replay buffer size 
      'PER_batch_size' (int): Experience replay buffer size 
      'PER_iterations' (int): Number of iterations to train from the experience replay buffer
      'PER_alpha' (double): Proportional prioritization constant 
      'PER_epsilon' (double): Small positive constant that ensures that no transition has zero priority
      'target_update_rate' (double): Rate to perform the soft target network update

  """
  def __init__(self,
               env,
               sess,
               network_params = {'network_type':'fully_connected_network',
                                 'network_size':'large'},
               algorithm_params = {'gamma':0.99,
                                   'learning_rate':1e-3,
                                   'target_network':True,
                                   'PER_size':200000, 
                                   'PER_batch_size':32, 
                                   'PER_iterations':300,
                                   'PER_alpha':0.6, 
                                   'PER_epsilon':0.01,
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

    # Experience replay buffer
    self.replay_buffer = PrioritizedExperienceReplay(self.algorithm_params['PER_size'], self.algorithm_params['PER_epsilon'], self.algorithm_params['PER_alpha'])

    ##### Placeholders #####

    # placeholder for learning rate for the optimizer
    self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    
    # Placeholder for observations and next observations
    self.observations = tf.placeholder(tf.float32, shape=[None, self.observation_shape], name="observations")
    self.target_observations = tf.placeholder(tf.float32, shape=[None, self.observation_shape], name="target_observations")
    
    # Placeholder for actions
    self.actions = tf.placeholder(tf.float32, shape=[None, self.action_shape], name="actions")
    self.target_actions = tf.placeholder(tf.float32, shape=[None, self.action_shape], name="target_actions")

    # Placeholder for Q values of the next observations
    self.target_q_values = tf.placeholder(tf.float32, shape=[None, 1], name="target_q_values")
        
    # Placeholder for the rewards
    self.rewards = tf.placeholder(tf.float32, shape=[None, 1], name="rewards")
    
    ##### Network #####
    
    # state action pair for both current Q network and target Q network
    self.state_action_pairs = tf.concat([self.observations, self.actions],1)  
    self.target_state_action_pairs = tf.concat([self.target_observations, self.target_actions],1)
    
    # Current Q network, outputs Q value of state action pair
    self.current_q_network = self.q_network_class.Network(self.sess, self.state_action_pairs, 1, "current_q_network", self.network_params['network_size'])
    self.current_q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_q_network')

    # Target Q network, outputs Q value of target state action pair
    self.target_q_network = self.q_network_class.Network(self.sess, self.target_state_action_pairs, 1, "target_q_network", self.network_params['network_size'])
    self.target_q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_network')

    # Best Q network, outputs Q value of target state action pair
    self.best_q_network = self.q_network_class.Network(self.sess, self.state_action_pairs, 1, "best_q_network", self.network_params['network_size'])
    self.best_q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='best_q_network')
    
    ##### Loss #####
    
    # Compute and log loss for Q network
    self.q_network_y = self.target_q_values
    self.q_network_loss = tf.reduce_mean(tf.squared_difference(self.current_q_network.out, self.q_network_y))
    tf.summary.scalar('q_network_loss', self.q_network_loss)

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

    self.backup_op = []
    for var, var_target in zip(sorted(self.target_q_network_vars,key=lambda v: v.name),sorted(self.best_q_network_vars, key=lambda v: v.name)):
      self.backup_op.append(var_target.assign(var))
    self.backup_op = tf.group(*self.backup_op)
    
    self.restore_op = []
    for var, var_target in zip(sorted(self.best_q_network_vars,key=lambda v: v.name),sorted(self.target_q_network_vars, key=lambda v: v.name)):
      self.restore_op.append(var_target.assign(var))
    for var, var_target in zip(sorted(self.best_q_network_vars,key=lambda v: v.name),sorted(self.current_q_network_vars, key=lambda v: v.name)):
      self.restore_op.append(var_target.assign(var))
    self.restore_op = tf.group(*self.restore_op)

    ##### Logging #####

    # Log useful information
    self.summary = tf.summary.merge_all()

    # Initialize all tf variables
    self.sess.run(tf.global_variables_initializer())

  # Compute Q value of observation and action
  def compute_q(self, observation, action):
    q_value = self.sess.run([self.current_q_network.out],{self.observations: observation[None], self.actions:action})[0]
    return q_value

  # Compute batch Q value of observations and actions
  def compute_q_batch(self, observations, actions):
    q_values = self.sess.run([self.current_q_network.out],{self.observations: observations, self.actions:actions})
    return q_values
  
  # Compute target Q value from current Q network
  def compute_target_q(self, observation, action):
    q_value = self.sess.run([self.target_q_network.out],{self.target_observations: observation[None], self.target_actions:action})[0]
    return q_value

  # Compute best Q value from current Q network
  def compute_best_q(self, observation, action):
    q_value = self.sess.run([self.best_q_network.out],{self.observations: observation[None], self.actions:action})[0]
    return q_value

  # Compute batch target Q value of observations and actions
  def compute_target_q_batch(self, observations, actions):
    q_values = self.sess.run([self.target_q_network.out],{self.target_observations: observations, self.target_actions:actions})
    return q_values

  # target_update the target networks
  def soft_target_update(self):
    _ = self.sess.run([self.target_update],{})

  # Restore the last best network
  def restore(self):
    _ = self.sess.run([self.restore_op],{})

  # Backup the target network
  def backup(self):
    _ = self.sess.run([self.backup_op],{})

  # Add batch to replay buffer
  def replay_buffer_add_batch(self, batch_size, observations_batch, actions_batch, rewards_batch, next_observations_batch, y):
    for i in range(batch_size):
      q_value_estimate = self.compute_q(observations_batch[i,:], actions_batch[i,:][None])
      error = y[i,:][0] - q_value_estimate[0][0]
      sample_ER = ExperienceReplayData(observation=observations_batch[i,:], next_observation=next_observations_batch[i,:], action=actions_batch[i,:], y=y, error=error, reward=rewards_batch[i,:])
      self.replay_buffer.add(error, sample_ER)

  # Update batch to replay buffer
  def replay_buffer_update_batch(self, batches):
    for sample_batch in batches:
      for sample in sample_batch:
        # Update error for the sample batch 
        q_value_estimate = self.compute_q(sample[1].observation, sample[1].action[None])
        error = sample[1].y - q_value_estimate
        self.replay_buffer.update(sample[0], error)

  def get_batches(self):
    batches = []
    for i in range(self.algorithm_params['PER_iterations']):
      sample_batch = self.replay_buffer.sample(self.algorithm_params['PER_batch_size'])
      batches.append(sample_batch)
    return batches

  # Train the Q network from the given batches
  def train(self, batches, learning_rate):
    self.algorithm_params['learning_rate'] = learning_rate
    summaries = []
    losses = []
    stats = {}
    # Train with Experience Replay
    for i in range(20):
      for sample_batch in batches:
        observations_mini_batch, actions_mini_batch, rewards_mini_batch, next_observations_mini_batch,y_mini_batch = [], [], [], [], []
        for sample in sample_batch:
          # Build the mini batch from sample batch
          observations_mini_batch.append(sample[1].observation)
          next_observations_mini_batch.append(sample[1].next_observation)
          actions_mini_batch.append(sample[1].action)
          rewards_mini_batch.append(sample[1].reward)
          y_mini_batch.append(sample[1].y)

        observations_mini_batch = np.array(observations_mini_batch)
        next_observations_mini_batch = np.array(next_observations_mini_batch)
        actions_mini_batch = np.array(actions_mini_batch)
        rewards_mini_batch = np.array(rewards_mini_batch)
        y_mini_batch = np.array(y_mini_batch).reshape([-1,1])

        # Training with sample batch
        summary, q_network_loss, _ = self.sess.run([self.summary, self.q_network_loss, self.train_q_network], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.rewards:rewards_mini_batch, self.target_q_values:y_mini_batch, self.learning_rate:self.algorithm_params['learning_rate']})
        summaries.append(summary)
        losses.append(q_network_loss)

    stats['q_network_loss'] = np.mean(np.array(losses))
    self.soft_target_update()

    return [summaries, stats]

  def train_current(self, batch_size, observations_batch, actions_batch, rewards_batch, y, learning_rate):
    self.algorithm_params['learning_rate'] = learning_rate
    summaries = []
    losses = []
    stats = {}
    for i in range(100):
      mini_batch_idx = np.random.choice(batch_size, 128)
      observations_mini_batch = observations_batch[mini_batch_idx,:]
      actions_mini_batch = actions_batch[mini_batch_idx,:]
      rewards_mini_batch = rewards_batch[mini_batch_idx,:]
      y_mini_batch = y[mini_batch_idx,:]

      summary, q_network_loss, _ = self.sess.run([self.summary, self.q_network_loss, self.train_q_network], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.rewards:rewards_mini_batch, self.target_q_values:y_mini_batch, self.learning_rate:self.algorithm_params['learning_rate']})
      summaries.append(summary)
      losses.append(q_network_loss)

    for i in range(5000):
      mini_batch_idx = np.random.choice(batch_size, 128)
      observations_mini_batch = observations_batch[mini_batch_idx,:]
      actions_mini_batch = actions_batch[mini_batch_idx,:]
      rewards_mini_batch = rewards_batch[mini_batch_idx,:]
      y_mini_batch = y[mini_batch_idx,:]

      summary, q_network_loss, _ = self.sess.run([self.summary, self.q_network_loss, self.train_q_network], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.rewards:rewards_mini_batch, self.target_q_values:y_mini_batch, self.learning_rate:self.algorithm_params['learning_rate']})
      summaries.append(summary)
      losses.append(q_network_loss)
      if np.mean(np.array(losses)) < 100:
        break


    # for i in range(50):
    #   summary, q_network_loss, _ = self.sess.run([self.summary, self.q_network_loss, self.train_q_network], {self.observations:observations_batch, self.actions:actions_batch, self.rewards:rewards_batch, self.target_q_values:y, self.learning_rate:self.algorithm_params['learning_rate']})
    #   summaries.append(summary)
    #   losses.append(q_network_loss)

    # for i in range(5000):
    # #   # mini_batch_idx = np.random.choice(batch_size, 128)
    # #   # observations_mini_batch = observations_batch[mini_batch_idx,:]
    # #   # actions_mini_batch = actions_batch[mini_batch_idx,:]
    # #   # rewards_mini_batch = rewards_batch[mini_batch_idx,:]
    # #   # y_mini_batch = y[mini_batch_idx,:]

    # #   # summary, q_network_loss, _ = self.sess.run([self.summary, self.q_network_loss, self.train_q_network], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.rewards:rewards_mini_batch, self.target_q_values:y_mini_batch, self.learning_rate:self.algorithm_params['learning_rate']})
    # #   # summaries.append(summary)
    # #   # losses.append(q_network_loss)

    #   summary, q_network_loss, _ = self.sess.run([self.summary, self.q_network_loss, self.train_q_network], {self.observations:observations_batch, self.actions:actions_batch, self.rewards:rewards_batch, self.target_q_values:y, self.learning_rate:self.algorithm_params['learning_rate']})
    #   summaries.append(summary)
    #   losses.append(q_network_loss)
    #   if np.mean(np.array(losses)) < 200:
    #     break

    stats['q_network_loss'] = np.mean(np.array(losses))
    self.soft_target_update()

    return [summaries, stats]
