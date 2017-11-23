import numpy as np 
import tensorflow as tf 
import importlib
from algorithms.utils.utils import *
import itertools

class PolicyNetwork:
  """
  Policy Network with Policy Gradient

  Note:
    Implementation of Adavantage Policy Gradient
  
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
                                   'target_update_rate':0.001,
                                   'std_dev':['fixed', 0.2],
                                   'number_of_suggestions':10}):

    # Tensorflow Session
    self.sess = sess
    
    # OpenAI Environment
    self.env = env
    
    # Hyper Parameters
    self.network_params = network_params
    self.algorithm_params = algorithm_params
    
    # Importing the desired network architecture
    self.policy_network_class = importlib.import_module("architectures." + self.network_params['network_type'])

    # Getting the shape of observation space and action space of the environment
    self.observation_shape = self.env.observation_space.shape[0]
    self.action_shape = self.env.action_space.shape[0]

    ##### Placeholders #####

    # placeholder for learning rate for the optimizer
    self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    
    # Placeholder for observations
    self.observations = tf.placeholder(tf.float32, shape=[None, self.observation_shape], name="observations")
    
    # Placeholder for actions taken, this is used for the policy gradient
    self.actions = tf.placeholder(tf.float32, shape=[None, self.action_shape], name="actions")
            
    # Placeholder for the advantage function
    self.advantages = tf.placeholder(tf.float32, shape=[None, 1], name="advantages") 

    self.average_advantage = tf.reduce_mean(self.advantages)
    
    ##### Networks #####
      
    # Define the output shape of the policy networks
    if self.algorithm_params['std_dev'][0] == 'network':
      policy_output_shape = self.action_shape*2
    else:
      policy_output_shape = self.action_shape

    # Current policy that will be updated and used to act, if the updated policy performs worse, the policy will be target_updated from the back up policy. 
    self.current_policy_network = self.policy_network_class.Network(self.sess, self.observations, policy_output_shape, "current_policy_network", self.network_params['network_size'])
    self.current_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_policy_network')

    # Backup of the target policy network so far, backs up the target policy in case the update is bad. outputs the mean for the action
    self.target_policy_network = self.policy_network_class.Network(self.sess, self.observations, policy_output_shape, "target_policy_network", self.network_params['network_size'])
    self.target_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_policy_network')

    # Policy network from last update, used for KL divergence calculation, outputs the mean for the action
    self.last_policy_network = self.policy_network_class.Network(self.sess, self.observations, policy_output_shape, "last_policy_network", self.network_params['network_size'])
    self.last_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='last_policy_network')

    ##### Policy Action Probability #####
      
    # Isolating the mean outputted by the policy network
    self.current_mean_policy = tf.reshape(tf.squeeze(self.current_policy_network.out[:,:self.action_shape]),[-1, self.action_shape])
    self.target_mean_policy = tf.reshape(tf.squeeze(self.target_policy_network.out[:,:self.action_shape]),[-1, self.action_shape])
    self.last_mean_policy = tf.reshape(tf.squeeze(self.last_policy_network.out[:,:self.action_shape]),[-1, self.action_shape])
    
    # Isolating the std dev outputted by the policy network, softplus is used to make sure that the std dev is positive
    if self.algorithm_params['std_dev'][0]=='network':
      self.current_std_dev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.current_policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
      self.target_std_dev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.target_policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
      self.last_std_dev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.last_policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
    else:
      self.current_std_dev_policy = tf.constant(self.algorithm_params['std_dev'][1])
      self.target_std_dev_policy = tf.constant(self.algorithm_params['std_dev'][1])
      self.last_std_dev_policy = tf.constant(self.algorithm_params['std_dev'][1])
    
    # Gaussian distribution is built with mean and std dev from the policy network
    self.current_gaussian_policy_distribution = tf.contrib.distributions.Normal(self.current_mean_policy, self.current_std_dev_policy)
    self.target_gaussian_policy_distribution = tf.contrib.distributions.Normal(self.target_mean_policy, self.target_std_dev_policy)
    self.last_gaussian_policy_distribution = tf.contrib.distributions.Normal(self.last_mean_policy, self.last_std_dev_policy)
    
    # Compute and log the KL divergence from last policy distribution to the current policy distribution
    self.kl = tf.reduce_mean(tf.contrib.distributions.kl_divergence(self.current_gaussian_policy_distribution, self.last_gaussian_policy_distribution))
    # tf.summary.scalar('kl', self.kl)
    
    # Action suggested by the current policy network
    number_of_suggestions = self.algorithm_params['number_of_suggestions']
    self.suggested_actions_out = tf.reshape(self.target_gaussian_policy_distribution._sample_n(number_of_suggestions),[-1,number_of_suggestions,self.action_shape])

    self.actions_out = tf.reshape(self.target_gaussian_policy_distribution._sample_n(1),[-1,self.action_shape])

    ##### Loss #####
    
    # Compute and log loss for policy network
    self.negative_log_prob = -self.current_gaussian_policy_distribution.log_prob(self.actions)
    self.policy_network_losses = self.negative_log_prob*self.advantages # be careful with this operation, it should be element wise not matmul!
    self.policy_network_loss = tf.reduce_mean(self.policy_network_losses)
    self.summary = tf.summary.scalar('policy_network_loss', self.policy_network_loss)

    ##### Optimization #####
    
    # Optimizers for the network
    self.policy_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    
    # Training operation for the network
    self.train_policy_network = self.policy_network_optimizer.minimize(self.policy_network_loss)
    
    ##### Target Update #####
    
    # Soft target update operations for the policy network
    # self.target_update = []
    # tau = self.algorithm_params['target_update_rate']
    # for var, var_target in zip(sorted(self.current_policy_network_vars,key=lambda v: v.name),sorted(self.target_policy_network_vars, key=lambda v: v.name)):
    #  self.target_update.append(var_target.assign((1. - tau) * var_target + tau * var))
    # self.target_update = tf.group(*self.target_update)

    self.target_update = []
    for var, var_target in zip(sorted(self.current_policy_network_vars,key=lambda v: v.name),sorted(self.target_policy_network_vars, key=lambda v: v.name)):
      self.target_update.append(var_target.assign(var))
    self.target_update = tf.group(*self.target_update)

    # Copy over the current policy network to last policy network for KL divergence Calculation
    self.update_last = []
    for var, var_target in zip(sorted(self.current_policy_network_vars,key=lambda v: v.name),sorted(self.last_policy_network_vars, key=lambda v: v.name)):
      self.update_last.append(var_target.assign(var))
    self.update_last = tf.group(*self.update_last)

    ##### Logging #####

    # Log useful information
    # self.summary = tf.summary.merge_all()

    # Initialize all tf variables
    # self.sess.run(tf.global_variables_initializer())

  # Compute suggested actions from observation
  def compute_suggested_actions(self, observation):
    suggested_actions = self.sess.run([self.suggested_actions_out], {self.observations: observation[None]})
    actions = []
    for i in range(self.algorithm_params['number_of_suggestions']):
      actions.append(suggested_actions[0][:,i,:])
    return actions

  # Compute suggested actions from observation
  def compute_action(self, observation):
    action = self.sess.run([self.actions_out], {self.observations: observation[None]})[0]
    return action
  
  # update the target network
  def soft_target_update(self):
    _ = self.sess.run([self.target_update],{})

  # update the last policy network
  def update_last_policy(self):
    _ = self.sess.run([self.update_last],{})

  # Train the Q network from the given batches
  def train(self, observations_batch, advantages_batch, actions_batch, learning_rate):
    self.algorithm_params['learning_rate'] = learning_rate
    summaries = []
    stats = {}

    # Taking the gradient step to optimize (train) the policy network.
    policy_network_losses, policy_network_loss, _ = self.sess.run([self.policy_network_losses, self.policy_network_loss, self.train_policy_network], {self.observations:observations_batch, self.actions:actions_batch, self.advantages:advantages_batch, self.learning_rate:self.algorithm_params['learning_rate']})

    # Get stats
    summary, average_advantage, kl  = self.sess.run([self.summary, self.average_advantage, self.kl], {self.observations:observations_batch, self.actions:actions_batch, self.advantages:advantages_batch})

    # Backup the current policy network to last policy network
    self.update_last_policy()
    
    # Shape of the policy_network_losses is check since its common to have nonsense size due to unintentional matmul instead of non element wise multiplication. Action dimension and advantage dimension are very important and hard to debug when not correct
    if isinstance(policy_network_losses, list):
      policy_network_losses = policy_network_losses[0]
    assert policy_network_losses.shape==actions_batch.shape, "Dimensions mismatch. Policy Distribution is incorrect! " + str(policy_network_losses.shape)

    summaries.append(summary)
    stats['policy_network_loss'] = policy_network_loss
    stats['kl'] = kl
    stats['average_advantage'] = average_advantage

    self.soft_target_update()
 
    return [summaries, stats]