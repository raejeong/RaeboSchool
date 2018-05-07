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
    self.dtype = tf.float64
    
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
    self.state_shape = 3

    self.std_dev = self.algorithm_params['std_dev'][1]

    ##### Placeholders #####

    # placeholder for learning rate for the optimizer
    self.learning_rate = tf.placeholder(self.dtype, name="learning_rate")

    self.std_dev_ph = tf.placeholder(self.dtype, name="std_dev")
    
    # Placeholder for observations
    self.observations = tf.placeholder(self.dtype, shape=[None, self.observation_shape], name="observations")
    
    # Placeholder for actions taken, this is used for the policy gradient
    self.actions = tf.placeholder(self.dtype, shape=[None, self.action_shape], name="actions")
            
    # Placeholder for the advantage function
    self.advantages = tf.placeholder(self.dtype, shape=[None, 1], name="advantages") 

    self.average_advantage = tf.reduce_mean(self.advantages)
    
    ##### Networks #####
      
    # Define the output shape of the policy networks
    if self.algorithm_params['std_dev'][0] == 'network':
      policy_output_shape = self.action_shape*2
    else:
      policy_output_shape = self.state_shape + self.action_shape**2 + self.state_shape*self.action_shape + self.state_shape**2

    # Current policy that will be updated and used to act, if the updated policy performs worse, the policy will be target_updated from the back up policy. 
    self.current_policy_network = self.policy_network_class.Network(self.sess, self.observations, policy_output_shape + self.state_shape, "current_policy_network", self.network_params['network_size'], dtype=self.dtype)
    self.current_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_policy_network')

    # Backup of the target policy network so far, backs up the target policy in case the update is bad. outputs the mean for the action
    self.target_policy_network = self.policy_network_class.Network(self.sess, self.observations, policy_output_shape + self.state_shape, "target_policy_network", self.network_params['network_size'], dtype=self.dtype)
    self.target_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_policy_network')

    # Policy network from last update, used for KL divergence calculation, outputs the mean for the action
    self.last_policy_network = self.policy_network_class.Network(self.sess, self.observations, policy_output_shape + self.state_shape, "last_policy_network", self.network_params['network_size'], dtype=self.dtype)
    self.last_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='last_policy_network')

    # Policy network with best average reward, used for KL divergence calculation, outputs the mean for the action
    self.best_policy_network = self.policy_network_class.Network(self.sess, self.observations, policy_output_shape + self.state_shape, "best_policy_network", self.network_params['network_size'], dtype=self.dtype)
    self.best_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='best_policy_network')


    ##### LQR ######

    self.current_policy_network_A, self.current_policy_network_B, self.current_policy_network_u = self.ctrlnet(self.current_policy_network.out[:,:policy_output_shape], self.current_policy_network.out[:,policy_output_shape:],policy_output_shape,self.state_shape,self.action_shape)
    self.target_policy_network_A, self.target_policy_network_B, self.target_policy_network_u = self.ctrlnet(self.target_policy_network.out[:,:policy_output_shape], self.target_policy_network.out[:,policy_output_shape:],policy_output_shape,self.state_shape,self.action_shape)
    self.last_policy_network_A, self.last_policy_network_B, self.last_policy_network_u = self.ctrlnet(self.last_policy_network.out[:,:policy_output_shape], self.last_policy_network.out[:,policy_output_shape:],policy_output_shape,self.state_shape,self.action_shape)
    self.best_policy_network_A, self.best_policy_network_B, self.best_policy_network_u = self.ctrlnet(self.best_policy_network.out[:,:policy_output_shape], self.best_policy_network.out[:,policy_output_shape:],policy_output_shape,self.state_shape,self.action_shape)


    ##### Policy Action Probability #####
      
    # Isolating the mean outputted by the policy network
    self.current_mean_policy = tf.reshape(tf.squeeze(self.current_policy_network_u),[-1, self.action_shape])
    self.target_mean_policy = tf.reshape(tf.squeeze(self.target_policy_network_u),[-1, self.action_shape])
    self.last_mean_policy = tf.reshape(tf.squeeze(self.last_policy_network_u),[-1, self.action_shape])
    self.best_mean_policy = tf.reshape(tf.squeeze(self.best_policy_network_u),[-1, self.action_shape])
    
    # Isolating the std dev outputted by the policy network, softplus is used to make sure that the std dev is positive
    if self.algorithm_params['std_dev'][0]=='network':
      self.current_std_dev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.current_policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
      self.target_std_dev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.target_policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
      self.last_std_dev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.last_policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
      self.best_std_dev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.best_policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
    else:
      self.current_std_dev_policy = self.std_dev_ph
      self.target_std_dev_policy = self.std_dev_ph
      self.last_std_dev_policy = self.std_dev_ph
      self.best_std_dev_policy = self.std_dev_ph
    
    # Gaussian distribution is built with mean and std dev from the policy network
    self.current_gaussian_policy_distribution = tf.contrib.distributions.Normal(self.current_mean_policy, self.current_std_dev_policy)
    self.target_gaussian_policy_distribution = tf.contrib.distributions.Normal(self.target_mean_policy, self.target_std_dev_policy)
    self.last_gaussian_policy_distribution = tf.contrib.distributions.Normal(self.last_mean_policy, self.last_std_dev_policy)
    self.best_gaussian_policy_distribution = tf.contrib.distributions.Normal(self.best_mean_policy, self.best_std_dev_policy)
    
    # Compute and log the KL divergence from last policy distribution to the current policy distribution
    self.kl = tf.reduce_mean(tf.contrib.distributions.kl_divergence(self.current_gaussian_policy_distribution, self.last_gaussian_policy_distribution))
    # tf.summary.scalar('kl', self.kl)
    
    # Action suggested by the current policy network
    number_of_suggestions = self.algorithm_params['number_of_suggestions']
    self.suggested_actions_out = tf.reshape(self.target_gaussian_policy_distribution._sample_n(number_of_suggestions),[-1,number_of_suggestions,self.action_shape])
    self.suggested_actions_out_best = tf.reshape(self.best_gaussian_policy_distribution._sample_n(number_of_suggestions),[-1,number_of_suggestions,self.action_shape])

    self.actions_out = tf.reshape(self.target_gaussian_policy_distribution._sample_n(1),[-1,self.action_shape])

    ##### Loss #####

    self.q_action_loss = tf.reduce_mean(tf.squared_difference(self.current_mean_policy, self.actions))
    self.q_action_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.train_q_action = self.q_action_optimizer.minimize(self.q_action_loss)
    
    # Compute and log loss for policy network
    self.negative_log_prob = -self.current_gaussian_policy_distribution.log_prob(self.actions)
    self.policy_network_losses = self.negative_log_prob*self.advantages # be careful with this operation, it should be element wise not matmul!
    self.policy_network_loss = tf.reduce_mean(self.policy_network_losses) #- tf.reduce_mean(self.current_gaussian_policy_distribution.entropy())
    self.summary = tf.summary.scalar('policy_network_loss', self.policy_network_loss)

    ##### Optimization #####
    
    # Optimizers for the network
    self.policy_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    
    self.grads, self.vars = zip(*self.policy_network_optimizer.compute_gradients(self.policy_network_loss))
    self.grads, _ = tf.clip_by_global_norm(self.grads, 1.0)
    self.train_policy_network = self.policy_network_optimizer.apply_gradients(zip(self.grads,self.vars))
    # Training operation for the network
    # self.train_policy_network = self.policy_network_optimizer.minimize(self.policy_network_loss)
    
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

    self.backup_op = []
    for var, var_target in zip(sorted(self.target_policy_network_vars,key=lambda v: v.name),sorted(self.best_policy_network_vars, key=lambda v: v.name)):
      self.backup_op.append(var_target.assign(var))
    self.backup_op = tf.group(*self.backup_op)
    
    self.restore_op = []
    for var, var_target in zip(sorted(self.best_policy_network_vars,key=lambda v: v.name),sorted(self.target_policy_network_vars, key=lambda v: v.name)):
      self.restore_op.append(var_target.assign(var))
    for var, var_target in zip(sorted(self.best_policy_network_vars,key=lambda v: v.name),sorted(self.current_policy_network_vars, key=lambda v: v.name)):
      self.restore_op.append(var_target.assign(var))
    self.restore_op = tf.group(*self.restore_op)

    ##### Logging #####

    # Log useful information
    # self.summary = tf.summary.merge_all()

    # Initialize all tf variables
    # self.sess.run(tf.global_variables_initializer())

  # CtrlNet
  def ctrlnet(self, sys_vec, state_vec, output_size, state_size, action_size):
    small_number = 0.0000001
    t = self.dtype
    dense = tf.layers.dense(inputs=sys_vec, units=output_size)
    dense = dense/tf.reduce_mean(dense)
    matrixTransformA = tf.matrix_band_part(tf.reshape(dense[:,:state_size*state_size],[-1,state_size,state_size]),0,-1)
    idx = state_size*state_size
    matrixDiagonalA = tf.matrix_diag(dense[:,idx:idx+state_size])
    idx += state_size
    matrixF = tf.reshape(dense[:,idx:idx+state_size*action_size],[-1,state_size,action_size])/(state_size*action_size)
    idx += state_size*action_size
    matrixGramR = tf.reshape(dense[:,idx:],[-1,action_size,action_size])/(state_size*action_size)
    matrixGramA = tf.matmul(tf.matmul(matrixTransformA,matrixDiagonalA),tf.matrix_inverse(matrixTransformA))/(state_size**2)
    matrixAU = tf.matrix_band_part(tf.matmul(matrixGramA,matrixGramA,transpose_a=True)/(state_size**2),0,-1)
    matrixB = tf.matmul(matrixTransformA,matrixF)
    matrixRU = tf.matrix_band_part(tf.matmul(matrixGramR,matrixGramR,transpose_a=True),0,-1)
    matrixA = 0.5*(matrixAU + tf.transpose(matrixAU,perm=[0,2,1]))
    matrixR = 0.5*(matrixRU + tf.transpose(matrixRU,perm=[0,2,1]))
    matrixQU = tf.matrix_band_part(tf.transpose(tf.matmul(tf.matmul(matrixB,tf.matrix_inverse(matrixR)),matrixB,transpose_b=True),perm=[0,2,1]),0,-1)
    matrixQ = 0.5*(matrixQU + tf.transpose(matrixQU,perm=[0,2,1]))
    matrixH11 = matrixA
    matrixH12 = -matrixQ
    matrixH21 = -matrixQ
    matrixH22 = -tf.transpose(matrixA,perm=[0,2,1])
    matrixH1 = tf.concat((matrixH11,matrixH12),axis=2)
    matrixH2 = tf.concat((matrixH21,matrixH22),axis=2)
    matrixH = tf.concat((matrixH1,matrixH2),axis=1)
    matrixH = tf.where(tf.logical_and((matrixH >= 0),(matrixH<small_number)), small_number*tf.ones(tf.shape(matrixH),t), matrixH)
    matrixH = tf.where(tf.logical_and((matrixH < 0),(tf.abs(matrixH)<small_number)), -small_number*tf.ones(tf.shape(matrixH),t), matrixH)
    matrixHU = tf.matrix_band_part(matrixH,0,-1)
    matrixH = 0.5*(matrixHU + tf.transpose(matrixHU,perm=[0,2,1]))/tf.reduce_mean(matrixH)
    eigVals, eigVecs = tf.self_adjoint_eig(matrixH)
    eigVecs = tf.transpose(eigVecs,perm=[0,2,1])
    stableEigIdxs = tf.where(eigVals<0)
    matrixV = tf.reshape(tf.gather_nd(eigVecs,stableEigIdxs),[-1,state_size,2*state_size])
    matrixV1 = tf.transpose(matrixV[:,:,:state_size],perm=[0,2,1])
    matrixV2 = tf.transpose(matrixV[:,:,state_size:],perm=[0,2,1])
    matrixP = tf.matmul(matrixV2,tf.matrix_inverse(matrixV1))
    matrixK = tf.matmul(tf.matmul(tf.matrix_inverse(matrixR),matrixB,transpose_b=True),matrixP)
    state_vec = tf.expand_dims(state_vec,axis=2)
    u = tf.matmul(matrixK,state_vec)

    return [matrixA, matrixB, u]

  # Compute suggested actions from observation
  def compute_suggested_actions(self, observation):
    suggested_actions = self.sess.run([self.suggested_actions_out], {self.observations: observation[None], self.std_dev_ph:self.std_dev})
    actions = []
    for i in range(self.algorithm_params['number_of_suggestions']):
      actions.append(suggested_actions[0][:,i,:])
    return actions

  # Compute best suggested actions from observation
  def compute_suggested_actions_best(self, observation):
    suggested_actions = self.sess.run([self.suggested_actions_out_best], {self.observations: observation[None], self.std_dev_ph:self.std_dev})
    actions = []
    for i in range(self.algorithm_params['number_of_suggestions']):
      actions.append(suggested_actions[0][:,i,:])
    return actions

  # Compute suggested actions from observation
  def compute_action(self, observation):
    action = self.sess.run([self.actions_out], {self.observations: observation[None], self.std_dev_ph:self.std_dev})[0]
    return action
  
  # update the target network
  def soft_target_update(self):
    _ = self.sess.run([self.target_update],{})

  # update the last policy network
  def update_last_policy(self):
    _ = self.sess.run([self.update_last],{})

  # Restore the last best network
  def restore(self):
    _ = self.sess.run([self.restore_op],{})

  # Backup the target network
  def backup(self):
    _ = self.sess.run([self.backup_op],{})

  # Train the Q network from the given batches
  def train(self, observations_batch, advantages_batch, actions_batch, learning_rate):
    self.algorithm_params['learning_rate'] = learning_rate
    summaries = []
    stats = {}
    for i in range(1):
      # Taking the gradient step to optimize (train) the policy network.
      policy_network_losses, policy_network_loss, _ = self.sess.run([self.policy_network_losses, self.policy_network_loss, self.train_policy_network], {self.observations:observations_batch, self.actions:actions_batch, self.advantages:advantages_batch, self.learning_rate:self.algorithm_params['learning_rate'], self.std_dev_ph:self.std_dev})

    # Get stats
    summary, average_advantage, kl  = self.sess.run([self.summary, self.average_advantage, self.kl], {self.observations:observations_batch, self.actions:actions_batch, self.advantages:advantages_batch, self.std_dev_ph:self.std_dev})

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

  # Train the Q network from the given batches
  def train_q(self, batch_size, observations_batch, actions_batch, learning_rate):
    self.algorithm_params['learning_rate'] = learning_rate
    observations_mini_batch = observations_batch
    actions_mini_batch = actions_batch
    # loss, _ = self.sess.run([self.q_action_loss, self.train_q_action], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.learning_rate:self.algorithm_params['learning_rate']})
    # for i in range(500):
    #   mini_batch_idx = np.random.choice(batch_size, 32)
    #   observations_mini_batch = observations_batch[mini_batch_idx,:]
    #   actions_mini_batch = actions_batch[mini_batch_idx,:]
    #   loss, _ = self.sess.run([self.q_action_loss, self.train_q_action], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.learning_rate:self.algorithm_params['learning_rate']})
    # # print(loss)

    for i in range(50):
      mini_batch_idx = np.random.choice(batch_size, 128)
      observations_mini_batch = observations_batch[mini_batch_idx,:]
      actions_mini_batch = actions_batch[mini_batch_idx,:]
      loss, _ = self.sess.run([self.q_action_loss, self.train_q_action], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.learning_rate:self.algorithm_params['learning_rate']})
    # print(loss)

    # for i in range(1000):
    #   mini_batch_idx = np.random.choice(batch_size, 1000)
    #   observations_mini_batch = observations_batch[mini_batch_idx,:]
    #   actions_mini_batch = actions_batch[mini_batch_idx,:]
    #   loss, _ = self.sess.run([self.q_action_loss, self.train_q_action], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.learning_rate:self.algorithm_params['learning_rate']})

    # print(loss)

    # for i in range(50):
    #   self.algorithm_params['learning_rate'] = learning_rate
    #   observations_mini_batch = observations_batch
    #   actions_mini_batch = actions_batch
    #   loss, _ = self.sess.run([self.q_action_loss, self.train_q_action], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.learning_rate:self.algorithm_params['learning_rate']})
    # Backup the current policy network to last policy network
    # print(loss)

    self.update_last_policy()
    
    self.soft_target_update()
 
  def update_std_dev(self):
    self.std_dev -= 0.000
    if self.std_dev < 0.3:
      self.std_dev = 0.3
