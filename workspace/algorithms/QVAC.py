import numpy as np 
import tensorflow as tf 
import importlib
from algorithms.utils.utils import *
from algorithms.utils.PolicyNetwork import PolicyNetwork
from algorithms.utils.ValueNetwork import ValueNetwork
from algorithms.utils.QNetworkReturns import QNetwork


import itertools

class Agent:
  """
  Advantage Actor Suggestor Algorithm

  Note:
    Implementation of Adavantage Actor Suggestor algorithm as proposed in ###
  
  Args: 
    env: OpenAI environment object
    
    sess: TensorFlow Session object
    
    data_collection_params (dict): Parameters for data collection / interacting with the environment
      'min_batch_size' (int): Minimum batch size for interacting with the environment
      'min_episodes' (int): Minimum episodes to interact with the environment per batch 
      'episode_adapt_rate' (int): amount to increase or decrease for min_epidoes
    
    training_params (dict): Parameters for training
      'total_timesteps' (int): Total time steps to train for 
      'adaptive_lr' (bool): Use adaptive learning rate based on the desired kl divergence between current and last policy 
      'desired_kl' (double): Desired kl divergence for adaptive learning rate
    
    network_params (dict): Parameters to define the network used
      'value_network' (list): Defines the value network e.g. ['fully_connected_network','small/medium/large'] 
      'policy_network' (list): Defines the policy network e.g. ['fully_connected_network','small/medium/large'] 
    
    algorithm_params (dict): Parameters specific to the algorithm 
      'gamma' (double): discount rate 
      'learning_rate' (double): Learning rate for gradient updates 
      'std_dev' (list): Different ways to define the standard devation of the policy e.g. ['fixed'/'linear'/'network', (double)/(double)/-] NOTE: network option has the policy network to output the std dev 
      'target_update_rate' (double): Rate to perform the soft updates for the networks
               
    logs_path (string): Path to save training logs

  """
  def __init__(self,
               env,
               sess,
               data_collection_params = {'min_batch_size':1000,
                                         'min_episodes':1, 
                                         'episode_adapt_rate':3},
               training_params = {'total_timesteps':1000000,  
                                  'adaptive_lr':True, 
                                  'desired_kl':2e-3},
               network_params = {'q_network':['fully_connected_network','large'],
                                 'value_network':['fully_connected_network','large'], 
                                 'policy_network':['fully_connected_network','large']},
               algorithm_params = {'gamma':0.99, 
                                   'learning_rate':1e-3,
                                   'std_dev':['fixed', 0.2], 
                                   'target_update_rate':0.001},
               logs_path="/home/user/workspace/logs/"):

    # Tensorflow Session
    self.sess = sess
    
    # OpenAI Environment
    self.env = env

    # Getting the shape of observation space and action space of the environment
    self.observation_shape = self.env.observation_space.shape[0]
    self.action_shape = self.env.action_space.shape[0]
    
    # Hyper Parameters
    self.data_collection_params = data_collection_params
    self.training_params = training_params
    self.network_params = network_params
    self.algorithm_params = algorithm_params

    # Hyper Paramters for Networks
    q_network_params = {'network_type':self.network_params['q_network'][0], 'network_size':self.network_params['q_network'][1]}
    value_network_params = {'network_type':self.network_params['value_network'][0], 'network_size':self.network_params['value_network'][1]}
    policy_network_params = {'network_type':self.network_params['policy_network'][0], 'network_size':self.network_params['policy_network'][1]}

    # Path to save training logs
    self.logs_path = logs_path
    
    ##### Networks #####

    self.q_network = QNetwork(self.env, self.sess, q_network_params, self.algorithm_params)
    self.value_network = ValueNetwork(self.env, self.sess, value_network_params, self.algorithm_params)
    self.policy_network = PolicyNetwork(self.env, self.sess, policy_network_params, self.algorithm_params)

    ##### Logging #####

    # Placeholder for average reward for logging
    self.average_reward = tf.placeholder(tf.float32, name="average_reward")
    
    # Log useful information
    self.reward_summary = tf.summary.scalar("average_reward", self.average_reward)

    # Setup the tf summary writer and initialize all tf variables
    self.writer = tf.summary.FileWriter(logs_path, sess.graph)
    self.sess.run(tf.global_variables_initializer())

  # Collecting experience (data) and training the agent (networks)
  def train(self, saver=None, save_dir=None):
    
    # Keeping count of total timesteps and episodes of environment experience for stats
    total_timesteps = 0
    total_episodes = 0
    
    # KL divergence, used to adjust the learning rate
    kl = 0
    
    # Keeping track of the best averge reward
    best_average_reward = -np.inf

    ##### Training #####
    
    # Training iterations
    while total_timesteps < self.training_params['total_timesteps']:

      # Collect batch of data
      trajectories, returns, undiscounted_returns, advantages, batch_size, episodes = self.collect_trajs(total_timesteps)
      observations_batch, actions_batch, rewards_batch, returns_batch, next_observations_batch, advantages_batch = self.traj_to_batch(trajectories, returns, advantages) 

      # update total timesteps and total episodes
      total_timesteps += batch_size
      total_episodes += episodes
      
      # Learning rate adaptation
      learning_rate = self.update_lr(kl)

      # Average undiscounted return for the last data collection
      average_reward = np.mean(undiscounted_returns)
      
      # Save the best model
      if average_reward > best_average_reward:
        # Save the model
        best_average_reward = average_reward     
        saver.save(self.sess, save_dir)
      
      ##### Optimization #####

      q_summaries, q_stats =self.train_q_network(observations_batch, returns_batch, actions_batch, learning_rate)
      q_network_loss = q_stats['q_network_loss']
      self.add_summaries(q_summaries, total_timesteps)

      value_summaries, value_stats =self.train_value_network(observations_batch, returns_batch, learning_rate)
      value_network_loss = value_stats['value_network_loss']
      self.add_summaries(value_summaries, total_timesteps)

      policy_summaries, policy_stats =self.train_policy_network(observations_batch, actions_batch, advantages_batch, learning_rate)
      policy_network_loss = policy_stats['policy_network_loss']
      kl = policy_stats['kl']
      average_advantage = policy_stats['average_advantage']
      
      self.print_stats(total_timesteps, total_episodes, best_average_reward, average_reward, kl, policy_network_loss, value_network_loss, q_network_loss, average_advantage, learning_rate, batch_size)

    self.writer.close()

  ##### Helper Functions #####
  
  # Collect trajectores
  def collect_trajs(self, total_timesteps):
    # Batch size and episodes experienced in current iteration
    batch_size = 0
    episodes = 0

    # Lists to collect data
    trajectories, returns, undiscounted_returns, advantages = [], [], [], []

    ##### Collect Batch #####

    # Collecting minium batch size or minimum episodes of experience
    while episodes < self.data_collection_params['min_episodes'] or batch_size < self.data_collection_params['min_batch_size']:
                      
      ##### Episode #####
      
      # Run one episode
      observations, actions, rewards, dones = self.run_one_episode()

      ##### Data Appending #####
      
      # Get sum of reward for this episode
      undiscounted_returns.append(np.sum(rewards))

      # Update the counters
      batch_size += len(rewards)
      total_timesteps += len(rewards)
      episodes += 1

      self.log_rewards(np.sum(rewards), total_timesteps)

      # Episode trajectory
      trajectory = {"observations":np.array(observations), "actions":np.array(actions), "rewards":np.array(rewards), "dones":np.array(dones)}
      trajectories.append(trajectory)
      
      # Computing the discounted return for this episode (NOT A SINGLE NUMBER, FOR EACH OBSERVATION)
      return_ = discount(trajectory["rewards"], self.algorithm_params['gamma'])

      # Compute the value estimates for the observations seen during this episode
      values = self.value_network.compute_value(observations)

      # Compute the q value estimates for the observations seen during this episode
      observations_batch = np.concatenate(observations).reshape([-1,self.observation_shape])
      actions_batch = np.concatenate([actions]).reshape([-1,self.action_shape])
      q_values = self.q_network.compute_q(observations_batch, actions_batch)
      
      # Computing the advantage estimate
      advantage = np.concatenate(q_values[0]) - np.concatenate(values[0])
      returns.append(return_)
      advantages.append(advantage)

    return [trajectories, returns, undiscounted_returns, advantages, batch_size, episodes]

  # Run one episode
  def run_one_episode(self):

    # Restart env
    observation = self.env.reset()
    
    # Flag that env is in terminal state
    done = False

    observations, actions, rewards, dones = [], [], [], []
    while not done:
      # Collect the observation
      observations.append(observation)

      # Sample action with current policy
      action = self.compute_action(observation)

      # For single dimension actions, wrap it in np array
      if not isinstance(action, (list, tuple, np.ndarray)):
        action = np.array([action])
      action = np.concatenate(action)

      # Take action in environment
      observation, reward, done, _ = self.env.step(action)

      # Collect reward and action
      rewards.append(reward)
      actions.append(action)
      dones.append(done)

    return [observations, actions, rewards, dones]

  # Compute action using policy network
  def compute_action(self, observation):
    action = self.policy_network.compute_action(observation)
    return action
  
  # Log rewards
  def log_rewards(self, rewards, timestep):
    reward_summary = self.sess.run([self.reward_summary], {self.average_reward:np.sum(rewards)})[0]
    self.writer.add_summary(reward_summary, timestep)
  
  # Convert trajectories to batches
  def traj_to_batch(self, trajectories, returns, advantages):
    ##### Data Prep #####
      
    # Observations for this batch
    observations_batch = np.concatenate([trajectory["observations"] for trajectory in trajectories])
    next_observations_batch = np.roll(observations_batch, 1, axis=0)
    next_observations_batch[0,:] = observations_batch[0,:]
    
    # Actions for this batch, reshapeing to handel 1D action space
    actions_batch = np.concatenate([trajectory["actions"] for trajectory in trajectories]).reshape([-1,self.action_shape])
    
    # Rewards of the trajectory as a batch
    rewards_batch = np.concatenate([trajectory["rewards"] for trajectory in trajectories]).reshape([-1,1])

    # Binary dones from environment in a batch
    dones_batch = np.concatenate([trajectory["dones"] for trajectory in trajectories])
    
    # Discounted returns for this batch. itertool used to make batch into long np array
    returns_batch = np.array(list(itertools.chain.from_iterable(returns))).reshape([-1,1])

    # Advantages for this batch. itertool used to make batch into long np array
    advantages_batch = np.array(list(itertools.chain.from_iterable(advantages))).flatten().reshape([-1,1])

    return [observations_batch, actions_batch, rewards_batch, returns_batch, next_observations_batch, advantages_batch]
  
  # Update learning rate
  def update_lr(self, kl):
    if self.training_params['adaptive_lr']:
      if kl > self.training_params['desired_kl'] * 2: 
        self.algorithm_params['learning_rate'] /= 1.5
      elif kl < self.training_params['desired_kl'] / 2: 
        self.algorithm_params['learning_rate'] *= 1.5
      learning_rate = self.algorithm_params['learning_rate']
    else:
      learning_rate = self.algorithm_params['learning_rate']
    return learning_rate
  
  # Add summaries to the writer
  def add_summaries(self, summaries, timestep):
    for summary in summaries:
      # Write summary for tensorboard visualization
      self.writer.add_summary(summary, timestep)

  # Train vq network
  def train_q_network(self, observations_batch, returns_batch, actions_batch, learning_rate):
    summaries, stats = self.q_network.train(observations_batch, returns_batch, actions_batch, learning_rate)
    return [summaries, stats]

  # Train value network
  def train_value_network(self, observations_batch, returns_batch, learning_rate):
    summaries, stats = self.value_network.train(observations_batch, returns_batch, learning_rate)
    return [summaries, stats]

  # Train policy network
  def train_policy_network(self, observations_batch, actions_batch, advantages_batch, learning_rate):
    summaries, stats = self.policy_network.train(observations_batch, advantages_batch, actions_batch, learning_rate)
    return [summaries, stats]
  
  # Print stats
  def print_stats(self, total_timesteps, total_episodes, best_average_reward, average_reward, kl, policy_network_loss, value_network_loss, q_network_loss, average_advantage, learning_rate, batch_size):
    ##### Reporting Performance #####
      
    # Printing performance progress and other useful infromation
    print("_______________________________________________________________________________________________________________________________________________________________________________________________________________")
    print("{:>15} {:>15} {:>15} {:>15} {:>20} {:>20} {:>20} {:>20} {:>20} {:>10} {:>15}".format("total_timesteps", "episodes", "best_reward", "reward", "kl_divergence", "policy_loss", "value_loss", "q_network_loss", "average_advantage", "lr", "batch_size"))
    print("{:>15} {:>15} {:>15.2f} {:>15.2f} {:>20.5E} {:>20.2f} {:>20.2f} {:>20.2f} {:>20.2f} {:>10.2E} {:>15}".format(total_timesteps, total_episodes, best_average_reward, average_reward, kl, policy_network_loss, value_network_loss, q_network_loss, average_advantage, learning_rate, batch_size))



