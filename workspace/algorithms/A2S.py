import numpy as np 
import tensorflow as tf 
import importlib
from algorithms.utils import *
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
      'learning_rate' (double): Learning rate for gradient updates 
      'adaptive_lr' (bool): Use adaptive learning rate based on the desired kl divergence between current and last policy 
      'desired_kl' (double): Desired kl divergence for adaptive learning rate
    
    rl_params (dict): Parameters for general reinforcement learning
      'gamma' (double): discount rate 
      'num_policy_update' (int): number of policy update from the current batch collected from the environment
    
    network_params (dict): Parameters to define the network used
      'q_network' (list): Defines the Q network e.g. ['fully_connected_network','small/medium/large'] 
      'value_network' (list): Defines the value network e.g. ['fully_connected_network','small/medium/large'] 
      'policy_network' (list): Defines the policy network e.g. ['fully_connected_network','small/medium/large'] 
    
    algorithm_params (dict): Parameters specific to the algorithm 
      'number_of_suggestions' (int): Number of suggestion given by the policy network to the Q network 
      'restore' (bool): Restore the best networks backed up when the agent performs bad with some probabilty
      'DDQN' (bool): Use double Q-learning 
      'std_dev' (list): Different ways to define the standard devation of the policy e.g. ['fixed'/'linear'/'network', (double)/(double)/-] NOTE: network option has the policy network to output the stddev 
      'experience_replay' (None/string): Which experience replay to use e.g. None/'ER'/'PER' NOTE: no experience replay, experience replay, priotized experience replay
      'experience_replay_size' (int): Experience replay buffer size 
      'ER_batch_size' (int): Experience replay buffer size 
      'PER_alpha' (double): Proportional prioritization constant 
      'PER_epsilon' (double): Small positive constant that ensures that no transition has zero priority
               
    logs_path (string): Path to save training logs

  """
  def __init__(self,
               env,
               sess,
               data_collection_params = {'min_batch_size':1000,
                                         'min_episodes':1, 
                                         'episode_adapt_rate':3},
               training_params = {'total_timesteps':1000000, 
                                  'learning_rate':1e-3, 
                                  'adaptive_lr':True, 
                                  'desired_kl':2e-3},
               rl_params = {'gamma':0.99, 
                            'num_policy_update':1},
               network_params = {'q_network':['fully_connected_network','large'], 
                                 'value_network':['fully_connected_network','large'], 
                                 'policy_network':['fully_connected_network','large']},
               algorithm_params = {'number_of_suggestions':10, 
                                  'restore': True,
                                  'DDQN':True, 
                                  'std_dev':['fixed', 0.2], 
                                  'experience_replay':'PER', 
                                  'experience_replay_size':200000, 
                                  'ER_batch_size':32, 
                                  'PER_alpha':0.6, 
                                  'PER_epsilon':0.01},
               logs_path="/home/user/workspace/logs/"):

  PE 
  DDQN
  output variance/ adaptive variance
  epison greedy
  check std_dev
  
    # Tensorflow Session
    self.sess = sess
    
    # OpenAI Environment
    self.env = env
    
    # Hyper Parameters
    self.data_collection_params = data_collection_params
    self.training_params = training_params
    self.rl_params = rl_params
    self.network_params = network_params
    self.algorithm_params = algorithm_params
    
    # Path to save training logs
    self.logs_path = logs_path
    
    # Importing the desired network architecture
    self.q_network_class = importlib.import_module("architectures." + self.network_params['q_network'][0])
    self.value_network_class = importlib.import_module("architectures." + self.network_params['value_network'][0])
    self.policy_network_class = importlib.import_module("architectures." + self.network_params['policy_network'][0])

    # Getting the shape of observation space and action space of the environment
    self.observation_shape = self.env.observation_space.shape[0]
    self.action_shape = self.env.action_space.shape[0]

    # Experience replay buffer
    if self.algorithm_params['experience_replay']=="PER":
      self.replay_buffer = PrioritizedExperienceReplay(self.algorithm_params['experience_replay_size'], self.algorithm_params['PER_epsilon'], self.algorithm_params['PER_alpha'])
    elif self.algorithm_params['experience_replay']=="ER":
      self.replay_buffer = ReplayBuffer(self.algorithm_params['experience_replay_size'])
    else:
      self.replay_buffer = None
    
    # Scoping variables with A2S
    with tf.variable_scope("A2S"):

      ##### Placeholders #####
      
      # Placeholder for observations
      self.observations = tf.placeholder(tf.float32, shape=[None, self.observation_shape], name="observations")
      
      # Placeholder for actions taken, this is used for the policy gradient
      self.actions = tf.placeholder(tf.float32, shape=[None, self.action_shape], name="actions")
      
      # Placeholder for the advantage function
      self.advantages = tf.placeholder(tf.float32, shape=[None, 1], name="advantages")
      
      # Placeholder for learning rate for the optimizer
      self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

      # Placeholder for std dev
      self.std_dev = tf.placeholder(tf.float32, name="std_dev")
      
      # Mean and the stddev of the last policy distrubution for KL calculation
      self.last_mean_policy_ph = tf.placeholder(tf.float32, name="last_mean_policy")
      self.last_std_dev_policy_ph = tf.placeholder(tf.float32, name="last_std_dev_policy")
      
      # Placeholder for the returns e.g. r + gamma*next_value
      self.returns = tf.placeholder(tf.float32, shape=[None, 1], name="returns")
      
      # Placeholder for average reward for logging
      self.average_reward = tf.placeholder(tf.float32, name="average_reward")
      
      ##### Networks #####
      
      # Define the output shape of the policy networks
      if self.algorithm_params['std_dev'][0] == 'network':
        policy_output_shape = self.action_shape*2
      else
        policy_output_shape = self.action_shape

      # Current policy that will be updated and used to act, if the updated policy performs worse, the policy will be restored from the back up policy. 
      self.current_policy_network = self.policy_network_class.Network(sess, self.observations, policy_output_shape, "current_policy_network", self.network_params['policy_network'][1])
      self.current_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A2S/current_policy_network')

      # Backup of the best policy network so far, backs up the best policy in case the update is bad. outputs the mean for the action
      self.best_policy_network = self.policy_network_class.Network(sess, self.observations, policy_output_shape, "backup_policy_network", self.network_params['policy_network'][1])
      self.best_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A2S/backup_policy_network')

      # Policy network from last update, used for KL divergence calculation, outputs the mean for the action
      self.last_policy_network = self.policy_network_class.Network(sess, self.observations, policy_output_shape, "last_policy_network", self.network_params['policy_network'][1])
      self.last_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A2S/last_policy_network')
      
      # Current value function used for advantage estimate
      self.current_value_network = self.value_network_class.Network(sess, self.observations, 1, "current_value_network", self.network_params['value_network'][1])
      self.current_value_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A2S/current_value_network')

      # Backup of the best value network so far, backs up the best value network in case the update is bad
      self.best_value_network = self.value_network_class.Network(sess, self.observations, 1, "best_value_network", self.network_params['value_network'][1])
      self.best_value_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A2S/best_value_network')
      
      ##### Policy Action Probability #####
      
      # Isolating the mean outputted by the policy network
      self.current_mean_policy = tf.reshape(tf.squeeze(self.current_policy_network.out[:,:self.action_shape]),[-1, self.action_shape])
      self.best_mean_policy = tf.reshape(tf.squeeze(self.best_policy_network.out[:,:self.action_shape]),[-1, self.action_shape])
      self.last_mean_policy = tf.reshape(tf.squeeze(self.last_policy_network.out[:,:self.action_shape]),[-1, self.action_shape])
      
      # Isolating the stddev outputted by the policy network, softplus is used to make sure that the stddev is positive
      if self.algorithm_params['std_dev'][0]=='network':
        self.current_std_dev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.current_policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
        self.best_std_dev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.best_policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
        self.last_std_dev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.last_policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
      elif self.algorithm_params['std_dev'][0]=='linear':
      
      # Gaussian distribution is built with mean and stddev from the policy network
      self.current_gaussian_policy_distribution = tf.contrib.distributions.Normal(self.current_mean_policy, self.current_std_dev_policy)
      self.best_gaussian_policy_distribution = tf.contrib.distributions.Normal(self.best_mean_policy, self.best_std_dev_policy)
      self.last_gaussian_policy_distribution = tf.contrib.distributions.Normal(self.last_mean_policy, self.last_std_dev_policy)
      
      # Compute and log the KL divergence from last policy distribution to the current policy distribution
      self.kl = tf.reduce_mean(tf.contrib.distributions.kl_divergence(self.current_gaussian_policy_distribution, self.last_gaussian_policy_distribution))
      tf.summary.scalar('kl', self.kl)
      
      # Action suggested by the current policy network
      number_of_suggestions = self.algorithm_params['number_of_suggestions']
      self.suggested_actions_out = tf.reshape(self.current_gaussian_policy_distribution._sample_n(number_of_suggestions),[-1,number_of_suggestions,self.action_shape])
      self.state_action_pairs = tf.concat([self.observations, self.actions],1)  

      ##### Q Network #####
      
      # Current Q network, outputs Q value of observation, used for advantage estimate
      self.current_q_network = self.q_network_class.Network(self.sess, self.state_actions, 1, "current_q_network", self.network_params['q_network'][1])
      self.current_q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A2S/current_q_network')

      # Best Q network, outputs Q value of observation, used for advantage estimate
      self.best_q_network = self.q_network_class.Network(self.sess, self.state_actions, 1, "best_q_network", self.network_params['q_network'][1])
      self.best_q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A2S/best_q_network')
      
      ##### Loss #####
      
      # Compute and log loss for policy network
      self.negative_log_prob = -self.current_gaussian_policy_distribution.log_prob(self.actions)
      self.policy_network_losses = self.negative_log_prob*self.advantages # be careful with this operation, it should be element wise not matmul!
      self.policy_network_loss = tf.reduce_mean(self.policy_network_losses)
      tf.summary.scalar('policy_network_loss', self.policy_network_loss)

      # Compute and log loss for value network
      self.value_network_y = self.returns
      self.value_network_loss = tf.reduce_mean(tf.squared_difference(self.current_value_network.out, self.value_network_y))
      tf.summary.scalar('value_network_loss', self.value_network_loss)
      
      # Compute and log loss for Q network
      self.q_network_y = self.returns
      self.q_network_loss = tf.reduce_mean(tf.squared_difference(self.current_q_network.out,self.q_network_y))
      tf.summary.scalar('q_network_loss', self.q_network_loss)

      ##### Optimization #####
      
      # Optimizers for the networks
      self.policy_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      self.value_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      self.q_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      
      # Training operation for the pnetworks
      self.train_policy_network = self.policy_network_optimizer.minimize(self.policy_network_loss)
      self.train_value_network = self.value_network_optimizer.minimize(self.value_network_loss)
      self.train_q_network = self.q_network_optimizer.minimize(self.q_network_loss)
      
      ##### Copy Network Operations #####
      
      # Restore operations for the policy, value and Q networks for when the agent performs very bad and need to recover
      self.restore = []
      for var, var_target in zip(sorted(self.best_policy_network_vars,key=lambda v: v.name),sorted(self.current_policy_network_vars, key=lambda v: v.name)):
        self.restore.append(var_target.assign(var))
      for var, var_target in zip(sorted(self.best_value_network_vars,key=lambda v: v.name),sorted(self.current_value_network_vars, key=lambda v: v.name)):
       self.restore.append(var_target.assign(var))
      for var, var_target in zip(sorted(self.best_q_network_vars,key=lambda v: v.name),sorted(self.current_q_network_vars, key=lambda v: v.name)):
       self.restore.append(var_target.assign(var))
      self.restore = tf.group(*self.restore)

      # Backup operations for the policy, value and Q networks for when the agent performs very well and want to backup the best network
      self.backup = []
      for var, var_target in zip(sorted(self.current_policy_network_vars,key=lambda v: v.name),sorted(self.best_policy_network_vars, key=lambda v: v.name)):
        self.backup.append(var_target.assign(var))
      for var, var_target in zip(sorted(self.current_value_network_vars,key=lambda v: v.name),sorted(self.best_value_network_vars, key=lambda v: v.name)):
       self.backup.append(var_target.assign(var))
      for var, var_target in zip(sorted(self.current_q_network_vars,key=lambda v: v.name),sorted(self.best_q_network_vars, key=lambda v: v.name)):
       self.backup.append(var_target.assign(var))
      self.backup = tf.group(*self.backup)

      # Copy over the current policy network to last policy network for KL divergence Calculation
      self.last = []
      for var, var_target in zip(sorted(self.current_policy_network_vars,key=lambda v: v.name),sorted(self.last_policy_network_vars, key=lambda v: v.name)):
        self.last.append(var_target.assign(var))
      self.last = tf.group(*self.last)

      ##### Logging #####

      # Log useful information
      self.summary = tf.summary.merge_all()
      self.reward_summary = tf.summary.scalar("average_reward", self.average_reward)
      self.average_advantage = tf.reduce_mean(self.advantages)
      tf.summary.scalar('average_advantage', self.average_advantage)

    # Setup the tf summary writer and initialize all tf variables
    self.writer = tf.summary.FileWriter(logs_path, sess.graph)
    self.sess.run(tf.global_variables_initializer())

  # Samples action from the current policy network and selects the action with highest Q value
  def compute_action(self, observation):
    suggested_actions = self.sess.run([self.suggested_actions_out], {self.observations: observation[None]})
    best_action = None
    best_q = -np.inf
    for i in range(self.number_of_suggestions):
      current_q = self.compute_q(observation, suggested_actions[0][:,i,:])
      if current_q > best_q:
        best_q = current_q
        best_action = suggested_actions[0][:,i,:]
    return best_action

  # Compute Q value of observation and action
  def compute_q(self, observation, action):
    if self.algorithm_params['DDQN']:
      q_value = self.sess.run([self.best_q_network.out],{self.observations: observation[None], self.actions:action})[0]
    else:
      q_value = self.sess.run([self.current_q_network.out],{self.observations: observation[None], self.actions:action})[0]

    return q_value

  # Computes the value for given observations
  def compute_value(self, observations):
    return self.sess.run(self.best_value_network.out, {self.observations: observations})
  
  # Restore the best networks
  def restore(self):
    _ = self.sess.run([self.restore],{})
  
  # Backup the current networks
  def backup(self):
    _ = self.sess.run([self.backup],{})
  
  # Backup the current policy to last policy network for KL divergence calculation
  def backup_current_policy_to_last_policy(self):
    _ = self.sess.run([self.last],{})

  # Collecting experience (data) and training the agent (networks)
  def train(self, saver=None, save_dir=None):
    #
    # keeping count of total timesteps of environment experience
    total_timesteps = 0
    total_episodes = 0
    last_timestep = 0
    #
    # sum of rewards of 100 episodes, not for learning, just a performance metric and logging
    #
    # kl divergence, used to adjust the learning rate
    kl = 0
    #
    # keeping track of the best averge reward
    best_average_reward = -np.inf
    iteration = 0
    #
    ##### Training #####
    #
    # training iterations
    while total_timesteps < self.iterations:
      iteration += 1
      #
      # batch size changes from episode to episode
      batch_size = 0
      episodes = 0
      trajectories, returns, advantages_ = [], [], []
      _rewards = []
      #
      ##### Collect Batch #####
      #
      # collecting minium batch size of experience
      # while batch_size < self.min_batch_size:
      while episodes < self.min_episodes or batch_size < self.min_batch_size:
        #
        # restart env
        observation = self.env.reset()
        #
        # data for this episode
        observations, actions, rewards, dones = [], [], [], []
        #
        # flag that env is in terminal state
        done = False

        #
        ##### Episode #####
        #
        while not done:
          #
          # collect the observation
          observations.append(observation)
          #
          # sample action with current policy
          action = self.compute_action(observation)
          #
          # for single dimension actions, wrap it in np array
          if not isinstance(action, (list, tuple, np.ndarray)):
            action = np.array([action])
          action = np.concatenate(action)
          #
          # take action in environment
          observation, reward, done, _ = self.env.step(action)
          #
          # collect reward and action
          rewards.append(reward)
          actions.append(action)
          dones.append(done)
        #
        ##### Data Appending #####
        #
        # keeping trajectory_rewards as fifo of 100
        #
        # get sum of reward for this episode
        _rewards.append(np.sum(rewards))
        #
        # add timesteps of this episode to batch_size
        batch_size += len(rewards)
        total_timesteps += len(rewards)
        reward_summary = self.sess.run([self.reward_summary], {self.average_reward:np.sum(rewards)})[0]
        self.writer.add_summary(reward_summary, total_timesteps)
        episodes += np.sum(dones)
        #
        # episode trajectory
        trajectory = {"observations":np.array(observations), "actions":np.array(actions), "rewards":np.array(rewards), "dones":np.array(dones)}
        trajectories.append(trajectory)
        #
        # computing the discounted return for this episode (NOT A SINGLE NUMBER, FOR EACH OBSERVATION)
        return_ = discount(trajectory["rewards"], self.gamma)
        returns.append(return_)
        values = self.compute_value(observations)
        #
        # computing the advantage estimate
        advantage_ = return_ - np.concatenate(values)
        advantages_.append(advantage_)
      #
      ##### Data Prep #####
      #
      # total timesteps is sum of all batch size
      #total_timesteps += batch_size
      #
      # observations for this batch
      observations_batch = np.concatenate([trajectory["observations"] for trajectory in trajectories])
      #
      # actions for this batch, reshapeing to handel 1D action space
      actions_batch = np.concatenate([trajectory["actions"] for trajectory in trajectories]).reshape([-1,self.action_shape])
      #
      # discounted returns for this batch. itertool used to make batch into long np array
      rewards_batch = np.concatenate([trajectory["rewards"] for trajectory in trajectories]).reshape([-1,1])
      #
      # calc number of episodes
      dones_batch = np.concatenate([trajectory["dones"] for trajectory in trajectories])
      total_episodes += np.sum(dones_batch)
      #
      # discounted returns for this batch. itertool used to make batch into long np array
      returns_batch = np.array(list(itertools.chain.from_iterable(returns))).reshape([-1,1])
      #
      # advantages for this batch. itertool used to make batch into long np array
      advantages_batch_ = np.array(list(itertools.chain.from_iterable(advantages_))).flatten().reshape([-1,1])
      #
      # learning rate
      if kl > 6e-3 * 2: 
        self.lr /= 1.5
      elif kl < 6e-3 / 2: 
        self.lr *= 1.5
      learning_rate = self.lr

      # 
      ##### Optimization #####
      #
      #
      # if total_timesteps - last_timestep > 50000:
      #   self.mini_iterations *= 2
      #   self.episodes *= 2
      #   last_timestep = total_timesteps
      #   self.lr *= 2
      #
      # average reward of past 100 episodes


      average_reward = np.mean(_rewards) 
      if average_reward < best_average_reward and 1-(abs(average_reward- best_average_reward)/(abs(best_average_reward)+abs(average_reward)))<np.random.random():
        print("restored!")
        print(average_reward)
        # q_loss = np.inf
        # # for i in range(500):
        # j = 0
        # for i in range(1000):
        #   mini_batch_idx = np.random.choice(batch_size, self.mini_batch_size)
        #   q_network_loss, value_network_loss, _, _ = self.sess.run([self.q_network_loss, self.value_network_loss, self.train_value_network, self.train_q_network], {self.observations:observations_batch[mini_batch_idx,:], self.actions:actions_batch[mini_batch_idx,:], self.returns:returns_batch[mini_batch_idx,:], self.learning_rate:learning_rate})
        #   # mini_batch_idx = np.random.choice(self.total_returns_batch.shape[0], self.mini_batch_size)
        #   # observations_mini_batch = self.total_observations_batch[mini_batch_idx,:]
        #   # actions_mini_batch = self.total_actions_batch[mini_batch_idx,:]
        #   # returns_mini_batch = self.total_returns_batch[mini_batch_idx,:]
        #   # q_loss, _, _ = self.sess.run([self.q_network_loss, self.train_q_network, self.train_value_network], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.returns:returns_mini_batch, self.learning_rate:learning_rate})
        # last_q_loss = q_network_loss
        # self.min_batch_size += 100
        self.min_episodes += self.episode_increase 
        if self.min_episodes > 8:
          self.min_episodes = 8
        # self.min_episodes += self.episode_increase
        _ = self.sess.run([self.restore],{})
      else:
        if average_reward > best_average_reward:
          _ = self.sess.run([self.backup,{}])
          best_average_reward = average_reward
          saver.save(self.sess, save_dir)
          self.min_episodes -= self.episode_increase
          if self.min_episodes < 1:
            self.min_episodes = 1
        # self.replay_buffer_size = batch_size
        #
        q_loss = np.inf
        # for i in range(500):
        j = 0
        for i in range(300):
          mini_batch_idx = np.random.choice(batch_size, self.mini_batch_size)
          q_network_loss, value_network_loss, _, _ = self.sess.run([self.q_network_loss, self.value_network_loss, self.train_value_network, self.train_q_network], {self.observations:observations_batch[mini_batch_idx,:], self.actions:actions_batch[mini_batch_idx,:], self.returns:returns_batch[mini_batch_idx,:], self.learning_rate:learning_rate})
          # mini_batch_idx = np.random.choice(self.total_returns_batch.shape[0], self.mini_batch_size)
          # observations_mini_batch = self.total_observations_batch[mini_batch_idx,:]
          # actions_mini_batch = self.total_actions_batch[mini_batch_idx,:]
          # returns_mini_batch = self.total_returns_batch[mini_batch_idx,:]
          # q_loss, _, _ = self.sess.run([self.q_network_loss, self.train_q_network, self.train_value_network], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.returns:returns_mini_batch, self.learning_rate:learning_rate})
        last_q_loss = q_network_loss
        lr_in = 2e-2
        # mini updates  
        # while q_loss > 100:
        #   j += 1
        #   if j>10000:
        #     # import pdb
        #     # pdb.set_trace()
        #     break
        #   lr_in = 8e-3
        #   # if j<1000:
        #   #   lr_in = 5e-2
        #   # elif j<3000:
        #   #   lr_in = 1e-2
        #   # elif j<5000:
        #   #   lr_in = 5e-3
        #   # elif j<6000:
        #   #   lr_in = 1e-3
        #   # elif j<8000:
        #   #   lr_in = 5e-4
        #   # elif j<9000:
        #   #   lr_in = 2e-4
        #   # else:
        #   #   lr_in = 1e-4
        #   # mini_batch_idx = np.random.choice(batch_size, self.mini_batch_size)
        #   # q_loss, value_network_loss, _, _ = self.sess.run([self.q_network_loss, self.value_network_loss, self.train_value_network, self.train_q_network], {self.observations:observations_batch[mini_batch_idx,:], self.actions:actions_batch[mini_batch_idx,:], self.returns:returns_batch[mini_batch_idx,:], self.learning_rate:lr_in})      
        #   q_loss, value_network_loss, _, _ = self.sess.run([self.q_network_loss, self.value_network_loss, self.train_value_network, self.train_q_network], {self.observations:observations_batch, self.actions:actions_batch, self.returns:returns_batch, self.learning_rate:lr_in})

        for i in range(1):
          #
          # Taking the gradient step to optimize (train) the policy network (actor) and the value network (critic). mean and stddev is computed for kl divergence in the next step
          q_network_loss, value_network_loss, _, _ = self.sess.run([self.q_network_loss, self.value_network_loss, self.train_value_network, self.train_q_network], {self.observations:observations_batch, self.actions:actions_batch, self.returns:returns_batch, self.learning_rate:learning_rate})
          #
          # comute value and q value to calculate advantage
          q, v = self.sess.run([self.best_q_network.out, self.best_value_network.out], {self.observations:observations_batch, self.actions:actions_batch})
          advantages_batch = np.squeeze(q-v).reshape([-1,1])
          #
          # Taking the gradient step to optimize (train) the policy network (actor) and the value network (critic). mean and stddev is computed for kl divergence in the next step
          policy_network_losses, policy_network_loss, _ = self.sess.run([self.policy_network_losses, self.policy_network_loss, self.train_policy_network], {self.observations:observations_batch, self.actions:actions_batch, self.advantages:advantages_batch, self.returns:returns_batch, self.learning_rate:learning_rate, self.average_reward:average_reward})
          summary, average_advantage, kl  = self.sess.run([self.summary, self.average_advantage, self.kl], {self.observations:observations_batch, self.actions:actions_batch, self.advantages:advantages_batch, self.returns:returns_batch, self.learning_rate:learning_rate, self.average_reward:average_reward})
        
        
        
          # q_loss, _, _ = self.sess.run([self.q_network_loss, self.train_q_network, self.train_value_network], {self.observations:observations_batch, self.actions:actions_batch, self.returns:returns_batch, self.learning_rate:learning_rate})
        q_network_loss
        #   # mini_batch_idx = np.random.choice(self.total_returns_batch.shape[0], self.mini_batch_size)
          # observations_mini_batch = self.total_observations_batch[mini_batch_idx,:]
          # actions_mini_batch = self.total_actions_batch[mini_batch_idx,:]
          # returns_mini_batch = self.total_returns_batch[mini_batch_idx,:]
          # q_loss, _, _ = self.sess.run([self.q_network_loss, self.train_q_network, self.train_value_network], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.returns:returns_mini_batch, self.learning_rate:learning_rate})

        _ = self.sess.run([self.last,{}])      
        #
        # shape of the policy_network_losses is check since its common to have nonsense size due to unintentional matmul instead of non element wise multiplication. Action dimension and advantage dimension are very important and hard to debug when not correct
        if isinstance(policy_network_losses, list):
          policy_network_losses = policy_network_losses[0]
        assert policy_network_losses.shape==actions_batch.shape, "Dimensions mismatch. Policy Distribution is incorrect! " + str(policy_network_losses.shape)
        #
        ##### Reporting Performance #####
        #
        # Printing performance progress and other useful infromation
        print("_______________________________________________________________________________________________________________________________________________________________________________________________________________")
        print("{:>15} {:>15} {:>10} {:>15} {:>15} {:>20} {:>20} {:>20} {:>20} {:>20} {:>10} {:>15}".format("total_timesteps", "episodes", "iteration", "best_reward", "reward", "kl_divergence", "policy_loss", "value_loss", "q_network_loss", "average_advantage", "lr", "batch_size"))
        print("{:>15} {:>15} {:>10} {:>15.2f} {:>15.2f} {:>20.5E} {:>20.2f} {:>20.2f} {:>20.2f} {:>20.2f} {:>10.2E} {:>15}".format(total_timesteps, total_episodes,iteration, best_average_reward, average_reward, kl, policy_network_loss, value_network_loss, q_network_loss, average_advantage, self.lr, batch_size))
        self.writer.add_summary(summary, total_timesteps)

    self.writer.close()
  
