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
      'std_dev' (list): Different ways to define the standard devation of the policy e.g. ['fixed'/'linear'/'network', (double)/(double)/-] NOTE: network option has the policy network to output the std dev 
      'experience_replay' (None/string): Which experience replay to use e.g. None/'ER'/'PER' NOTE: no experience replay, experience replay, priotized experience replay
      'experience_replay_size' (int): Experience replay buffer size 
      'ER_batch_size' (int): Experience replay buffer size 
      'ER_iterations' (int): Number of iterations to train from the experience replay buffer
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
                                  'ER_iterations':300,
                                  'PER_alpha':0.6, 
                                  'PER_epsilon':0.01},
               logs_path="/home/user/workspace/logs/"):

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
      
      # Mean and the std dev of the last policy distrubution for KL calculation
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
      else:
        policy_output_shape = self.action_shape

      # Current policy that will be updated and used to act, if the updated policy performs worse, the policy will be restored from the back up policy. 
      self.current_policy_network = self.policy_network_class.Network(sess, self.observations, policy_output_shape, "current_policy_network", self.network_params['policy_network'][1])
      self.current_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A2S/current_policy_network')

      # Backup of the best policy network so far, backs up the best policy in case the update is bad. outputs the mean for the action
      self.best_policy_network = self.policy_network_class.Network(sess, self.observations, policy_output_shape, "best_policy_network", self.network_params['policy_network'][1])
      self.best_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A2S/best_policy_network')

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
      
      # Isolating the std dev outputted by the policy network, softplus is used to make sure that the std dev is positive
      if self.algorithm_params['std_dev'][0]=='network':
        self.current_std_dev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.current_policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
        self.best_std_dev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.best_policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
        self.last_std_dev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.last_policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
      else:
        self.current_std_dev_policy = tf.constant(self.algorithm_params['std_dev'][1])
        self.best_std_dev_policy = tf.constant(self.algorithm_params['std_dev'][1])
        self.last_std_dev_policy = tf.constant(self.algorithm_params['std_dev'][1])
      
      # Gaussian distribution is built with mean and std dev from the policy network
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
      self.current_q_network = self.q_network_class.Network(self.sess, self.state_action_pairs, 1, "current_q_network", self.network_params['q_network'][1])
      self.current_q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='A2S/current_q_network')

      # Best Q network, outputs Q value of observation, used for advantage estimate
      self.best_q_network = self.q_network_class.Network(self.sess, self.state_action_pairs, 1, "best_q_network", self.network_params['q_network'][1])
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
    for i in range(self.algorithm_params['number_of_suggestions']):
      current_q = self.compute_q_current(observation, suggested_actions[0][:,i,:])
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

  # Compute Q value from current Q network
  def compute_q_current(self, observation, action):
    q_value = self.sess.run([self.current_q_network.out],{self.observations: observation[None], self.actions:action})[0]

    return q_value

  # Computes the value for given observations
  def compute_value(self, observations):
    return self.sess.run(self.current_value_network.out, {self.observations: observations})
  
  # Restore the best networks
  def networks_restore(self):
    _ = self.sess.run([self.restore],{})
  
  # Backup the current networks
  def networks_backup(self):
    _ = self.sess.run([self.backup],{})
  
  # Backup the current policy to last policy network for KL divergence calculation
  def backup_current_policy_to_last_policy(self):
    _ = self.sess.run([self.last],{})

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

      # Batch size and episodes experienced in current iteration
      batch_size = 0
      episodes = 0

      # Lists to collect data
      trajectories, returns, undiscounted_returns = [], [], []

      ##### Collect Batch #####

      # Collecting minium batch size or minimum episodes of experience
      while episodes < self.data_collection_params['min_episodes'] or batch_size < self.data_collection_params['min_batch_size']:
        
        # Restart env
        observation = self.env.reset()
        
        # Data for this episode
        observations, actions, rewards, dones = [], [], [], []
        
        # Flag that env is in terminal state
        done = False

        ##### Episode #####
        
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

        ##### Data Appending #####
        
        # Get sum of reward for this episode
        undiscounted_returns.append(np.sum(rewards))
        
        # Update the counters
        batch_size += len(rewards)
        total_timesteps += len(rewards)
        episodes += 1

        # Log reward
        reward_summary = self.sess.run([self.reward_summary], {self.average_reward:np.sum(rewards)})[0]
        self.writer.add_summary(reward_summary, total_timesteps)
        
        # Episode trajectory
        trajectory = {"observations":np.array(observations), "actions":np.array(actions), "rewards":np.array(rewards), "dones":np.array(dones)}
        trajectories.append(trajectory)
        
        # Computing the discounted return for this episode (NOT A SINGLE NUMBER, FOR EACH OBSERVATION)
        return_ = discount(trajectory["rewards"], self.rl_params['gamma'])
        returns.append(return_)
      
      ##### Data Prep #####
      
      # Observations for this batch
      observations_batch = np.concatenate([trajectory["observations"] for trajectory in trajectories])
      
      # Actions for this batch, reshapeing to handel 1D action space
      actions_batch = np.concatenate([trajectory["actions"] for trajectory in trajectories]).reshape([-1,self.action_shape])
      
      # Discounted returns for this batch. itertool used to make batch into long np array
      rewards_batch = np.concatenate([trajectory["rewards"] for trajectory in trajectories]).reshape([-1,1])
      
      # Calc number of episodes
      dones_batch = np.concatenate([trajectory["dones"] for trajectory in trajectories])
      total_episodes += np.sum(dones_batch)
      
      # Discounted returns for this batch. itertool used to make batch into long np array
      returns_batch = np.array(list(itertools.chain.from_iterable(returns))).reshape([-1,1])
      
      # Learning rate adaptation
      if self.training_params['adaptive_lr']:
        if kl > self.training_params['desired_kl'] * 2: 
          self.training_params['learning_rate'] /= 1.5
        elif kl < self.training_params['desired_kl'] / 2: 
          self.training_params['learning_rate'] *= 1.5
      learning_rate = self.training_params['learning_rate']

      ##### Optimization #####

      # Average undiscounted return for the last data collection
      average_reward = np.mean(undiscounted_returns)

      # Heuristic method for restoring when average reward is sufficently worse than the best average reward
      if average_reward < best_average_reward and 1-(abs(average_reward- best_average_reward)/(abs(best_average_reward)+abs(average_reward)))<np.random.random() and self.algorithm_params['restore']:
        
        # Report the bad average reward that caused a restore
        print("RESTORED due to average reward of {}".format(average_reward))

        # Adaptive minimum episode size. Increase by episode_adapt_rate 
        self.data_collection_params['min_episodes'] += self.data_collection_params['episode_adapt_rate'] 
        # self.algorithm_params['number_of_suggestions'] -= 1

        # if self.algorithm_params['number_of_suggestions'] < 1:
          # self.algorithm_params['number_of_suggestions'] = 1

        # Max episodes allowed per iteration
        # if self.data_collection_params['min_episodes'] > 8:
          # self.data_collection_params['min_episodes'] = 8

        # Restore the best networks
        self.networks_restore()
      else:

        if average_reward > best_average_reward:

          # Backup the networks when the current networks perform better than the best
          if self.algorithm_params['restore']:
            self.networks_backup()
          best_average_reward = average_reward

          # Save the model
          saver.save(self.sess, save_dir)

          # Adaptive minimum episode size. Decrease by episode_adapt_rate 
          self.data_collection_params['min_episodes'] -= self.data_collection_params['episode_adapt_rate']
          if self.data_collection_params['min_episodes'] < 1:
            self.data_collection_params['min_episodes'] = 1

          # self.algorithm_params['number_of_suggestions'] += 1
          # if self.algorithm_params['number_of_suggestions'] > 6:
            # self.algorithm_params['number_of_suggestions'] = 6
        # Train with Experience Replay
        if self.algorithm_params['experience_replay'] is not None:
          for i in range(batch_size):
            if self.algorithm_params['experience_replay'] == 'PER':
              q_value_estimate = self.compute_q(observations_batch[i,:], actions_batch[i,:][None])
              error = returns_batch[i,:][0] - q_value_estimate[0][0]
              sample_ER = ExperienceReplayData(observations_batch[i,:], actions_batch[i,:], returns_batch[i,:], error)
              self.replay_buffer.add(error, sample_ER)
            else:
              sample_ER = ExperienceReplayData(observations_batch[i,:], actions_batch[i,:], returns_batch[i,:])
              self.replay_buffer.add(sample_ER)

          for i in range(self.algorithm_params['ER_iterations']):
            sample_batch = self.replay_buffer.sample(self.algorithm_params['ER_batch_size'])
            observations_mini_batch, actions_mini_batch, returns_mini_batch = [], [], []
            for sample in sample_batch:
              # Build the mini batch from sample batch
              observations_mini_batch.append(sample[1].observation)
              actions_mini_batch.append(sample[1].action)
              returns_mini_batch.append(sample[1].return_)

            observations_mini_batch = np.array(observations_mini_batch)
            actions_mini_batch = np.array(actions_mini_batch)
            returns_mini_batch = np.array(returns_mini_batch)

            # Training with sample batch
            q_network_loss, _, value_network_loss, _ = self.sess.run([self.q_network_loss, self.train_q_network, self.value_network_loss, self.train_value_network], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.returns:returns_mini_batch, self.learning_rate:learning_rate})

            if self.algorithm_params['experience_replay'] == 'PER':
              for sample in sample_batch:
                # Update error for the sample batch 
                q_value_estimate = self.compute_q(sample[1].observation, sample[1].action[None])
                error = sample[1].return_ - q_value_estimate[0][0]
                self.replay_buffer.update(sample[0], error)

        # Compute std dev
        if self.algorithm_params['std_dev'][0] == 'linear':
          max_std_dev = 2.0
          self.algorithm_params['std_dev'][1] += 0.005
          if self.algorithm_params['std_dev'][1] > max_std_dev:
            self.algorithm_params['std_dev'][1] = max_std_dev

        # Train with Policy Graident
        for i in range(self.rl_params['num_policy_update']):
          
          # Taking the gradient step to optimize Q network and the value network with the whole batch.
          q_network_loss, value_network_loss, _, _ = self.sess.run([self.q_network_loss, self.value_network_loss, self.train_value_network, self.train_q_network], {self.observations:observations_batch, self.actions:actions_batch, self.returns:returns_batch, self.learning_rate:learning_rate*10})
          
          # Compute value and q value to calculate advantage
          if self.algorithm_params['restore']:
            q, v = self.sess.run([self.best_q_network.out, self.best_value_network.out], {self.observations:observations_batch, self.actions:actions_batch})          
          else:
            q, v = self.sess.run([self.current_q_network.out, self.current_value_network.out], {self.observations:observations_batch, self.actions:actions_batch})
          # if self.algorithm_params['number_of_suggestions']!=1:          
          #   advantages_batch = np.squeeze(q-v).reshape([-1,1])
          # else:
          advantages_batch = np.squeeze(returns_batch-np.mean(returns_batch)).reshape([-1,1])
          # advantages_batch = np.squeeze(returns_batch-v).reshape([-1,1])
          # print(np.mean(returns_batch-q))
          
          # if np.random.random() > 0.5:
          #   advantages_batch = np.squeeze(returns_batch-v).reshape([-1,1])
          # else:
          # advantages_batch = np.squeeze(q-v).reshape([-1,1])
          
          # Taking the gradient step to optimize (train) the policy network.
          policy_network_losses, policy_network_loss, _ = self.sess.run([self.policy_network_losses, self.policy_network_loss, self.train_policy_network], {self.observations:observations_batch, self.actions:actions_batch, self.advantages:advantages_batch, self.returns:returns_batch, self.learning_rate:learning_rate, self.average_reward:average_reward})

          # Get stats
          summary, average_advantage, kl  = self.sess.run([self.summary, self.average_advantage, self.kl], {self.observations:observations_batch, self.actions:actions_batch, self.advantages:advantages_batch, self.returns:returns_batch, self.learning_rate:learning_rate, self.average_reward:average_reward})
               
        # Backup the current policy network to last policy network
        self.backup_current_policy_to_last_policy()
        
        # Shape of the policy_network_losses is check since its common to have nonsense size due to unintentional matmul instead of non element wise multiplication. Action dimension and advantage dimension are very important and hard to debug when not correct
        if isinstance(policy_network_losses, list):
          policy_network_losses = policy_network_losses[0]
        assert policy_network_losses.shape==actions_batch.shape, "Dimensions mismatch. Policy Distribution is incorrect! " + str(policy_network_losses.shape)
        
        ##### Reporting Performance #####
        
        # Printing performance progress and other useful infromation
        print("_______________________________________________________________________________________________________________________________________________________________________________________________________________")
        print("{:>15} {:>15} {:>15} {:>15} {:>20} {:>20} {:>20} {:>20} {:>20} {:>10} {:>15}".format("total_timesteps", "episodes", "best_reward", "reward", "kl_divergence", "policy_loss", "value_loss", "q_network_loss", "average_advantage", "lr", "batch_size"))
        print("{:>15} {:>15} {:>15.2f} {:>15.2f} {:>20.5E} {:>20.2f} {:>20.2f} {:>20.2f} {:>20.2f} {:>10.2E} {:>15}".format(total_timesteps, total_episodes, best_average_reward, average_reward, kl, policy_network_loss, value_network_loss, q_network_loss, average_advantage, learning_rate, batch_size))

        # Write summary for tensorboard visualization
        self.writer.add_summary(summary, total_timesteps)

    self.writer.close()
  
