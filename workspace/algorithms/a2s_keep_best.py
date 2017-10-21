import numpy as np 
import tensorflow as tf 
import importlib
from algorithms.utils import *
import itertools


class Agent:
  """
  Advantage Actor Critic Algorithm
  """
  def __init__(self,
               env,
               sess,
               network_type='fully_connected_network',
               network_size='small',
               iterations=500,
               min_batch_size=1000,
               lr=1e-3,
               lr_schedule='linear',
               gamma=0.99,
               animate=True,
               logs_path="/home/user/workspace/logs/",
               number_of_suggestions=5,
               mini_batch_size=32,
               mini_iterations=350,
               batch_rate=1.3):
    #
    # Tensorflow Session
    self.sess = sess
    #
    # OpenAI Environment
    self.env = env
    #
    # Hyper Parameters
    self.lr = lr
    self.iterations = iterations
    self.min_batch_size = min_batch_size
    self.batch_rate = batch_rate
    self.gamma = gamma
    self.number_of_suggestions = number_of_suggestions
    self.mini_batch_size = mini_batch_size
    self.mini_iterations = mini_iterations
    #
    # importing the desired network architecture
    self.network_class = importlib.import_module("architectures." + network_type)
    #
    # getting the shape of observation space and action space of the environment
    self.observation_shape = self.env.observation_space.shape[0]
    self.action_shape = self.env.action_space.shape[0]
    #
    # scoping variables with A2S
    with tf.variable_scope("A2S"):
      #
      ##### Placeholders #####
      #
      # placeholder for observation
      self.observations = tf.placeholder(tf.float32, shape=[None, self.observation_shape], name="observations")
      #
      # placeholder for actions taken, this is used for the policy gradient
      self.actions = tf.placeholder(tf.float32, shape=[None, self.action_shape], name="actions")
      #
      # placeholder for the advantage function
      self.advantages = tf.placeholder(tf.float32, shape=[None, 1], name="advantages")
      #
      # placeholder for learning rate for the optimizer
      self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
      #
      # mean and the stddev of the old policy distrubution for KL calculation
      self.mean_policy_old = tf.placeholder(tf.float32, name="mean_policy_old")
      self.stddev_policy_old = tf.placeholder(tf.float32, name="stddev_policy_old")
      #
      # placeholder for the r + gamma*next_value
      self.returns = tf.placeholder(tf.float32, shape=[None, 1], name="returns")
      #
      ##### Networks #####
      #
      # backup policy network, backs up the best poloicy in case the update is bad. outputs the mean for the action
      self.backup_policy_network = self.network_class.Network(sess, self.observations, self.action_shape, "backup_policy_network", network_size)
      self.backup_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='backup_policy_network')
      #
      # the current best policy that will be updated, if the updated policy performs worse, the policy will be restored from the back up policy. 
      self.best_policy_network = self.network_class.Network(sess, self.observations, self.action_shape, "best_policy_network", network_size)
      self.best_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='best_policy_network')
      #
      # backup value network, outputs value of observation, used for advantage estimate
      self.backup_value_network = self.network_class.Network(sess, self.observations, 1, "backup_value_network", network_size,)
      self.backup_value_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='backup_value_network')
      #
      # best value network
      self.best_value_network = self.network_class.Network(sess, self.observations, 1, "best_value_network", network_size)
      self.best_value_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='best_value_network')
      #
      ##### Policy Action Probability #####
      #
      # isolating the mean outputted by the policy network
      self.backup_mean_policy = tf.reshape(self.backup_policy_network.out,[-1, self.action_shape])
      self.best_mean_policy = tf.reshape(self.best_policy_network.out,[-1, self.action_shape])
      #
      # isolating the stddev outputted by the policy network, softplus is used to make sure that the stddev is positive
      self.backup_stddev_policy = tf.constant(0.5)
      self.best_stddev_policy = tf.constant(0.5)
      #
      # gaussian distribution is built with mean and stddev from the policy network
      self.backup_gaussian_policy_distribution = tf.contrib.distributions.Normal(self.backup_mean_policy, self.backup_stddev_policy)
      self.best_gaussian_policy_distribution = tf.contrib.distributions.Normal(self.best_mean_policy, self.best_stddev_policy)
      #
      # KL divergence from old policy distribution to the new policy distribution
      self.kl = tf.reduce_mean(tf.contrib.distributions.kl_divergence(self.best_gaussian_policy_distribution, self.backup_gaussian_policy_distribution))
      tf.summary.scalar('kl', self.kl)
      #
      # action suggested by the policy network
      self.suggested_actions_out = tf.reshape(self.best_gaussian_policy_distribution._sample_n(self.number_of_suggestions),[-1,self.number_of_suggestions,self.action_shape])
      self.state_actions = tf.concat([self.observations, self.actions],1)      
      ##### Q Network #####
      #
      # backup q network, outputs value of observation, used for advantage estimate
      self.backup_q_network = self.network_class.Network(self.sess, self.state_actions, 1, "backup_q_network", network_size)
      self.backup_q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='backup_q_network')
      #
      # best q network
      self.best_q_network = self.network_class.Network(self.sess, self.state_actions, 1, "best_q_network", network_size)
      self.best_q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='best_q_network')
      #
      ##### Loss #####
      #
      # loss for the policy network and value network
      self.negative_log_prob = -self.best_gaussian_policy_distribution.log_prob(self.actions)
      self.policy_network_losses = self.negative_log_prob*self.advantages # be careful with this operation, it should be element wise not matmul!
      self.average_advantage = tf.reduce_mean(self.advantages)
      tf.summary.scalar('average_advantage', self.average_advantage)
      self.policy_network_loss = tf.reduce_mean(self.policy_network_losses)
      tf.summary.scalar('policy_network_loss', self.policy_network_loss)
      self.value_network_y = self.returns
      self.value_network_loss = tf.reduce_mean(tf.squared_difference(self.best_value_network.out, self.value_network_y))
      tf.summary.scalar('value_network_loss', self.value_network_loss)
      self.q_network_y = self.returns
      self.q_network_loss = tf.reduce_mean(tf.squared_difference(self.best_q_network.out,self.q_network_y))
      tf.summary.scalar('q_network_loss', self.q_network_loss)
      #
      ##### Optimization #####
      #
      # optimizer for the policy network and value network
      self.policy_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      self.value_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      self.q_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      #
      # training operation for the policy network and value network
      self.train_policy_network = self.policy_network_optimizer.minimize(self.policy_network_loss)
      self.train_value_network = self.value_network_optimizer.minimize(self.value_network_loss)
      self.train_q_network = self.q_network_optimizer.minimize(self.q_network_loss)
      #
      ##### Update #####
      #
      # update 
      self.restore = []
      for var, var_target in zip(sorted(self.backup_policy_network_vars,key=lambda v: v.name),sorted(self.best_policy_network_vars, key=lambda v: v.name)):
        self.restore.append(var_target.assign(var))
      
      for var, var_target in zip(sorted(self.backup_value_network_vars,key=lambda v: v.name),sorted(self.best_value_network_vars, key=lambda v: v.name)):
        self.restore.append(var_target.assign(var))
      for var, var_target in zip(sorted(self.backup_q_network_vars,key=lambda v: v.name),sorted(self.best_q_network_vars, key=lambda v: v.name)):
        self.restore.append(var_target.assign(var))
      self.restore = tf.group(*self.restore)

      self.backup = []
      for var, var_target in zip(sorted(self.best_policy_network_vars,key=lambda v: v.name),sorted(self.backup_policy_network_vars, key=lambda v: v.name)):
        self.backup.append(var_target.assign(var))
      
      for var, var_target in zip(sorted(self.best_value_network_vars,key=lambda v: v.name),sorted(self.backup_value_network_vars, key=lambda v: v.name)):
        self.backup.append(var_target.assign(var))
      for var, var_target in zip(sorted(self.best_q_network_vars,key=lambda v: v.name),sorted(self.backup_q_network_vars, key=lambda v: v.name)):
        self.backup.append(var_target.assign(var))
      self.backup = tf.group(*self.backup)

      self.summary = tf.summary.merge_all()
    #
    # initialize all tf variables
    self.writer = tf.summary.FileWriter(logs_path, sess.graph)
    self.sess.run(tf.global_variables_initializer())
  #
  # samples action from the current policy network
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
  #
  # computes the value for a given observation
  def compute_value(self, observations):
    return self.sess.run(self.best_value_network.out, {self.observations: observations})
  #
  #
  def compute_q(self, observation, actions):
    return self.sess.run([self.best_q_network.out],{self.observations: observation[None], self.actions:actions})[0]
  #
  # Collecting experience (data) and training the agent (networks)
  def train(self, saver=None, save_dir=None):
    #
    # keeping count of total timesteps of environment experience
    total_timesteps = 0
    total_episodes = 0
    #
    # sum of rewards of 100 episodes, not for learning, just a performance metric and logging
    trajectory_rewards = []
    #
    # kl divergence, used to adjust the learning rate
    kl = 0
    #
    # keeping track of the best averge reward
    best_average_reward = -np.inf
    #
    ##### Training #####
    #
    # training iterations
    for iteration in range(self.iterations):
      #
      # batch size changes from episode to episode
      batch_size = 0
      trajectories, returns = [], []
      #
      ##### Collect Batch #####
      #
      # collecting minium batch size of experience
      while batch_size < self.min_batch_size:
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
        if len(trajectory_rewards) > 100:
          trajectory_rewards.pop(0)
        #
        # get sum of reward for this episode
        trajectory_rewards.append(np.sum(rewards))
        #
        # add timesteps of this episode to batch_size
        batch_size += len(rewards)
        #
        # episode trajectory
        trajectory = {"observations":np.array(observations), "actions":np.array(actions), "rewards":np.array(rewards), "dones":np.array(dones)}
        trajectories.append(trajectory)
        #
        # computing the discounted return for this episode (NOT A SINGLE NUMBER, FOR EACH OBSERVATION)
        return_ = discount(trajectory["rewards"], self.gamma)
        returns.append(return_)
      #
      ##### Data Prep #####
      #
      # total timesteps is sum of all batch size
      total_timesteps += batch_size
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
      # learning rate
      learning_rate = self.lr
      # 
      ##### Optimization #####
      #
      # average reward of past 100 episodes
      average_reward = np.mean(trajectory_rewards[-100:])
      if average_reward > best_average_reward:
        _ = sess.run([self.backup,{}])
      else:
        _ = sess.run([self.restore],{})
      for i in range(self.mini_iterations):
        mini_batch_idx = np.random.choice(batch_size, self.mini_batch_size)
        observations_mini_batch = observations_batch[mini_batch_idx,:]
        actions_mini_batch = actions_batch[mini_batch_idx,:]
        returns_mini_batch = returns_batch[mini_batch_idx,:]
        _, _ = self.sess.run([self.train_q_network, self.train_value_network], {self.observations:observations_mini_batch, self.actions:actions_mini_batch, self.returns:returns_mini_batch, self.learning_rate:learning_rate})
      #
      # Taking the gradient step to optimize (train) the policy network (actor) and the value network (critic). mean and stddev is computed for kl divergence in the next step
      q_network_loss, value_network_loss, _, _ = self.sess.run([self.q_network_loss, self.value_network_loss, self.kl, self.train_q_network, self.train_value_network], {self.observations:observations_batch, self.actions:actions_batch, self.returns:returns_batch, self.learning_rate:learning_rate*10})
      #
      # comute value and q value to calculate advantage
      q, v = self.sess.run([self.q_network.out, self.value_network.out], {self.observations:observations_batch, self.actions:actions_batch})
      advantages_batch = np.squeeze(q-v).reshape([-1,1])
      #
      # Taking the gradient step to optimize (train) the policy network (actor) and the value network (critic). mean and stddev is computed for kl divergence in the next step
      summary, kl, policy_network_losses, policy_network_loss, _ = self.sess.run([self.kl, self.policy_network_losses, self.policy_network_loss, self.train_policy_network], {self.observations:observations_batch, self.actions:actions_batch, self.advantages:advantages_batch, self.learning_rate:learning_rate})
      #
      # shape of the policy_network_losses is check since its common to have nonsense size due to unintentional matmul instead of non element wise multiplication. Action dimension and advantage dimension are very important and hard to debug when not correct
      if isinstance(policy_network_losses, list):
        policy_network_losses = policy_network_losses[0]
      assert policy_network_losses.shape==actions_batch.shape, "Dimensions mismatch. Policy Distribution is incorrect! " + str(policy_network_losses.shape)
      #
      ##### Reporting Performance #####
      #
      # update best average reward and save only if this is the best performing network so far
      if average_reward > best_average_reward:
        best_average_reward = average_reward
        saver.save(self.sess, save_dir)
      #
      # Printing performance progress and other useful infromation
      print("______________________________________________________________________________________________________________________________________________________________________________")
      print("{:>15} {:>15} {:>10} {:>15} {:>15} {:>20} {:>20} {:>20} {:>20} {:>20}".format("total_timesteps", "episodes", "iteration", "best_reward", "reward", "kl_divergence", "policy_loss", "value_loss", "q_network_loss", "average_advantage"))
      print("{:>15} {:>15} {:>10} {:>15.2f} {:>15.2f} {:>20.5f} {:>20.2f} {:>20.2f} {:>20.2f} {:>20.2f}".format(total_timesteps, total_episodes,iteration, best_average_reward, average_reward, kl, policy_network_loss, value_network_loss, q_network_loss, average_advantage))
      self.writer.add_summary(summary, total_timesteps)

    self.writer.close()
  
