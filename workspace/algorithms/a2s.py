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
               animate=True):
    #
    # Tensorflow Session
    self.sess = sess
    #
    # OpenAI Environment
    self.env = env
    #
    # Hyper Parameters
    self.learning_rate_scheduler = Scheduler(v=lr, nvalues=iterations*2, schedule=lr_schedule)
    self.iterations = iterations
    self.min_batch_size = min_batch_size
    self.gamma = gamma
    self.number_of_suggestions = 16
    self.update_freq = 10
    #
    # animate the environment while training
    self.animate = animate
    #
    # importing the desired network architecture
    self.network_class = importlib.import_module("architectures."+network_type)
    #
    # getting the shape of observation space and action space of the environment
    self.observation_shape = self.env.observation_space.shape[0]
    self.action_shape = self.env.action_space.shape[0]
    #
    # scoping variables with A2C
    with tf.variable_scope("A2S"):
      #
      ##### Placeholders #####
      #
      # placeholder for observation
      self.observations = tf.placeholder(tf.float32, shape=[None, self.observation_shape], name="observations")
      self.next_observations = tf.placeholder(tf.float32, shape=[None, self.observation_shape], name="next_observations")
      #
      # placeholder for actions taken, this is used for the policy gradient
      self.actions = tf.placeholder(tf.float32, shape=[None, self.action_shape], name="actions")
      #
      # placeholder for the advantage function
      self.advantages = tf.placeholder(tf.float32, shape=[None, 1], name="advantages")
      #
      # placeholder for the rewards
      self.rewards = tf.placeholder(tf.float32, shape=[None, 1], name="rewards")
      #
      # placeholder for learning rate for the optimizer
      self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
      #
      # mean and the stddev of the old policy distrubution for KL calculation
      self.mean_policy_old = tf.placeholder(tf.float32, name="mean_policy_old")
      self.stddev_policy_old = tf.placeholder(tf.float32, name="stddev_policy_old")
      self.done_mask = tf.placeholder(tf.float32, shape=[None], name="done_mask")
      self.suggested_actions = tf.placeholder(tf.float32, shape=[None, None, self.action_shape], name="suggested_actions")
      #
      ##### Networks #####
      #
      #
      self.shared_network = self.network_class.Network(sess, self.observations, self.observation_shape, "shared_network", network_size)
      self.shared_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared_network')
      self.target_shared_network = self.network_class.Network(sess, self.next_observations, self.observation_shape, "target_shared_network", network_size) 
      self.target_shared_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_shared_network')
      #
      # policy network, outputs the mean and the stddev (needs to go through softplus) for the action
      self.policy_network = self.network_class.Network(sess, self.shared_network.out, self.action_shape, "policy_network", network_size)
      self.policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy_network')
      self.target_policy_network = self.network_class.Network(sess, self.target_shared_network.out, self.action_shape, "target_policy_network", network_size)
      self.target_policy_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_policy_network')
      #
      # value network, outputs value of observation, used for advantage estimate
      self.value_network = self.network_class.Network(sess, self.shared_network.out, 1, "value_network", network_size)
      self.value_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value_network')
      self.target_value_network = self.network_class.Network(sess, self.target_shared_network.out, 1, "target_value_network", network_size)
      self.target_value_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_value_network')
      #
      ##### Policy Action Probability #####
      #
      # isolating the mean outputted by the policy network
      self.mean_policy = tf.reshape(self.policy_network.out,[-1, self.action_shape])
      self.target_mean_policy = tf.reshape(self.target_policy_network.out,[-1, self.action_shape])
      #
      # isolating the stddev outputted by the policy network, softplus is used to make sure that the stddev is positive
      self.stddev_policy = tf.constant(1.0)
      self.target_stddev_policy = tf.constant(1.0)
      #
      # gaussian distribution is built with mean and stddev from the policy network
      self.gaussian_policy_distribution = tf.contrib.distributions.Normal(self.mean_policy, self.stddev_policy)
      self.target_gaussian_policy_distribution = tf.contrib.distributions.Normal(self.target_mean_policy, self.target_stddev_policy)
      #
      # gaussian distribution is built with mean and stddev from the old policy network
      self.gaussian_policy_distribution_old = tf.contrib.distributions.Normal(self.mean_policy_old, self.stddev_policy_old)
      #
      # KL divergence from old policy distribution to the new policy distribution
      self.kl = tf.reduce_mean(tf.contrib.distributions.kl_divergence(self.gaussian_policy_distribution, self.gaussian_policy_distribution_old))
      #
      # action suggested by the policy network
      self.suggested_actions_out = tf.reshape(self.gaussian_policy_distribution._sample_n(self.number_of_suggestions),[-1,self.number_of_suggestions,self.action_shape])
      self.shared_net_tile = tf.reshape(tf.tile(self.shared_network.out, [1,self.number_of_suggestions]),[-1,self.number_of_suggestions,self.observation_shape])
      
      self.state_actions = tf.reshape(tf.concat(tf.split(self.suggested_actions,self.action_shape,2)+tf.split(self.shared_net_tile,self.observation_shape,2),2),[-1,self.action_shape+self.observation_shape])
      self.target_suggested_actions = tf.reshape(self.target_gaussian_policy_distribution._sample_n(self.number_of_suggestions),[-1,self.number_of_suggestions,self.action_shape])
      self.target_shared_net_tile = tf.reshape(tf.tile(self.target_shared_network.out, [1,self.number_of_suggestions]),[-1,self.number_of_suggestions,self.observation_shape])
      self.target_state_actions = tf.reshape(tf.concat(tf.split(self.target_suggested_actions,self.action_shape,2)+tf.split(self.target_shared_net_tile,self.observation_shape,2),2),[-1,self.action_shape+self.observation_shape])
      ##### Q Network #####
      #
      # q network, outputs value of observation, used for advantage estimate
      self.q_network = self.network_class.Network(self.sess, self.state_actions, 1, "q_network", network_size)
      self.q_values = tf.reshape(self.q_network.out,[-1,self.number_of_suggestions])
      self.q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_network')
      self.target_q_network = self.network_class.Network(self.sess, self.target_state_actions, 1, "target_q_network", network_size)
      self.target_q_values = tf.reshape(self.target_q_network.out,[-1,self.number_of_suggestions])
      self.target_q_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_network')
      #
      ##### Action Selection #####
      self.selected_action_idxs = tf.argmax(self.q_values, axis=1)
      self.selected_actions = self.suggested_actions[:,tf.squeeze(self.selected_action_idxs),:]
      #
      ##### Loss #####
      #
      # loss for the policy network and value network
      self.negative_log_prob = -self.gaussian_policy_distribution.log_prob(self.actions)
      self.policy_network_losses = self.negative_log_prob*self.advantages # be careful with this operation, it should be element wise not matmul!
      self.policy_network_loss = tf.reduce_mean(self.policy_network_losses)
      self.value_network_y = tf.squeeze(self.rewards) + tf.multiply((1-self.done_mask),self.gamma*tf.reduce_mean(self.target_value_network.out, axis=1))
      self.value_network_loss = tf.reduce_mean(tf.squared_difference(self.value_network.out, self.value_network_y))
      self.q_action_taken = tf.squeeze(self.q_values[:,0])
      self.q_network_y = tf.squeeze(self.rewards) + tf.multiply((1-self.done_mask),self.gamma*tf.reduce_max(self.target_q_values, axis=1))
      self.q_network_loss = tf.reduce_mean(tf.squared_difference(self.q_action_taken,self.q_network_y))
      #
      ##### Optimization #####
      #
      # optimizer for the policy network and value network
      self.policy_network_optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=1e-2)
      self.value_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate*10)
      self.q_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate*10)
      #
      # training operation for the policy network and value network
      self.train_policy_network = self.policy_network_optimizer.minimize(self.policy_network_loss)
      self.train_value_network = self.value_network_optimizer.minimize(self.value_network_loss)
      self.train_q_network = self.q_network_optimizer.minimize(self.q_network_loss)
      #
      ##### Update #####
      #
      # update 
      self.update_target_fn = []
      
      for var, var_target in zip(sorted(self.shared_network_vars,key=lambda v: v.name),sorted(self.target_shared_network_vars, key=lambda v: v.name)):
        self.update_target_fn.append(var_target.assign(var))
      
      for var, var_target in zip(sorted(self.policy_network_vars,key=lambda v: v.name),sorted(self.target_policy_network_vars, key=lambda v: v.name)):
        self.update_target_fn.append(var_target.assign(var))
      
      for var, var_target in zip(sorted(self.value_network_vars,key=lambda v: v.name),sorted(self.target_value_network_vars, key=lambda v: v.name)):
        self.update_target_fn.append(var_target.assign(var))
      
      for var, var_target in zip(sorted(self.q_network_vars,key=lambda v: v.name),sorted(self.target_q_network_vars, key=lambda v: v.name)):
        self.update_target_fn.append(var_target.assign(var))
      
      self.update_target_fn = tf.group(*self.update_target_fn)

    #
    # initialize all tf variables
    self.sess.run(tf.global_variables_initializer())

  # samples action from the current policy network
  def compute_action(self, observation):
    suggested_actions = self.sess.run([self.suggested_actions_out], {self.observations: observation[None]})
    return self.sess.run([self.selected_actions],{self.observations: observation[None], self.suggested_actions:suggested_actions[0]})[0]

  # computes the value for a given observation
  def compute_value(self, observations):
    return self.sess.run(self.value_network.out, {self.observations: observations})

  # Collecting experience (data) and training the agent (networks)
  def train(self, saver=None, save_dir=None):
    #
    # keeping count of total timesteps of environment experience
    total_timesteps = 0
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
      trajectories = []
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
          # animate every 10 iterations
          if len(trajectories)==0 and (iteration%10==0) and self.animate:
            self.env.render()
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
      ##### Data Prep #####
      #
      # total timesteps is sum of all batch size
      total_timesteps += batch_size
      #
      # observations for this batch
      observations_batch = np.concatenate([trajectory["observations"] for trajectory in trajectories])
      next_observations_batch = np.roll(observations_batch,-1,axis=0)
      #
      # actions for this batch, reshapeing to handel 1D action space
      actions_batch = np.concatenate([trajectory["actions"] for trajectory in trajectories]).reshape([-1,self.action_shape])
      suggested_actions_batch = np.repeat(actions_batch.reshape(-1,1,self.action_shape),self.number_of_suggestions,axis=1)
      #
      # discounted returns for this batch. itertool used to make batch into long np array
      rewards_batch = np.concatenate([trajectory["rewards"] for trajectory in trajectories]).reshape([-1,1])
      #
      #
      dones_batch = np.concatenate([trajectory["dones"] for trajectory in trajectories])
      #
      # learning rate calculated from scheduler
      learning_rate = self.learning_rate_scheduler.value()
      #
      # adjusting the learning rate based on KL divergence
      if kl > 2e-3 * 2: 
        learning_rate /= 1.5
      elif kl < 2e-3 / 2: 
        learning_rate *= 1.5
      #
      ##### Optimization #####
      #
      # Taking the gradient step to optimize (train) the policy network (actor) and the value network (critic). mean and stddev is computed for kl divergence in the next step
      state_action_q, state_value, q_network_loss, value_network_loss, mean_policy_old, stddev_policy_old, _, _ = self.sess.run([self.q_action_taken, self.value_network.out, self.q_network_loss, self.value_network_loss, self.mean_policy, self.stddev_policy, self.train_q_network, self.train_value_network], {self.observations:observations_batch, self.next_observations:next_observations_batch, self.actions:actions_batch, self.suggested_actions:suggested_actions_batch, self.rewards:rewards_batch, self.done_mask:dones_batch, self.learning_rate:learning_rate})
      
      advantages_batch = np.squeeze(state_action_q - state_value.T).reshape([-1,1])
      #
      # Taking the gradient step to optimize (train) the policy network (actor) and the value network (critic). mean and stddev is computed for kl divergence in the next step
      policy_network_losses, policy_network_loss, _ = self.sess.run([self.policy_network_losses, self.policy_network_loss, self.train_policy_network], {self.observations:observations_batch, self.actions:actions_batch, self.advantages:advantages_batch, self.learning_rate:learning_rate})
      #
      # shape of the policy_network_losses is check since its common to have nonsense size due to unintentional matmul instead of non element wise multiplication. Action dimension and advantage dimension are very important and hard to debug when not correct
      if isinstance(policy_network_losses, list):
        policy_network_losses = policy_network_losses[0]
      assert policy_network_losses.shape==actions_batch.shape, "Dimensions mismatch. Policy Distribution is incorrect! " + str(policy_network_losses.shape)
      #
      # kl divergence computed
      kl = self.sess.run([self.kl], {self.observations:observations_batch, self.actions:actions_batch, self.mean_policy_old:mean_policy_old, self.stddev_policy_old:stddev_policy_old})[0]
      #
      # update
      if iteration%self.update_freq == 0:
        self.sess.run(self.update_target_fn)
      #
      ##### Reporting Performance #####
      #
      # average reward of past 100 episodes
      average_reward = np.mean(trajectory_rewards[-100:])
      #
      # update best average reward and save only if this is the best performing network so far
      if average_reward > best_average_reward:
        best_average_reward = average_reward
        saver.save(self.sess, save_dir)
      #
      # Printing performance progress and other useful infromation
      print("__________________________________________________________________________________________________________________________________________________________")
      print("{:>15} {:>10} {:>15} {:>15} {:>20} {:>20} {:>20} {:>20} {:>20}".format("total_timesteps", "iteration", "best_reward", "reward", "kl_divergence", "policy_loss", "value_loss", "q_network_loss", "average_advantage"))
      print("{:>15} {:>10} {:>15.2f} {:>15.2f} {:>20.5f} {:>20.2f} {:>20.2f} {:>20.2f} {:>20.2f}".format(total_timesteps, iteration, best_average_reward, average_reward, kl, policy_network_loss, value_network_loss, q_network_loss, np.mean(advantages_batch)))