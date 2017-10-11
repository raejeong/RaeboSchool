import numpy as np 
import tensorflow as tf 
import importlib
from algorithms.utils import *
import itertools

class A2C:
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
    with tf.variable_scope("A2C"):
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
      # placeholder for the r + gamma*next_value
      self.target_values = tf.placeholder(tf.float32, shape=[None, 1], name="target_values")
      #
      # placeholder for learning rate for the optimizer
      self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
      #
      # mean and the stddev of the old policy distrubution for KL calculation
      self.mean_policy_old = tf.placeholder(tf.float32, name="mean_policy_old")
      self.stddev_policy_old = tf.placeholder(tf.float32, name="stddev_policy_old")
      #
      ##### Networks #####
      #
      # policy network, outputs the mean and the stddev (needs to go through softplus) for the action
      self.policy_network = self.network_class.Network(sess, self.observations, self.action_shape*2, "policy_network", network_size)
      #
      # value network, outputs value of observation, used for advantage estimate
      self.value_network = self.network_class.Network(sess, self.observations, 1, "value_network", network_size)
      #
      ##### Policy Action Probability #####
      #
      # isolating the mean outputted by the policy network
      self.mean_policy = tf.reshape(tf.squeeze(self.policy_network.out[:,:self.action_shape]),[-1, self.action_shape])
      #
      # isolating the stddev outputted by the policy network, softplus is used to make sure that the stddev is positive
      self.stddev_policy = tf.reshape((tf.nn.softplus(tf.squeeze(self.policy_network.out[:,self.action_shape:])) + 1e-5),[-1, self.action_shape])
      #
      # gaussian distribution is built with mean and stddev from the policy network
      self.gaussian_policy_distribution = tf.contrib.distributions.Normal(self.mean_policy, self.stddev_policy)
      #
      # gaussian distribution is built with mean and stddev from the old policy network
      self.gaussian_policy_distribution_old = tf.contrib.distributions.Normal(self.mean_policy_old, self.stddev_policy_old)
      #
      # KL divergence from old policy distribution to the new policy distribution
      self.kl = tf.reduce_mean(tf.contrib.distributions.kl_divergence(self.gaussian_policy_distribution, self.gaussian_policy_distribution_old))
      #
      # action sampled from the gaussian distribution of the policy network
      self.action_sampled = self.gaussian_policy_distribution._sample_n(1)
      #
      ##### Loss #####
      #
      # loss for the policy network and value network
      self.negative_log_prob = -self.gaussian_policy_distribution.log_prob(self.actions)
      self.policy_network_losses = self.negative_log_prob*self.advantages # be careful with this operation, it should be element wise not matmul!
      self.policy_network_loss = tf.reduce_mean(self.policy_network_losses)
      self.value_network_loss = tf.reduce_mean(tf.squared_difference(self.value_network.out, self.target_values))
      #
      ##### Optimization #####
      #
      # optimizer for the policy network and value network
      self.policy_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      self.value_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate*10)
      #
      # training operation for the policy network and value network
      self.train_policy_network = self.policy_network_optimizer.minimize(self.policy_network_loss)
      self.train_value_network = self.value_network_optimizer.minimize(self.value_network_loss)
    #
    # initialize all tf variables
    self.sess.run(tf.global_variables_initializer())

  # samples action from the current policy network
  def compute_action(self, observation):
    return self.sess.run(self.action_sampled, {self.observations: observation[None]})[0]

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
      trajectories, returns, advantages = [], [], []
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
        observations, actions, rewards = [], [], []
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
        trajectory = {"observations":np.array(observations), "actions":np.array(actions), "rewards":np.array(rewards)}
        trajectories.append(trajectory)
        #
        # computing the discounted return for this episode (NOT A SINGLE NUMBER, FOR EACH OBSERVATION)
        return_ = discount(trajectory["rewards"], self.gamma)
        #
        # compute the value estimates for the observations seen during this episode
        values = self.compute_value(observations)
        #
        # computing the advantage estimate
        advantage = return_ - np.concatenate(values)
        returns.append(return_)
        advantages.append(advantage)
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
      # advantages for this batch. itertool used to make batch into long np array
      advantages_batch = np.array(list(itertools.chain.from_iterable(advantages))).flatten().reshape([-1,1])
      #
      # discounted returns for this batch. itertool used to make batch into long np array
      returns_batch = np.array(list(itertools.chain.from_iterable(returns))).reshape([-1,1])
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
      policy_network_losses, policy_network_loss, value_network_loss, mean_policy_old, stddev_policy_old, _, _ = self.sess.run([self.policy_network_losses, self.policy_network_loss, self.value_network_loss, self.mean_policy, self.stddev_policy, self.train_policy_network, self.train_value_network], {self.observations:observations_batch, self.actions:actions_batch, self.advantages:advantages_batch, self.target_values:returns_batch, self.learning_rate:learning_rate})
      #
      # shape of the policy_network_losses is check since its common to have nonsense size due to unintentional matmul instead of non element wise multiplication. Action dimension and advantage dimension are very important and hard to debug when not correct
      if isinstance(policy_network_losses, list):
        policy_network_losses = policy_network_losses[0]
      assert policy_network_losses.shape==actions_batch.shape, "Dimensions mismatch. Policy Distribution is incorrect! " + str(policy_network_losses.shape)
      #
      # kl divergence computed
      kl = self.sess.run([self.kl], {self.observations:observations_batch, self.actions:actions_batch, self.mean_policy_old:mean_policy_old, self.stddev_policy_old:stddev_policy_old})[0]
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
      print("________________________________________________________________________________________________________________________________________")
      print("{:>15} {:>10} {:>15} {:>15} {:>20} {:>20} {:>20}".format("total_timesteps", "iteration", "best_reward", "reward", "kl_divergence", "policy_loss", "value_loss"))
      print("{:>15} {:>10} {:>15.2f} {:>15.2f} {:>20.5f} {:>20.2f} {:>20.2f}".format(total_timesteps, iteration, best_average_reward, average_reward, kl, policy_network_loss, value_network_loss))