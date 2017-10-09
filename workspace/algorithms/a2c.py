import numpy as np 
import tensorflow as tf 
from architectures import fully_connected_network
from algorithms.utils import *
import itertools

class A2C:

  def __init__(self,
               env,
               sess,
               policy_network_param=[32,16],
               value_network_param=[32,16],
               iterations=300,
               min_batch_size=1000,
               lr=1e-3,
               lr_schedule='linear',
               gamma=0.99):

    self.sess = sess
    self.env = env
    self.learning_rate_scheduler = Scheduler(v=lr, nvalues=iterations, schedule=lr_schedule)
    self.iterations = iterations
    self.min_batch_size = min_batch_size
    self.gamma = gamma
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
      #
      self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
      #
      ##### Networks #####
      #
      # policy network, outputs the mean and the stddev (needs to go through softplus) for the action
      self.policy_network = fully_connected_network.FullyConnectedNetwork(sess, self.observations, self.action_shape*2, "policy_network", policy_network_param)
      #
      # value network, outputs value of observation, used for advantage estimate
      self.value_network = fully_connected_network.FullyConnectedNetwork(sess, self.observations, 1, "value_network", value_network_param)
      #
      ##### Policy Action Probability #####
      #
      # isolating the mean outputted by the policy network
      self.mean_policy = tf.squeeze(self.policy_network.out[:,:self.action_shape])
      #
      # isolating the stddev outputted by the policy network, softplus is used to make sure that the stddev is positive
      self.stddev_policy = tf.nn.softplus(tf.squeeze(self.policy_network.out[:,self.action_shape:])) + 1e-5
      #
      # gaussian distribution is built with mean and stddev from the policy network
      self.gaussian_policy_distribution = tf.contrib.distributions.Normal(self.mean_policy, self.stddev_policy)
      #
      # action sampled from the gaussian distribution of the policy network
      self.action_sampled = self.gaussian_policy_distribution._sample_n(1)
      #
      # action sampled is cliped to environment action bounds
      self.action_sampled = tf.clip_by_value(self.action_sampled, self.env.action_space.low[0], self.env.action_space.high[0])
      #
      ##### Loss #####
      #
      # loss for the policy network and value network
      self.policy_network_loss = tf.multiply(-self.gaussian_policy_distribution.log_prob(self.actions), self.advantages)
      self.value_network_loss = tf.squared_difference(self.value_network.out, self.target_values)
      #
      ##### Optimization #####
      #
      # optimizer for the policy network and value network
      self.policy_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      self.value_network_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      #
      # training operation for the policy network and value network
      self.train_policy_network = self.policy_network_optimizer.minimize(self.policy_network_loss)
      self.train_value_network = self.value_network_optimizer.minimize(self.value_network_loss)

    self.sess.run(tf.global_variables_initializer())

  def compute_action(self, observation):
    return self.sess.run(self.action_sampled, {self.observations: observation[None]})[0]

  def compute_value(self, observations):
    return self.sess.run(self.value_network.out, {self.observations: observations})


  def train(self):
    total_timesteps = 0

    for iteration in range(self.iterations):
      batch_size = 0
      trajectories, returns, advantages = [], [], []
      while batch_size < self.min_batch_size:
        observation = self.env.reset()
        observations, actions, rewards = [], [], []
        done = False
        while not done:
          if len(trajectories)==0 and (iteration%10==0):
            self.env.render()
          observations.append(observation)
          action = self.compute_action(observation)
          observation, reward, done, _ = self.env.step(action)
          rewards.append(reward)
          actions.append(action)
        batch_size += len(rewards)
        trajectory = {"observations":np.array(observations), "actions":np.array(actions), "rewards":np.array(rewards)}
        trajectories.append(trajectory)
        return_ = discount(trajectory["rewards"], self.gamma)
        values = self.compute_value(trajectory["observations"])
        advantage = return_ - np.concatenate(values)
        returns.append(return_)
        advantages.append(advantage)
      total_timesteps += batch_size
      observations_batch = np.concatenate([trajectory["observations"] for trajectory in trajectories])
      actions_batch = np.concatenate([trajectory["actions"] for trajectory in trajectories])
      advantages_batch = np.array(list(itertools.chain.from_iterable(advantages))).flatten().reshape([-1,1])
      returns_batch = np.array(list(itertools.chain.from_iterable(returns))).reshape([-1,1])
      learning_rate = self.learning_rate_scheduler.value()
      policy_network_loss, value_network_loss, _, _ = self.sess.run([self.policy_network_loss, self.value_network_loss, self.train_policy_network, self.train_value_network], {self.observations:observations_batch, self.actions:actions_batch, self.advantages:advantages_batch, self.target_values:returns_batch, self.learning_rate:learning_rate})
      print("iteration " + str(iteration) + ": Average Batch Return: %.2f" % np.mean(returns_batch))
