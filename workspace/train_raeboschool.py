import gym
import roboschool
import numpy as np
import tensorflow as tf
from algorithms.a2c import A2C

env = gym.make("RoboschoolHopper-v1")
with tf.Session() as sess:
	a2c = A2C(env, sess)
	a2c.train()

