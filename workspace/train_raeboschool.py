from OpenGL import GLU
import gym
import roboschool
import numpy as np
import tensorflow as tf
from algorithms.a2c import A2C
from datetime import datetime

env_name = "RoboschoolInvertedPendulum-v1"
env = gym.make(env_name)
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

with tf.Session() as sess:
	a2c = A2C(env, sess, animate=False)
	saver = tf.train.Saver()
	a2c.train(saver=saver, save_dir="/home/user/workspace/agents/"+env_name+".ckpt")
