import numpy as np 
import tensorflow as tf
from architectures.utils import *
import itertools

class Network:
	def __init__(self, sess, x, out_shape, name, network_size, keep_prob=1.0):
		network_param = get_network_param(network_size)
		with tf.variable_scope(name):
			out = x
			for i in itertools.count():
				if i == len(network_param):
					break
				out = dense(out, network_param[i], name+"/fc"+str(i))
				out = tf.contrib.layers.layer_norm(out)
				out = tf.tanh(out)
				# out = lrelu(out)
			out = dense(out, out_shape, name+"/out", initializer=tf.random_uniform_initializer(-0.1,0.1))
		self.out = out