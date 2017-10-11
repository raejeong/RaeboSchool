import numpy as np 
import tensorflow as tf
from architectures.utils import *
import itertools

class FullyConnectedNetwork:
	def __init__(self, sess, x, out_shape, name, network_param):
		with tf.variable_scope(name):
			out = x
			for i in itertools.count():
				if i == len(network_param):
					break
				# out = tf.contrib.layers.layer_norm(out)
				out = lrelu(dense(out, network_param[i], "fc"+str(i)))
			out = dense(out, out_shape, "out", initializer=tf.random_uniform_initializer(-0.1,0.1))

		self.out = out