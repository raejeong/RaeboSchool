import numpy as np 
import tensorflow as tf
from architectures.utils import *

class FullyConnectedNetwork:
	def __init__(self, sess, x, out_shape, name, network_param):
		with tf.variable_scope(name):
			out = x
			for i in itertools.count():
				if i == len(network_param):
					break
				out = lrelu(dense(out, network_param[i], "fc"+str(i)))
			out = dense(out, out_shape, "out")

		self.out = out