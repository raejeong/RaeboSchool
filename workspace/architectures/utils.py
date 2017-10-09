import numpy as np 
import tensorflow as tf
import itertools

def dense(x, shape, name):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], shape], initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable(name + "/b", [shape], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)
