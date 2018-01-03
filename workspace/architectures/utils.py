import numpy as np 
import tensorflow as tf
import itertools

def dense(x, shape, name, initializer=tf.random_uniform_initializer(-1.0,1.0)):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], shape], initializer=initializer)
    b = tf.get_variable(name + "/b", [shape], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def get_network_param(network_size):
    network_param = None

    if network_size=='small':
        network_param = [32]

    elif network_size=='medium':
        network_param = [64, 64]

    elif network_size=='large':
        network_param = [128, 64]

    elif network_size=='xlarge':
        network_param = [256, 128]

    return network_param