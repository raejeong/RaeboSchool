from OpenGL import GLU
import gym
import roboschool
import numpy as np
import tensorflow as tf
from algorithms.a2c import A2C
from datetime import datetime

env_name = "RoboschoolInvertedPendulum-v1"
env = gym.make(env_name)

with tf.Session() as sess:
    a2c = A2C(env, sess)
    saver = tf.train.Saver()
    saver.restore(sess, "/home/user/workspace/agents/"+env_name+".ckpt")
    episode_reward = []
    for i in range(100):
        print("Episode " + str(i))
        observation = env.reset()
        done = False
        episode_reward.append(0)
        while not done:
            env.render()
            action = a2c.compute_action(observation)
            if not isinstance(action, (list, tuple, np.ndarray)):
                action = np.array([action])
            observation, reward, done, _ = env.step(action)
            episode_reward[-1] += reward

    print("Average Reward: %.2f" % np.mean(episode_reward))
