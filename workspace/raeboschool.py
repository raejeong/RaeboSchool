#!/usr/bin/env python
from OpenGL import GLU
import gym
import roboschool
import numpy as np
import tensorflow as tf
from algorithms.a2s_keep_best import Agent
from datetime import datetime
import argparse
import os

#
# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Learning Continous Control for OpenAI Roboschool')
    parser.add_argument('--phase', dest='phase', default='train', type=str, help='train or test')
    args = parser.parse_args()
    return args

def train(id, env_name, seed, network_type, network_size, iterations, min_batch_size, lr, lr_schedule, gamma, animate, record):
	env = gym.make(env_name)
	tf.set_random_seed(seed)
	np.random.seed(seed)

	with tf.Session() as sess:
		agent = Agent(env, sess, network_type, network_size, iterations, min_batch_size, lr, lr_schedule, gamma, animate)
		saver = tf.train.Saver()
		save_dir = "/home/user/workspace/agents/"+env_name+id
		try:
			os.mkdir(save_dir)
		except:
			pass
		agent.train(saver=saver, save_dir=save_dir+"/"+env_name+id+".ckpt")
		env.close()

def test(id, env_name, seed, network_type, network_size, iterations, min_batch_size, lr, lr_schedule, gamma, animate, record):
	env = gym.make(env_name)
	save_dir = "/home/user/workspace/agents/"+env_name+id
	video_dir = "/home/user/workspace/videos/"+env_name+id
	if record:
	    env = gym.wrappers.Monitor(env,video_dir,force=True)

	with tf.Session() as sess:
	    agent = Agent(env, sess)
	    saver = tf.train.Saver()
	    saver.restore(sess, save_dir+"/"+env_name+id+".ckpt")
	    episode_reward = []
	    for i in range(5):
	        print("Episode " + str(i))
	        observation = env.reset()
	        done = False
	        episode_reward.append(0)
	        while not done:
	            if record:
	                a=env.render("rgb_array")
	            else:
	                env.render()
	            action = agent.compute_action(observation)
	            if not isinstance(action, (list, tuple, np.ndarray)):
	                action = np.array([action])
	            action = np.concatenate(action)
	            observation, reward, done, _ = env.step(action)
	            episode_reward[-1] += reward

	    print("Average Reward: %.2f" % np.mean(episode_reward))
	    env.close()

#
# Main code.
if __name__ == "__main__":
	args = getInputArgs()
	env_setting = dict(id="-0",
					   env_name='RoboschoolInvertedPendulum-v1',
					   seed=1,
					   network_type='fully_connected_network',
			           network_size='small',
			           iterations=30,
		               min_batch_size=2000,
		               lr=8e-3,
		               lr_schedule='linear',
		               gamma=0.99,
		               animate=False,
		               record=False)
	
	if args.phase == 'train':
		train(**env_setting)

	elif args.phase == 'test':
		test(**env_setting)

	else:
		print("ERROR: INVALID ARGUMENT Please choose train or test for phase argument")
