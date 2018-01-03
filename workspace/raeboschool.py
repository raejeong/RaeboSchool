#!/usr/bin/env python
from OpenGL import GLU
import gym
import roboschool
import numpy as np
import tensorflow as tf
from datetime import datetime
import argparse
import os
import importlib
from env_settings import env_settings
import time

# Parse the input arguments.
def getInputArgs():
    parser = argparse.ArgumentParser('Learning Continous Control for OpenAI Roboschool')
    parser.add_argument('--phase', dest='phase', default='train', type=str, help='train or test')
    parser.add_argument('--algorithm', dest='algorithm', default='A2S', type=str, help='algorithm name')
    args = parser.parse_args()
    return args

def train(agent_class, id, env_name, seed, record, data_collection_params, training_params, network_params, algorithm_params, logs_path):
    env = gym.make(env_name)
    tf.set_random_seed(seed)
    np.random.seed(seed)

    with tf.Session() as sess:
        agent = agent_class.Agent(env, sess, data_collection_params, training_params, network_params, algorithm_params, logs_path)
        saver = tf.train.Saver()
        save_dir = "/home/user/workspace/agents/"+env_name+id
        try:
            os.mkdir(save_dir)
        except:
            pass
        agent.train(saver=saver, save_dir=save_dir)
        env.close()

def test(agent_class, id, env_name, seed, record, data_collection_params, training_params, network_params, algorithm_params, logs_path):
    env = gym.make(env_name)
    save_dir = "/home/user/workspace/agents/"+env_name+id
    video_dir = "/home/user/workspace/videos/"+env_name+id
    if record:
        env = gym.wrappers.Monitor(env,video_dir,force=True,video_callable=lambda episode_id: True)
    with tf.Session() as sess:
        agent = agent_class.Agent(env, sess, data_collection_params, training_params, network_params, algorithm_params, logs_path)
        saver = tf.train.Saver()
        saver.restore(sess, save_dir+"/"+"A2S-Best.ckpt")
        agent.restore_networks()
        episode_reward = []
        for i in range(500):
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
            print(episode_reward[-1])
            # time.sleep(5)

        print("Max Reward: %.2f" % np.max(episode_reward))
        print(np.argmax(episode_reward))
        env.close()

#
# Main code.
if __name__ == "__main__":
    args = getInputArgs()
    env_setting = env_settings[args.algorithm]
    
    if args.phase == 'train':
        train(**env_setting)

    elif args.phase == 'test':
        test(**env_setting)

    else:
        print("ERROR: INVALID ARGUMENT Please choose train or test for phase argument")