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

env_settings = {
    'A2C':dict(agent_class=importlib.import_module('algorithms.A2C'),
               id="-5",
               env_name='RoboschoolHopper-v1',
               seed=2,
               record=False,
               data_collection_params = {'min_batch_size':1000,
                                         'min_episodes':1, 
                                         'episode_adapt_rate':3},
               training_params = {'total_timesteps':1000000, 
                                  'learning_rate':1e-3, 
                                  'adaptive_lr':True, 
                                  'desired_kl':2e-3},
               rl_params = {'gamma':0.99, 
                            'num_policy_update':1},
               network_params = {'value_network':['fully_connected_network','medium'], 
                                 'policy_network':['fully_connected_network','medium']},
               algorithm_params = {'restore': True,
                                   'std_dev':['fixed', 0.2]},
               logs_path="/home/user/workspace/logs/",
               ),
    'A2S':dict(agent_class=importlib.import_module('algorithms.A2S'),
               id="-5",
               env_name='RoboschoolHopper-v1',
               seed=0,
               record=False,
               data_collection_params = {'min_batch_size':1000,
                                         'min_episodes':3, 
                                         'episode_adapt_rate':3},
               training_params = {'total_timesteps':1000000,  
                                  'adaptive_lr':True, 
                                  'desired_kl':2e-3},
               network_params = {'q_network':['fully_connected_network','medium'], 
                                 'value_network':['fully_connected_network','medium'], 
                                 'policy_network':['fully_connected_network','medium']},
               algorithm_params = {'gamma':0.99, 
                                   'learning_rate':1e-3,
                                   'number_of_suggestions':4, 
                                   'q_target_estimate_iteration':10,
                                   'std_dev':['fixed', 0.3], 
                                   'PER_size':200000, 
                                   'PER_batch_size':32, 
                                   'PER_iterations':2,
                                   'PER_alpha':0.6, 
                                   'PER_epsilon':0.01,
                                   'target_update_rate':0.001},
               logs_path="/home/user/workspace/logs/"
               )
}

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
        agent.train(saver=saver, save_dir=save_dir+"/"+env_name+id+".ckpt")
        env.close()

def test(agent_class, id, env_name, seed, record, data_collection_params, training_params, network_params, algorithm_params, logs_path):
    env = gym.make(env_name)
    save_dir = "/home/user/workspace/agents/"+env_name+id
    video_dir = "/home/user/workspace/videos/"+env_name+id
    if record:
        env = gym.wrappers.Monitor(env,video_dir,force=True)

    with tf.Session() as sess:
        agent = agent_class.Agent(env, sess, data_collection_params, training_params, network_params, algorithm_params, logs_path)
        saver = tf.train.Saver()
        saver.restore(sess, save_dir+"/"+env_name+id+".ckpt")
        agent.networks_restore()
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
                action = agent.compute_action(observation, epsilon=1.0)
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
    env_setting = env_settings[args.algorithm]
    
    if args.phase == 'train':
        train(**env_setting)

    elif args.phase == 'test':
        test(**env_setting)

    else:
        print("ERROR: INVALID ARGUMENT Please choose train or test for phase argument")