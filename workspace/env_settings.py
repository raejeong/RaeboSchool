import importlib

env_settings = {
    'REINFORCE':dict(agent_class=importlib.import_module('algorithms.REINFORCE'),
                     id="-5",
                     env_name='RoboschoolHopper-v1',
                     seed=2,
                     record=False,
                     data_collection_params = {'min_batch_size':500,
                                               'min_episodes':3, 
                                               'episode_adapt_rate':3},
                     training_params = {'total_timesteps':1000000,  
                                        'adaptive_lr':True, 
                                        'desired_kl':2e-3},
                     network_params = {'policy_network':['fully_connected_network','medium']},
                     algorithm_params = {'gamma':0.99, 
                                         'learning_rate':1e-3,
                                         'number_of_suggestions':0,
                                         'std_dev':['fixed', 0.5], 
                                         'target_update_rate':1.0},
                     logs_path="/home/user/workspace/logs/"
                     ),
    'QVAC':dict(agent_class=importlib.import_module('algorithms.QVAC'),
                id="-5",
                env_name='RoboschoolHopper-v1',
                seed=2,
                record=False,
                data_collection_params = {'min_batch_size':500,
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
                                   'number_of_suggestions':0,
                                   'std_dev':['fixed', 0.5], 
                                   'target_update_rate':1.0},
                logs_path="/home/user/workspace/logs/"
                ),
    'A2C':dict(agent_class=importlib.import_module('algorithms.A2C'),
               id="-5",
               env_name='RoboschoolHopper-v1',
               seed=2,
               record=False,
               data_collection_params = {'min_batch_size':500,
                                         'min_episodes':2, 
                                         'episode_adapt_rate':3},
               training_params = {'total_timesteps':1000000,  
                                  'adaptive_lr':True, 
                                  'desired_kl':2e-3},
               network_params = {'value_network':['fully_connected_network','medium'], 
                                 'policy_network':['fully_connected_network','medium']},
               algorithm_params = {'gamma':0.99, 
                                   'learning_rate':1e-3,
                                   'number_of_suggestions':0,
                                   'std_dev':['fixed', 0.5], 
                                   'target_update_rate':1.0},
               logs_path="/home/user/workspace/logs/"
               ),
    'A2S':dict(agent_class=importlib.import_module('algorithms.A2S'),
               id="-5",
               env_name='RoboschoolHopper-v1',
               seed=2,
               record=False,
               data_collection_params = {'min_batch_size':300,
                                         'min_episodes':3, 
                                         'episode_adapt_rate':1},
               training_params = {'total_timesteps':1000000,  
                                  'adaptive_lr':True, 
                                  'desired_kl':2e-3},
               network_params = {'q_network':['fully_connected_network','medium'], 
                                 'value_network':['fully_connected_network','medium'], 
                                 'policy_network':['fully_connected_network','medium']},
               algorithm_params = {'gamma':0.99, 
                                   'learning_rate':1e-3,
                                   'number_of_suggestions':8, 
                                   'q_target_estimate_iteration':3,
                                   'std_dev':['fixed', 0.5], 
                                   'PER_size':100000, 
                                   'PER_batch_size':32, 
                                   'PER_iterations':30,
                                   'PER_alpha':0.6, 
                                   'PER_epsilon':0.01,
                                   'target_update_rate':1.0},
               logs_path="/home/user/workspace/logs/"
               )
}
