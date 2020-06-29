from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile


from utils import import_test_configuration, set_sumo, set_test_path, set_train_path
from fixed_time import *

from generator import TrafficGenerator
from visualization import Visualization






if __name__ == "__main__":
    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
#     model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])
    plot_path = set_train_path(config['models_path_name'])
    
    
    #PARAMETERS
    fixed_green_time = 30
    
    
    
    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['penetration_rate']
    )
    
    
    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
    
    
    Simulation = FixedTimeTestSimulation(
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        fixed_green_time,
        config['yellow_duration'],
        config['num_actions'],
        config['scenario_number']
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    

    
    while episode < config['total_episodes']:
        print('\n----- Test Episode', str(episode+1), 'of', str(config['total_episodes']))
        
        #run simulation + train for one episode at a time
        simulation_time = Simulation.run(episode * 10000)  # run the simulation (with a guaranteed different seed than in training)
        print('Simulation time:', simulation_time, 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    
    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))


    
    Visualization.testing_save_data_and_plot_fixed(data=Simulation.delay_all_episodes, filename='average_delay', xlabel='Simulation step', ylabel='Average vehicle delay [s]')
    Visualization.testing_save_data_and_plot_fixed(data=Simulation.CV_delay_all_episodes, filename='average_CV_delay', xlabel='Simulation step', ylabel='Average connected vehicle delay [s]')
    Visualization.testing_save_data_and_plot_fixed(data=Simulation.RV_delay_all_episodes, filename='average_RV_delay', xlabel='Simulation step', ylabel='Average regular vehicle delay [s]')
    Visualization.testing_save_data_and_plot_fixed(data=Simulation.queue_length_all_episodes, filename='queue_length', xlabel='Simulation step', ylabel='Cumulative queue length [vehicles]')
    Visualization.testing_save_data_and_plot_fixed(data=Simulation.wait_all_episodes, filename='cumulative_wait', xlabel='Simulation step', ylabel='Average waiting time [s]')
