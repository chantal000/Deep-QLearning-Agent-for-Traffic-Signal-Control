from __future__ import absolute_import
from __future__ import print_function

import os
os.environ["SUMO_HOME"] = "C:\Program Files (x86)\Sumo18"
from shutil import copyfile
import datetime

from simulation import Simulation, TestSimulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path


if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'], config['scenario_number'])

    
    #SET STATE DIMENSION PARAMETERS  
    number_of_cells_per_lane = 10
    conv_state_shape = (number_of_cells_per_lane, 8, 2)
    green_phase_state_shape = 4
    elapsed_time_state_shape = 1
    state_shape = [conv_state_shape, green_phase_state_shape, elapsed_time_state_shape]
    
    
    
    Model = TestModel(
        model_path=model_path,
        state_shape=state_shape
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['penetration_rate']
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
    
    #make non-recurrent model
    if Model._recurrent == False:
        Simulation = VanillaTestSimulation(
            Model,
            TrafficGen,
            sumo_cmd,
            config['max_steps'],
            config['green_duration'],
            config['yellow_duration'],
            config['num_actions'],
            config['scenario_number']
        )
        # print("make vanilla simulation")
    #recurrent model
    else:
        Simulation = RNNTestSimulation(
            Model,
            TrafficGen,
            sumo_cmd,
            config['max_steps'],
            config['green_duration'],
            config['yellow_duration'],
            config['num_actions'],
            config['scenario_number']
        )
        # print("make recurrent simulation")
    



    
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

    
    Visualization.testing_save_data_and_plot(data=Simulation.delay_all_episodes, filename='average_delay', xlabel='Simulation step', ylabel='Average vehicle delay [s]')
    Visualization.testing_save_data_and_plot(data=Simulation.CV_delay_all_episodes, filename='average_CV_delay', xlabel='Simulation step', ylabel='Average connected vehicle delay [s]')
    Visualization.testing_save_data_and_plot(data=Simulation.RV_delay_all_episodes, filename='average_RV_delay', xlabel='Simulation step', ylabel='Average regular vehicle delay [s]')
    Visualization.testing_save_data_and_plot(data=Simulation.queue_length_all_episodes, filename='queue_length', xlabel='Simulation step', ylabel='Cumulative queue length [vehicles]')
    Visualization.testing_save_data_and_plot(data=Simulation.wait_all_episodes, filename='cumulative_wait', xlabel='Simulation step', ylabel='Average waiting time [s]')
