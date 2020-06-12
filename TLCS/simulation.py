import traci
import numpy as np
import random
import timeit
import os

from memory import Memory, NormalMemory, SequenceMemory

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7



class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_actions = num_actions

         
    def _get_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        cumulative_waiting_time = 0
        
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                cumulative_waiting_time += wait_time
        return cumulative_waiting_time
    



    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        
        state = np.zeros(self._Model._state_shape)
        
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            
            #we can only see connected vehicles locations, so we ignore all regular vehicles
            if traci.vehicle.getTypeID(car_id) == "connected_vehicle":
            
                lane_pos = traci.vehicle.getLanePosition(car_id)
                lane_id = traci.vehicle.getLaneID(car_id)
                lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

                # distance in meters from the traffic light -> mapping into cells
                if lane_pos < 7:
                    lane_cell = 0
                elif lane_pos < 14:
                    lane_cell = 1
                elif lane_pos < 21:
                    lane_cell = 2
                elif lane_pos < 28:
                    lane_cell = 3
                elif lane_pos < 40:
                    lane_cell = 4
                elif lane_pos < 60:
                    lane_cell = 5
                elif lane_pos < 100:
                    lane_cell = 6
                elif lane_pos < 160:
                    lane_cell = 7
                elif lane_pos < 400:
                    lane_cell = 8
                elif lane_pos <= 750:
                    lane_cell = 9

                # finding the lane where the car is located 
                # x2TL_3 are the "turn left only" lanes
                if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                    lane_group = 0
                elif lane_id == "W2TL_3":
                    lane_group = 1
                elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                    lane_group = 2
                elif lane_id == "N2TL_3":
                    lane_group = 3
                elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                    lane_group = 4
                elif lane_id == "E2TL_3":
                    lane_group = 5
                elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                    lane_group = 6
                elif lane_id == "S2TL_3":
                    lane_group = 7
                else:
                    lane_group = -1   #currently crossing or driving away from intersection



                if lane_group >= 0:  #if car is a valid car (on approach, so not crossing intersection or driving away from it)
                    state[lane_cell][lane_group][0] = 1 #there is a car in the specified cell

        return state


    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length
        
    def _get_vehicle_delay(self):
        """
        Retrieve the cumulative delay of every vehicle currently in the simulation
        """
        total_delay = 0
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            #actual driving time = current time - departure time
            actual_driving_time = self._step - self._TrafficGen._generated_vehicles[int(car_id)][0] 
            #optimal driving time = distance driven / optimal speed on the road (13.89m/s)
            optimal_driving_time = traci.vehicle.getDistance(car_id) / 13.89
            
            delay = actual_driving_time - optimal_driving_time
            total_delay += delay 
            
        cum_delay = total_delay / len(car_list) if len(car_list) > 0 else 0
        return cum_delay




class TrainSimulation(Simulation):
    def __init__(self, Model, TargetModel, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_actions, training_epochs, copy_step):
        super().__init__(Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_actions)
        self._TargetModel = TargetModel
        self._Memory = Memory
        self._gamma = gamma
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._cumulative_delay_store = []
        self._training_epochs = training_epochs
        self._copy_step = copy_step
        self._scenario_index = -1 #dummy
        
        #order of scenarios: 4x constant (super low, undersaturated, saturated, oversaturated)
        # and 4x dynamic (2x saturated peak, 2x oversaturated peak)
        self._scenario_list = [0,1,2,3,4,4,5,5]  
     

    def _pick_next_scenario(self):
        self._scenario_index += 1
        #reset if it reaches the end of the scneario list
        if self._scenario_index >= len(self._scenario_list):
            self._scenario_index = 0
            
        return self._scenario_list[self._scenario_index]
        
        
    def _copy_online_into_target_model(self):
        """
        Copy weights from the online model into the targetModel
        """
        self._TargetModel._model.set_weights(self._Model._model.get_weights()) 


    
    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()

            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds
            self._sum_delay += self._get_vehicle_delay()

    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode
        self._cumulative_delay_store.append(self._sum_delay) #total seconds delay by all vehicles in this episode


    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
        
    @property
    def cumulative_delay_store(self):
        return self._cumulative_delay_store






class VanillaTrainSimulation(TrainSimulation):
    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session 
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        scenario_number = self._pick_next_scenario()
        self._TrafficGen.generate_routefile(episode, scenario_number)
        
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._sum_delay = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1
        
        
        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._get_waiting_times()
            reward = old_total_wait - current_total_wait

            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))
                    
            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()        
        simulation_time = round(timeit.default_timer() - start_time, 1)


        # start training after the full episode is done
        print("Training...")
        start_time = timeit.default_timer()
        train_iteration = 0
        
        #train for "training epochs" times --> only updates the online Model (target model stays unchanged)
        for _ in range(self._training_epochs):   #epoch = one forward pass and one backward pass of all the training examples, in the neural network terminology. 
            #start one training round
            self._replay()
            
            train_iteration += 1
            if train_iteration % self._copy_step == 0:
                self._copy_online_into_target_model()
            
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time
    
    
    
    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train 
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state, action), for every sample
            q_s_a_d = self._TargetModel.predict_batch(next_states)  # predict Q(next_state, action), for every sample

            # setup training arrays
            x = np.zeros((len(batch), ) + self._Model._state_shape) #from online network
            y = np.zeros((len(batch), self._num_actions))  #from target network

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                
                #update with combination of online and target network
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN
    
    
    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state


    
    
    
    
    
    
    
    
    
    
    
class RNNTrainSimulation(TrainSimulation):
    def __init__(self, Model, TargetModel, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_actions, training_epochs, copy_step, PredictModel):
        self._PredictModel = PredictModel
        super().__init__(Model, TargetModel, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_actions, training_epochs, copy_step)
    
    
    
    
    
    
    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session 
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        scenario_number = self._pick_next_scenario()
        self._TrafficGen.generate_routefile(episode, scenario_number)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._sum_delay = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1
        
        #prepare the model used for prediction (based on the previously trained model)
        self._update_and_reset_predict_model()
        
        
        

        while self._step < self._max_steps:
            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._get_waiting_times()
            reward = old_total_wait - current_total_wait


            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_to_buffer((old_state, old_action, reward, current_state))
                    

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        
        #now add collected sequence to memory (only if RNN/sequence memory is used)
        self._Memory.add_sequence()
        
        simulation_time = round(timeit.default_timer() - start_time, 1)


        # start training after the full episode is done
        print("Training...")
        start_time = timeit.default_timer()
        train_iteration = 0
        
        #train for "training epochs" times --> only updates the online Model (target model stays unchanged)
        for _ in range(self._training_epochs):   #epoch = one forward pass and one backward pass of all the training examples, in the neural network terminology. 
            #start one training round
            self._replay()
            
            train_iteration += 1
            if train_iteration % self._copy_step == 0:
                self._copy_online_into_target_model()
            
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time
    
    
    
    
    
    
    
    def _choose_action(self, state, epsilon):
        """
        Decide whether to perform an explorative or exploitative action, according to an epsilon-greedy policy (adjusted for recurrent network)
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            state = np.expand_dims(state, axis = 0)
            return np.argmax(self._PredictModel.predict_one(state)) # the best action given the current state
    
    
    def _update_and_reset_predict_model(self):
        """
        Copy weights from the online model into the predictModel and reset the hidden state of the LSTM
        """
        self._PredictModel._model.set_weights(self._Model._model.get_weights()) 
        self._PredictModel._model.reset_states()
 
    
    
    
    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train 
        """
        batch = self._Memory.get_samples(self._Model.batch_size)  #batch is a list of sequences of experiences (each item has length sequence_length)
        
        

        if len(batch) > 0:  # if the memory is full enough          
            states = []
            next_states = []
            
            for index_sequence, sequence in enumerate(batch):
                states.append( np.array([val[0] for val in sequence]) )
                next_states.append( np.array([val[3] for val in sequence]) )

            states = np.asarray(states)
            next_states = np.asarray(next_states)
            

            
            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._TargetModel.predict_batch(next_states)  # predict Q(next_state), for every sample

            
            
            # setup training arrays
            x = np.zeros((len(batch), self._Model._sequence_length ) + self._Model._state_shape) #from online network
            y = np.zeros((len(batch), self._Model._sequence_length, self._num_actions))  #from target network
            
            
            for index_sequence, sequence in enumerate(batch):
                for index_step, step in enumerate(sequence):
                    state, action, reward, _ = step[0], step[1], step[2], step[3]  # extract data from one sample
                    
                    current_q = q_s_a[index_sequence][index_step]  # get the Q(state) predicted before

                    
                    #update with combination of online and target network
                    current_q[action] = reward + self._gamma * np.amax(q_s_a_d[index_sequence][index_step])  # update Q(state, action)
                    x[index_sequence][index_step] = state
                    y[index_sequence][index_step] = current_q

            self._Model.train_batch(x, y)  # train the NN










class TestSimulation(Simulation):
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_actions, scenario_number):
        super().__init__(Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_actions)
        # self._reward_episode = []
        # self._queue_length_episode = []
        
        self._queue_length_all_episodes = []
        self._delay_all_episodes = []
        self._wait_all_episodes = []
        
        self._scenario_number = scenario_number #only single scenario is tested at once
        
        
    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo for "steps_todo" steps
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        # temp_delay_episode = []
        # temp_queue_length_episode = []
        # temp_wait_episode = []
        
        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            
            # #ADD KPI TO LIST
            # temp_delay_episode.append(self._get_vehicle_delay())
            # temp_queue_length_episode.append(self._get_queue_length())
            # temp_wait_episode.append(self._collect_waiting_times())
            
            #ADD KPI TO LIST
            self._delay_episode.append(self._get_vehicle_delay())
            self._queue_length_episode.append(self._get_queue_length())
            self._wait_episode.append(self._get_waiting_times())
            
            # queue_length = self._get_queue_length() 
            # self._queue_length_episode.append(queue_length)
        # return temp_delay_episode, temp_queue_length_episode, temp_wait_episode
            
            
            
            


    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(episode, self._scenario_number)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # INITS
        self._step = 0
        # self._waiting_times = {}
        # old_total_wait = 0
        old_action = -1 # dummy init
        
        #list for the data for just this one episode. Reset for every new tested episode
        self._delay_episode = []
        self._queue_length_episode = []
        self._wait_episode = []

        while self._step < self._max_steps:
            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            # current_total_wait = self._collect_waiting_times()
            #reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)
                

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            # old_total_wait = current_total_wait

            # self._reward_episode.append(reward)

        
        #when episode is over, add the whole list with epsode stats to the full list of all episodes
        self._delay_all_episodes.append(self._delay_episode)
        self._queue_length_all_episodes.append(self._queue_length_episode)
        self._wait_all_episodes.append(self._wait_episode)
        
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time

    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        #expand dimension if it is a recurrent model (requires number of time steps, here = 1)
        if len(self._Model._model.layers[0].input.shape) > len(self._Model._state_shape)+1:
            state = np.expand_dims(state, axis = 0)
        
        return np.argmax(self._Model.predict_one(state)) # the best action given the current state
    
    


    @property
    def delay_all_episodes(self):
        return self._delay_all_episodes
        
    @property
    def queue_length_all_episodes(self):
        return self._queue_length_all_episodes
        
    @property
    def wait_all_episodes(self):
        return self._wait_all_episodes
