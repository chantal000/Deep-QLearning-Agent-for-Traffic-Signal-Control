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
        
        self._current_phase = -1 #dummy
        self._elapsed_time_since_phase_start = 0
 
        
        self._max_cars_per_lane_cell = [1,1,1,1,2,3,6,8,32,47] #how many cars fit into each lane cell
        

         
    def _get_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        cumulative_waiting_time = 0
   
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                cumulative_waiting_time += wait_time
        return cumulative_waiting_time
    
    
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

        return total_delay


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
        

    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        
        #initialize state arrays
        conv_state = np.zeros(self._Model._state_shape[0])
        green_phase_state = np.zeros(self._Model._state_shape[1])
        green_phase_state[self._current_phase] = 1  #one-hot encoded current green phase
        elapsed_time_state = self._elapsed_time_since_phase_start
        

        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            
            #we can only see connected vehicles locations, so we ignore all regular vehicles
            if traci.vehicle.getTypeID(car_id) == "connected_vehicle":
            
                lane_pos = traci.vehicle.getLanePosition(car_id)
                lane_id = traci.vehicle.getLaneID(car_id)
                lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

                speed = traci.vehicle.getSpeed(car_id)

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
                    conv_state[lane_cell][lane_group][0] += 1 #add one car to the list of cars in this cell (for now it keeps track of the absolute number of cars in this cell)
                    conv_state[lane_cell][lane_group][1] += speed #add all speeds of vehicles in this cell together

        
        #loop over every single cell
        for lane_cell in range(10):
            for lane_group in range(8):
                number_cars_in_cell = conv_state[lane_cell][lane_group][0]
                
                #only calculate stuff if there is a car in the cell, else the cell remains 0
                if number_cars_in_cell > 0:
              
                    #for the three sraight/right turning lanes, it is a triple lane. The left turning lane is singular.
                    if lane_group == 0 or lane_group == 2 or lane_group == 4 or lane_group == 6:
                        lane_multiplier = 3
                    else:
                        lane_multiplier = 1
                
                    #averaged normalized cell speed: divide summed speed in each cell by the number of cars in that cell and then normalize with the allowed max_speed
                    conv_state[lane_cell][lane_group][1] = (conv_state[lane_cell][lane_group][1] / number_cars_in_cell ) / 13.89
                
                    #calculate the cell occupancy/density per cell: #cars / #max cars (take into account lane multiplier!!)
                    conv_state[lane_cell][lane_group][0] = number_cars_in_cell / (self._max_cars_per_lane_cell[lane_cell] * lane_multiplier) 

        #encode the current traffic phase in the conv image (the lanes which currently have green will get a + sign, the others a - sign)
        # i made all conv fields negative until now (see - sign above) and now to make it positive i have to multiply it with -1 again
        if self._current_phase == 0:
            for lane_cell in range(10):
                conv_state[lane_cell][2] =  - conv_state[lane_cell][2]
                conv_state[lane_cell][6] =  - conv_state[lane_cell][6]
        if self._current_phase == 1:
            for lane_cell in range(10):
                conv_state[lane_cell][3] =  - conv_state[lane_cell][3]
                conv_state[lane_cell][7] =  - conv_state[lane_cell][7]
        if self._current_phase == 2:
            for lane_cell in range(10):
                conv_state[lane_cell][0] =  - conv_state[lane_cell][0]
                conv_state[lane_cell][4] =  - conv_state[lane_cell][4]
        if self._current_phase == 3:
            for lane_cell in range(10):
                conv_state[lane_cell][1] =  - conv_state[lane_cell][1]
                conv_state[lane_cell][5] =  - conv_state[lane_cell][5]
         
        return [conv_state, green_phase_state, elapsed_time_state]


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
        
        # reset elapsed green time and current phase
        self._current_phase = action_number
        self._elapsed_time_since_phase_start = 0


    
        
    




class TrainSimulation(Simulation):
    def __init__(self, Model, TargetModel, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_actions, training_epochs, copy_step):
        super().__init__(Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_actions)
        self._TargetModel = TargetModel
        self._Memory = Memory
        self._gamma = gamma
        self._reward_store = []
        
        self._train_iteration = 0
        
        
        self._training_epochs = training_epochs
        self._copy_step = copy_step
        self._scenario_index = -1 #dummy
        
        #order of scenarios: 3x constant (super low, undersaturated, saturated)
        # and 3x dynamic saturated peak
        self._scenario_list = [3,0,3,1,3,2]
        
        #list to test scenarios
        self.testing_reward_store = []
        
     

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
            self._elapsed_time_since_phase_start +=1 #update the elapsed green time counter
            steps_todo -= 1
            

    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        # self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        # self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode
        # self._cumulative_delay_store.append(self._sum_delay) #total seconds delay by all vehicles in this episode

    def _save_greedy_episode_stats(self):
        """
        Save the stats of the greedy episodes to plot the graphs at the end of the session
        """
        self.testing_reward_store.append(self._greedy_results_list)  # how much negative reward in this episode


        




    @property
    def reward_store(self):
        return self._reward_store

    # @property
    # def cumulative_wait_store(self):
        # return self._cumulative_wait_store

    # @property
    # def avg_queue_length_store(self):
        # return self._avg_queue_length_store
        
    # @property
    # def cumulative_delay_store(self):
        # return self._cumulative_delay_store






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
        # self._sum_queue_length = 0
        # self._sum_waiting_time = 0
        # self._sum_delay = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1
        self._current_phase = -1 #dummy
        self._elapsed_time_since_phase_start = 0
        
        
        
        while self._step < self._max_steps:

            # get current state of the intersection (shape: conv, current green phase, elapsed time since beginning green phase)
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative delay between actions)
            # delay time = seconds delay accumulated for all vehicles in incoming lanes
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

    
        #TRAINING
        # start training after the full episode is done
        print("Training...")
        start_time = timeit.default_timer()
        
        #train for "training epochs" times --> only updates the online Model (target model stays unchanged)
        for _ in range(self._training_epochs):   #epoch = one forward pass and one backward pass of all the training examples, in the neural network terminology. 
            #start one training round
            self._replay()
            
            self._train_iteration += 1
            if self._train_iteration % self._copy_step == 0:
                self._copy_online_into_target_model()
                
        training_time = round(timeit.default_timer() - start_time, 1)        
    
    
    
        #TESTING EPSIODES (PURE GREEDY) EVERY X EPISODES
        update_every_x_episodes = 15    
        if episode % update_every_x_episodes == 0:
            print("Testing ...")
            self._greedy_run()


        return simulation_time, training_time
    
    
    
    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train 
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # print("states shape: ", states.shape)
            # print("inner states shape: shape 0 ", states[0][0].shape, "shape 1", states[0][1].shape, "shape 2")
            
            # prediction           
            q_s_a = self._Model.predict_batch(states)  # predict Q(state, action), for every sample
            q_s_a_d = self._TargetModel.predict_batch(next_states)  # predict Q(next_state, action), for every sample

            # setup training arrays
            x_conv = np.zeros((len(batch), ) + self._Model._state_shape[0]) #from online network
            x_phase = np.zeros((len(batch), ) + (self._Model._state_shape[1], )) #from online network
            x_elapsed = np.zeros((len(batch), ) + (self._Model._state_shape[2], )) #from online network
            y = np.zeros((len(batch), self._num_actions))  #from target network

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                
                #update with combination of online and target network
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x_conv[i] = state[0]
                # x_phase[i] = state[1]
                x_elapsed[i] = state[2]
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch([x_conv, x_elapsed], y)  # train the NN
    
    
    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            # prediction = np.argmax(self._Model.predict_one(state))
            # print("predicted action: ", prediction)
            # return prediction
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state
            
    def _choose_greedy_action(self, state):
        """
        Do an exploitative action
        """
        return np.argmax(self._Model.predict_one(state)) # the best action given the current state


    def _greedy_run(self):
    
        self._greedy_results_list = []
    
        for scenario_number in range(4):
            self._TrafficGen.generate_routefile(episode*2000, scenario_number)
            traci.start(self._sumo_cmd)

            # inits
            self._step = 0
            self._sum_neg_reward = 0

            old_total_wait = 0
            old_state = -1
            old_action = -1
            self._current_phase = -1 #dummy
            self._elapsed_time_since_phase_start = 0
            
            
            
            while self._step < self._max_steps:
                # get current state of the intersection (shape: conv, current green phase, elapsed time since beginning green phase)
                current_state = self._get_state()

                # calculate reward of previous action: (change in cumulative delay between actions)
                # delay time = seconds delay accumulated for all vehicles in incoming lanes
                current_total_wait = self._get_waiting_times()
                reward = old_total_wait - current_total_wait

                # saving the data into the memory
                if self._step != 0:
                    self._Memory.add_sample((old_state, old_action, reward, current_state))
                        
                # choose the light phase to activate, based on the current state of the intersection
                action = self._choose_greedy_action(current_state)

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
                        
            self._greedy_results_list.append(self._sum_neg_reward)

            traci.close()   
        self._save_greedy_episode_stats()
    
    
    

    
    
    
    
    
    
    
    
    
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
        # scenario_number = 0
        self._TrafficGen.generate_routefile(episode, scenario_number)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._sum_neg_reward = 0
        # self._sum_queue_length = 0
        # self._sum_waiting_time = 0
        # self._sum_delay = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1
        self._current_phase = -1 #dummy
        self._elapsed_time_since_phase_start = 0
        
        #prepare the model used for prediction (based on the previously trained model)
        self._update_and_reset_predict_model()
        
        
        

        while self._step < self._max_steps:
            # get current state of the intersection
            current_state = self._get_state()
            # print("current_state shape: ", len(current_state))

            # calculate reward of previous action: (change in cumulative delay time between actions)
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
            # state = np.expand_dims(state, axis = 0)
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
            x_conv = np.zeros((len(batch), self._Model._sequence_length) + self._Model._state_shape[0]) #from online network
            x_phase = np.zeros((len(batch), self._Model._sequence_length) + (self._Model._state_shape[1], )) #from online network
            x_elapsed = np.zeros((len(batch), self._Model._sequence_length) + (self._Model._state_shape[2], )) #from online network
            y = np.zeros((len(batch), self._Model._sequence_length, self._num_actions))  #from target network


            for index_sequence, sequence in enumerate(batch):
                for index_step, step in enumerate(sequence):
                    state, action, reward, _ = step[0], step[1], step[2], step[3]  # extract data from one sample
                    current_q = q_s_a[index_sequence][index_step]  # get the Q(state) predicted before
                    
                    #update with combination of online and target network
                    current_q[action] = reward + self._gamma * np.amax(q_s_a_d[index_sequence][index_step])  # update Q(state, action)
                    x_conv[index_sequence][index_step] = state[0]
                    x_phase[index_sequence][index_step] = state[1]
                    x_elapsed[index_sequence][index_step] = state[2]
                    # x[index_sequence][index_step] = state
                    y[index_sequence][index_step] = current_q
            self._Model.train_batch([x_conv, x_phase, x_elapsed], y)  # train the NN
            # self._Model.train_batch(x, y)  # train the NN










class TestSimulation(Simulation):
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_actions, scenario_number):
        super().__init__(Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_actions)
        
        self._queue_length_all_episodes = []
        self._delay_all_episodes = []
        self._CV_delay_all_episodes = []
        self._RV_delay_all_episodes = []
        self._wait_all_episodes = []
        
        self._scenario_number = scenario_number #only single scenario is tested at once
        
        
    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo for "steps_todo" steps
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        
        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            self._elapsed_time_since_phase_start +=1 #update the elapsed green time counter
            steps_todo -= 1
            
            #ADD KPI TO LIST
            average_delay, average_CV_delay, average_RV_delay = self._get_average_vehicle_delay()
                        
            self._delay_episode.append(average_delay)
            self._CV_delay_episode.append(average_CV_delay)
            self._RV_delay_episode.append(average_RV_delay)
            self._queue_length_episode.append(self._get_queue_length())
            self._wait_episode.append(self._get_average_waiting_times())
            
            
            
            


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
        old_action = -1 # dummy init
        self._current_phase = -1 #dummy
        self._elapsed_time_since_phase_start = 0
        
        #list for the data for just this one episode. Reset for every new tested episode
        self._delay_episode = []
        self._CV_delay_episode = []
        self._RV_delay_episode = []
        self._queue_length_episode = []
        self._wait_episode = []

        while self._step < self._max_steps:
            # get current state of the intersection
            current_state = self._get_state()

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
        
        #when episode is over, add the whole list with epsode stats to the full list of all episodes
        self._delay_all_episodes.append(self._delay_episode)
        self._CV_delay_all_episodes.append(self._CV_delay_episode)
        self._RV_delay_all_episodes.append(self._RV_delay_episode)
        self._queue_length_all_episodes.append(self._queue_length_episode)
        self._wait_all_episodes.append(self._wait_episode)
        
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time

    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.argmax(self._Model.predict_one(state)) # the best action given the current state
    
    
    
    
    def _get_average_vehicle_delay(self):
        """
        Retrieve the average delay of every vehicle of all types together and the requested type currently in the simulation
        """
        total_delay = 0
        total_CV_delay = 0
        total_RV_delay = 0
        
        car_list = traci.vehicle.getIDList()
        
        number_of_cars = len(car_list)
        number_of_CV = 0
        number_of_RV = 0
        
        for car_id in car_list:
            cars_vehicle_type = self._TrafficGen._generated_vehicles[int(car_id)][1] 
            
            #actual driving time = current time - departure time
            actual_driving_time = self._step - self._TrafficGen._generated_vehicles[int(car_id)][0] 
            #optimal driving time = distance driven / optimal speed on the road (13.89m/s)
            optimal_driving_time = traci.vehicle.getDistance(car_id) / 13.89            
            delay = actual_driving_time - optimal_driving_time
            
            #ADD CAR TO TOTAL DELAY
            total_delay += delay 
            
            #ADD CAR TO SPECIFIC TYPE OF VEHICLE DELAY
            if cars_vehicle_type == "connected_vehicle":
                number_of_CV += 1
                total_CV_delay += delay
            else:
                number_of_RV += 1
                total_RV_delay += delay
                
        #CALCULATE THE AVERAGE DELAY
        if number_of_cars > 0:    
            average_delay = total_delay / number_of_cars
        else:
            average_delay = 0
            
        if number_of_CV > 0:    
            average_CV_delay = total_CV_delay / number_of_CV
        else:
            average_CV_delay = 0
            
        if number_of_RV > 0:    
            average_RV_delay = total_RV_delay / number_of_RV
        else:
            average_RV_delay = 0

        return average_delay, average_CV_delay, average_RV_delay
    
    
    
    
    
    
    def _get_average_waiting_times(self):
        """
        Retrieve the average waiting time of every car in the incoming roads
        """
        cumulative_waiting_time = 0
   
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        number_of_cars = len(car_list)
        for car_id in car_list:
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                cumulative_waiting_time += wait_time
        
        #calculate the average
        if number_of_cars > 0:    
            average_waiting_time = cumulative_waiting_time / number_of_cars
        else:
            average_waiting_time = 0
        
        return average_waiting_time


    @property
    def delay_all_episodes(self):
        return self._delay_all_episodes
        
    @property
    def CV_delay_all_episodes(self):
        return self._CV_delay_all_episodes
        
    @property
    def RV_delay_all_episodes(self):
        return self._RV_delay_all_episodes
        
    @property
    def queue_length_all_episodes(self):
        return self._queue_length_all_episodes
        
    @property
    def wait_all_episodes(self):
        return self._wait_all_episodes
