import traci
import timeit

class FixedTimeTestSimulation:
    def __init__(self, TrafficGen, sumo_cmd, max_steps, fixed_green_time, yellow_duration, num_actions, scenario_number):
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._fixed_green_time = fixed_green_time
        self._yellow_duration = yellow_duration
        self._num_actions = num_actions
        
        self._queue_length_all_episodes = []
        self._delay_all_episodes = []
        self._CV_delay_all_episodes = []
        self._RV_delay_all_episodes = []
        self._wait_all_episodes = []
        
        
        self._scenario_number = scenario_number  #only single scenario is tested at once
        
        
        self._traffic_light_cycle = [0 for x in range(self._fixed_green_time)] + \
                        [1 for x in range(self._yellow_duration)] + \
                        [2 for x in range(self._fixed_green_time)] + \
                        [3 for x in range(self._yellow_duration)] + \
                        [4 for x in range(self._fixed_green_time)] + \
                        [5 for x in range(self._yellow_duration)] + \
                        [6 for x in range(self._fixed_green_time)] + \
                        [7 for x in range(self._yellow_duration)]
        
        
        



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
            average_CV_delay = np.nan
            
        if number_of_RV > 0:    
            average_RV_delay = total_RV_delay / number_of_RV
        else:
            average_RV_delay = np.nan

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
        
        
        
    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(episode, self._scenario_number)
        traci.start(self._sumo_cmd)
        print("Simulating...")
        
        
        # inits
        self._step = 0
        cycle_step = 0
        delay_episode = []
        CV_delay_episode = []
        RV_delay_episode = []
        queue_length_episode = []
        wait_episode = []
        
        
        
        
        while self._step < self._max_steps:

            #SET TRAFFIC LIGHT + INCREASE CYCLE STEP  
            traci.trafficlight.setPhase("TL", self._traffic_light_cycle[cycle_step])
            cycle_step += 1
            if cycle_step >= len(self._traffic_light_cycle):
                cycle_step = 0
            
    
            # DO ONE SIMULATION STEP IN SUMO
            traci.simulationStep()
            self._step += 1 # update the step counter
            
            #ADD KPI TO LIST
            average_delay, average_CV_delay, average_RV_delay = self._get_average_vehicle_delay()
            
            delay_episode.append(average_delay)
            CV_delay_episode.append(average_CV_delay)
            RV_delay_episode.append(average_RV_delay)
            queue_length_episode.append(self._get_queue_length())
            wait_episode.append(self._get_average_waiting_times())
        
        #when episode is over, add the whole list with epsode stats to the full list of all episodes
        self._delay_all_episodes.append(delay_episode)
        self._CV_delay_all_episodes.append(CV_delay_episode)
        self._RV_delay_all_episodes.append(RV_delay_episode)
        self._queue_length_all_episodes.append(queue_length_episode)
        self._wait_all_episodes.append(wait_episode)

        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time
            

        
        
        
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

        
    