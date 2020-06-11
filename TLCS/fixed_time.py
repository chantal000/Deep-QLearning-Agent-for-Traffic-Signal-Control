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
        # self._queue_length_episode = []
        # self._average_delay_episode = []
        
        self._queue_length_all_episodes = []
        self._average_delay_all_episodes = []
        
        
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
        
    def _get_vehicle_delay(self):
        """
        Retrieve the average delay of every vehicle currently in the simulation
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
            
            # print("step:", str(self._step), ", vehicle: ", car_id, ", departure time: ", \
                    # self._TrafficGen._generated_vehicles[int(car_id)][0], \
                    # ", actual driving time: ", str(actual_driving_time), \
                    # ", distance driven: ", str(traci.vehicle.getDistance(car_id)), \
                    # ", optimal driving time: ", str(optimal_driving_time), \
                    # ", vehicle speed: ", str(traci.vehicle.getSpeed(car_id)), \
                    # ", --DELAY--: ", str(delay))
            
        average_delay = total_delay / len(car_list) if len(car_list) > 0 else 0
        # print("-------------------average delay in step ", str(self._step), ": ", str(average_delay))
        return average_delay
        
        
    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time
        
        
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
        self._waiting_times = {}
        cycle_step = 0
        average_delay_episode = []
        
        
        
        
        while self._step < self._max_steps:
            
            
            
            # current_total_wait = self._collect_waiting_times()
            
            #SET TRAFFIC LIGHT + INCREASE CYCLE STEP  
            traci.trafficlight.setPhase("TL", self._traffic_light_cycle[cycle_step])
            cycle_step += 1
            if cycle_step >= len(self._traffic_light_cycle):
                cycle_step = 0
            
    
            # DO ONE SIMULATION STEP IN SUMO
            traci.simulationStep()
            self._step += 1 # update the step counter
            # queue_length = self._get_queue_length() 
            # self._queue_length_episode.append(queue_length)
            
            # self._average_delay_episode.append(self._get_vehicle_delay())
            average_delay_episode.append(self._get_vehicle_delay())
        #when episode is over, add the whole list with epsode stats to the full list of all episodes
        self._average_delay_all_episodes.append(average_delay_episode)

        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time
            
    
    
    
        
    @property
    def queue_length_episode(self):
        return self._queue_length_episode  


    @property
    def average_delay_episode(self):
        return self._average_delay_episode
        
    @property
    def average_delay_all_episodes(self):
        return self._average_delay_all_episodes





        
    