import traci
import timeit

class FixedTimeTestSimulation:
    def __init__(self, TrafficGen, sumo_cmd, max_steps, fixed_green_time, yellow_duration, num_actions):
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._fixed_green_time = fixed_green_time
        self._yellow_duration = yellow_duration
        self._num_actions = num_actions
        self._queue_length_episode = []
        
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
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")
        
        
        # inits
        self._step = 0
        self._waiting_times = {}
        cycle_step = 0
        
        
        
        
        
        while self._step < self._max_steps:
        
            current_total_wait = self._collect_waiting_times()
            
            
            #SET TRAFFIC LIGHT + INCREASE CYCLE STEP  
            traci.trafficlight.setPhase("TL", self._traffic_light_cycle[cycle_step])
            # print("step: ", str(self._step), ", cycle_step: ", str(cycle_step), ", phase: " + str(self._traffic_light_cycle[cycle_step]) )
            
            cycle_step += 1
            
            
            if cycle_step >= len(self._traffic_light_cycle):
                cycle_step = 0
            
    
            # DO ONE SIMULATION STEP IN SUMO
            traci.simulationStep()
            self._step += 1 # update the step counter
            queue_length = self._get_queue_length() 
            self._queue_length_episode.append(queue_length)
            







        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time
            
        
        
    @property
    def queue_length_episode(self):
        return self._queue_length_episode   





        
    