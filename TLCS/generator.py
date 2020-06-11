import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, penetration_rate):
        self._max_steps = max_steps
        self._penetration_rate = penetration_rate
        self._generated_vehicles = None    #sorted list of all generated vehicles. Contains the departure time, and vehicle type

    def generate_routefile(self, seed, scenario_number):
        """
        Generation of the route of every car for one episode 
        """
        
        np.random.seed(seed)  # make tests reproducible
        
        # #GENERATE THE CORRECT SCNEARIO(NUMBER OF CARS, DYNAMIC/CONSTANT)
        if scenario_number == 0:
            cars_generated = 150
            dynamic = False
        elif scenario_number == 1:
            cars_generated = 2000
            dynamic = False
        elif scenario_number == 2:
            cars_generated = 5000
            dynamic = False
        elif scenario_number == 3:
            cars_generated = 7000
            dynamic = False
        elif scenario_number == 4:
            cars_generated = 2000
            dynamic = True
        elif scenario_number == 5:
            cars_generated = 2500
            dynamic = True
        else:  #must be 0-7
            cars_generated = None
            print("Scenario number must be between 0 and 7")
        
        #initialize generated vehicles array
        self._generated_vehicles = np.zeros(cars_generated, dtype = 'int, U20')
        
        
        #GENERATE TRAFFIC DEMAND FOR THE SCENARIO
        if dynamic == False:
            #uniform distribution
            timings = np.random.uniform(0, self._max_steps, cars_generated)
        else:
            #dynamic distribution (weibull)
            timings = np.random.weibull(2, cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[0])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="regular_vehicle" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" color ="red" />
            <vType accel="1.0" decel="4.5" id="connected_vehicle" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" color ="blue" />

            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                #randomly decide the arrival direction and whether the vehicle turns
                straight_or_turn = np.random.uniform()
                

                
                #decide if regular or connected vehicle:
                if np.random.uniform() < self._penetration_rate:
                    vehicle_type = "connected_vehicle"
                else:
                    vehicle_type = "regular_vehicle"

                
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 5)  # choose a random source & destination
                    if route_straight == 1:
                        print('    <vehicle id="'+str(car_counter)+'" type="' + str(vehicle_type) + '" route="W_E" depart="' + str(step) + '" departLane="random" departSpeed="13.89" />'       , file=routes)
                        # print('    <vehicle id="W_E_'+str(car_counter)+ '" type="' + str(vehicle_type) + '" route="W_E" depart="' + str(step) + '" departLane="random" departSpeed="10" />'       , file=routes)
                    elif route_straight == 2:                       
                        print('    <vehicle id="'+str(car_counter)+'" type="' + str(vehicle_type) + '" route="E_W" depart="' + str(step) + '" departLane="random" departSpeed="13.89" />'       , file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="'+str(car_counter)+'" type="' + str(vehicle_type) + '" route="N_S" depart="' + str(step) + '" departLane="random" departSpeed="13.89" />'       , file=routes)
                    else:                        
                        print('    <vehicle id="'+str(car_counter)+'" type="' + str(vehicle_type) + '" route="S_N" depart="' + str(step) + '" departLane="random" departSpeed="13.89" />'       , file=routes)
                else:  # car that turn - 25% of the time the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="'+str(car_counter)+'" type="' + str(vehicle_type) + '" route="W_N" depart="' + str(step) + '" departLane="random" departSpeed="13.89" />'       , file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="'+str(car_counter)+'" type="' + str(vehicle_type) + '" route="W_S" depart="' + str(step) + '" departLane="random" departSpeed="13.89" />'       , file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="'+str(car_counter)+'" type="' + str(vehicle_type) + '" route="N_W" depart="' + str(step) + '" departLane="random" departSpeed="13.89" />'       , file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="'+str(car_counter)+'" type="' + str(vehicle_type) + '" route="N_E" depart="' + str(step) + '" departLane="random" departSpeed="13.89" />'       , file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="'+str(car_counter)+'" type="' + str(vehicle_type) + '" route="E_N" depart="' + str(step) + '" departLane="random" departSpeed="13.89" />'       , file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="'+str(car_counter)+'" type="' + str(vehicle_type) + '" route="E_S" depart="' + str(step) + '" departLane="random" departSpeed="13.89" />'       , file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="'+str(car_counter)+'" type="' + str(vehicle_type) + '" route="S_W" depart="' + str(step) + '" departLane="random" departSpeed="13.89" />'       , file=routes)
                    else:
                        print('    <vehicle id="'+str(car_counter)+'" type="' + str(vehicle_type) + '" route="S_E" depart="' + str(step) + '" departLane="random" departSpeed="13.89" />'       , file=routes)   
                self._generated_vehicles[car_counter] = (step, vehicle_type)

                
                            

                            

            # print("generated vehicles: ", self._generated_vehicles)   
            print("</routes>", file=routes)
