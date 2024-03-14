import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Signal:
    def __init__(self, frequency, amplitude=1):
        self.frequency = frequency
        self.amplitude = amplitude

    def generate(self, time):

        return self.amplitude * np.sin(2 * np.pi * self.frequency * time)



class Fault:
    def __init__(self, fault_type, position, severity):
        self.fault_type = fault_type
        self.position = position
        self.severity = severity

    def apply_fault(self, signal_strength, time, sensor_position):
        if sensor_position < self.position:
            # The fault effect only appears after the fault's position
            return signal_strength, time

        # Convert severity to dB reduction for amplitude
        amplitude_reduction = self.severity * 0.5  # 0.5 dB reduction per severity unit
        # Calculate the new signal strength considering the dB reduction
        signal_strength *= 10 ** (-amplitude_reduction / 20)

        if self.fault_type == 'weight':
            # For weights, add a time shift based on the severity
            time += self.severity

        return signal_strength, time

class Sensor:
    def __init__(self, position, signal_speed, faults=[]):
        self.position = position
        self.signal_speed = signal_speed
        self.recordings = []
        self.faults = []  # List of Fault instances

    def capture_signal(self, train_position, signal, time):
        distance = abs(self.position - train_position)
        delay = distance / self.signal_speed
        attenuation = 1 / (1 + distance)  # Simplified attenuation model
        signal_strength = attenuation * signal.generate(time - delay)

        # Apply fault effects if any
        for fault in self.faults:
            signal_strength, adjusted_time = fault.apply_fault(signal_strength, time, self.position)
            # Use adjusted_time if necessary, for example in recording or further processing

        self.recordings.append((time, signal_strength))


def assign_faults_to_sensors(sensors, faults):
    for sensor in sensors:
        for fault in faults:
            if sensor.position > fault.position:
                sensor.faults.append(fault)


class Simulation:
    def __init__(self, signal, sensors):
        self.signal = signal
        self.sensors = sensors

    def run(self, start, end, step, train_speed):
        for time in np.arange(start, end, step):
            train_position = time * train_speed
            for sensor in self.sensors:
                sensor.capture_signal(train_position, self.signal, time)


def save_data(sensors , filename):
    # Initialize an empty DataFrame
    combined_df = pd.DataFrame()
    onearr = []
    # Loop through each sensor and append its signal strength values to the combined DataFrame
    for i, sensor in enumerate(sensors):
        # Extract just the signal strength values from each sensor's recordings
        # print ("i : >",i ," sensor : > " , sensor  )
        signal_strengths = [recording[1] for recording in sensor.recordings]
        # print (signal_strengths )
        # Create a DataFrame column for this sensor's signal strengths
        combined_df[f'Sensor_{i+1}_Signal_Strength'] = signal_strengths
        for sig in signal_strengths :

            onearr.append(sig)
    # Save the combined DataFrame to a CSV file
    return onearr
    # combined_df.to_csv(f"{filename}.csv", index_label='Index')


def visualize_data(sensors):
    plt.figure(figsize=(15, 7))
    for i, sensor in enumerate(sensors):
        df = pd.DataFrame(sensor.recordings, columns=['Time', 'Signal Strength'])
        plt.plot(df['Time'], df['Signal Strength'], label=f'Sensor {i}')

    plt.xlabel('Time')
    plt.ylabel('Signal Strength')
    plt.legend()
    plt.show()


def data_df(train_speed ,fault_pos  ,fault_sev  , fault_tp ):

    col = np.arange(0,400, dtype=int)
    col =np.append(col ,["feq","speed","fualt_type" , "fauly_sev" , "fault_dis"])
    dft = pd.DataFrame(columns=col)
    
    frq  =200
    trainspeed  = train_speed
    signal = Signal(frequency=frq)
    sensors = [Sensor(position=x, signal_speed=34) for x in [200, 400, 600, 800]]
    if fault_tp != None: 
        print ("fault is detected  " ,fault_tp )
        faults = [
            Fault(fault_type=fault_tp, position=fault_pos, severity=fault_sev), ]
        assign_faults_to_sensors(sensors, faults)

    print ("fault is detected  " ,sensors[0].faults,sensors[1].faults ,sensors[2].faults,sensors[3].faults )
    simulation = Simulation(signal, sensors)
    simulation.run(start=0, end=0.01, step=0.0001, train_speed=trainspeed)

    load  = np.array(save_data(sensors , "test_data12")).reshape(1,400)
    print ("sum ",load.sum())
    load = np.append(load[0] , [frq , trainspeed ,fault_tp , fault_sev , fault_pos])

    dft.loc[len(dft)] =load
    return dft



if __name__ == "__main__":
    i = 0
    col = np.arange(0,400, dtype=int)
    col =np.append(col ,["feq","speed","fualt_type" , "fauly_sev" , "fault_dis"])
    df = pd.DataFrame(columns=col)
    print (df.shape)
    # print (col)
    while i <2000:
        print (i )
        frq  =200
        trainspeed  = np.random.randint(30,50)
        signal = Signal(frequency=frq)
        sensors = [Sensor(position=x, signal_speed=34) for x in [200, 400, 600, 800]]
        # crack_fault = Fault(fault_type='crack', position=700, severity=5)
        # sensors[3].faults.append(crack_fault)
        # faults = [
        #     Fault(fault_type='crack', position=250, severity=3),

        # ]
        # assign_faults_to_sensors(sensors, faults)

        simulation = Simulation(signal, sensors)
        simulation.run(start=0, end=0.01, step=0.0001, train_speed=trainspeed)
        # print (sensors)
        load  = np.array(save_data(sensors , "test_data12")).reshape(1,400)
        # load  = np.array(save_data(sensors , "test_data12"))


        # print (load[0])


        # df.loc[len(df)] =np.append(load[0] , [None , 0 , 0])
        load = np.append(load[0] , [frq , trainspeed ,None , 0 , 0])

        print (load.size)

        df.loc[len(df)] =load
        # visualize_data(sensors)

        i =i+1

