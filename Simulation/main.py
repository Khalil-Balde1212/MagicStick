import FWP_Plant
import matplotlib.pyplot as plt
import numpy as np
import simpy

pendulum_params = {
    'length': 0.33,  # length of rod in meters
    'mass': 0.1,     # mass of rod in kg
    'point_masses': [
    {'mass': 0.129*3, 'pos': 0.33} # 129g motor
    ,{'mass': 0.600, 'pos': 0.15} # LiIon battery
    # ,{'mass': 0.350, 'pos': 0.1} # LiPo battery
    ],
    'theta_min': None,  # min angle in radians
    'theta_max': None,   # max angle in radians
}

flywheel_params = {
    'mass': 0.2,          # kg
    'inertia': 0.0045,    # kg*m^2 (moment of inertia of flywheel)
    'radius': 0.05,       # m (radius of flywheel)
    'max_torque': 5,      # N*m
}

if __name__ == "__main__":
    dt = 0.01          # time step for simulation
    sim_time = 20      # total simulation time in seconds
    time_steps = int(sim_time / dt)
    
    plant = FWP_Plant.FlywheelPendulumModel(pendulum_params, flywheel_params, dt)
    plant.set_state(theta=np.radians(5), theta_dot=0, phi=0, phi_dot=0)  # initial conditions
    
    
    # Use the simpy method from FWP_Plant to run the simulation
    env = simpy.Environment()
    env.process(plant.simpy_generator(env, time_steps))
    env.run()
    
    # Plot results
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(np.arange(time_steps) * dt, np.degrees(plant.theta_history))
    plt.title('Pendulum Angle (theta) over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta (degrees)')
    
    plt.subplot(2,1,2)
    plt.plot(np.arange(time_steps) * dt, np.degrees(plant.phidot_history))
    plt.title('Flywheel Speed (phi) over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Phi (degrees/s)')
    plt.tight_layout()
    plt.show()