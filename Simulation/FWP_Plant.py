import simpy
import math
import numpy as np

class FlywheelPendulumModel:
        
    """
    Flywheel Pendulum Model (Inverted Pendulum with Reaction Wheel)
    
    ANGLE CONVENTION:
    - theta = 0      : UPRIGHT (balanced vertically up)
    - theta = +π/2   : Horizontal right (90° clockwise from up)
    - theta = -π/2   : Horizontal left (90° counter-clockwise from up)
    - theta = ±π     : HANGING DOWN (180° from upright)
    
    EQUATIONS OF MOTION (derived from Lagrangian mechanics):
    
    Coordinates: theta (pendulum angle from upright), phi (flywheel absolute angle)
    
    Kinetic energy:
        T = ½(I_rod + m_w·l²)·theta_dot² + ½·I_w·phi_dot²
    
    Potential energy (theta=0 is up, so V is maximized at upright → unstable):
        V = (self.m * self.l_cog + self.mw * self.l) * self.g * np.cos(theta)
    
    Motor torque tau is internal (between rod and flywheel):
        Q_theta = -tau    (reaction on pendulum)
        Q_phi = +tau    (drives flywheel)
    
    Euler-Lagrange equations:
        (self.Ip_rod + self.mw * self.l**2) * theta_ddot = (self.m * self.l_cog + self.mw * self.l) * self.g * np.sin(theta) - tau - self.b_pendulum * theta_dot
        self.Iw * phi_ddot = tau - self.b_wheel * phi_dot  (b_p and b_w are damping coefficients for pendulum and flywheel)
    """
    
    def __init__(self, pendulum_params, flywheel_params, dt=0.01):
        # Pendulum parameters
        self.l = pendulum_params.get('length', 0.33)
        self.b_pendulum = pendulum_params.get('damping', 0.01)

        # Uniform rod
        rod_mass = pendulum_params.get('mass', 0.1)
        rod_inertia = rod_mass * self.l**2 / 12.0
        rod_cog = self.l / 2.0
        
        # Point masses: list of dicts with 'mass' and 'pos' (from pivot)
        self.point_masses = pendulum_params.get('point_masses', [])
        # Build mass list: rod as a mass at its CoG + all point masses
        mass_list = [{'mass': rod_mass, 'pos': rod_cog, 'inertia': rod_inertia}]
        for pm in self.point_masses:
            # Each point mass: inertia = m * r^2 (about pivot)
            mass_list.append({'mass': pm['mass'], 'pos': pm['pos'], 'inertia': pm['mass'] * pm['pos']**2})
        
        
        self.m = sum(m['mass'] for m in mass_list) # Total mass
        self.l_cog = sum(m['mass'] * m['pos'] for m in mass_list) / self.m if self.m > 0 else 0.0 # Center of mass
        self.Ip_rod = sum(m['inertia'] for m in mass_list) # Total inertia about pivot
        
        # Flywheel parameters
        self.mw = flywheel_params.get('mass', 0.5)
        self.radius = flywheel_params.get('radius', 0.05)
        self.Iw = flywheel_params.get('inertia', 0.003)
        self.max_torque = flywheel_params.get('max_torque', 0.35)
        self.b_wheel = 0.0  # No flywheel damping

        # Gravity
        self.g = 9.81
        self.dt = dt
        
        # Angle constraints (radians, around upright)
        self.theta_min = pendulum_params.get('theta_min', -np.radians(90))
        self.theta_max = pendulum_params.get('theta_max', np.radians(90))
        
        # Total inertia about pivot: rod + flywheel as point mass at tip
        self.J_total = self.Ip_rod + self.mw * (self.l ** 2)
        
        # Gravity torque coefficient: (m·l_cog + m_w·l)·g
        self.mgl_term = (self.m * self.l_cog + self.mw * self.l) * self.g

        # State: [theta, theta_dot, phi, phi_dot]
        self.state = np.zeros(4)
        self.motor_power = 0.0

        self.theta_history = []
        self.thetadot_history = []
        self.phi_history = []
        self.phidot_history = []

    def set_state(self, theta, theta_dot, phi, phi_dot):
        # Clamp to valid range
        theta = np.clip(theta, self.theta_min, self.theta_max)
        self.state = np.array([theta, theta_dot, phi, phi_dot], dtype=float)

    def get_state(self):
        return tuple(self.state)

    def set_motor_power(self, power):
        self.motor_power = np.clip(power, -1.0, 1.0)

    def get_angle_limits_deg(self):
        """Return angle limits in degrees."""
        return np.degrees(self.theta_min), np.degrees(self.theta_max)

    def step(self):
        theta, theta_dot, phi, phi_dot = self.state
        tau = self.motor_power * self.max_torque
        
        # Pendulum
        gravity_torque = self.mgl_term * np.sin(theta)
        theta_ddot = (gravity_torque - tau - self.b_pendulum * theta_dot) / self.J_total
        
        # Flywheel
        phi_ddot = tau / self.Iw
        
        # Integrators
        theta_dot_new = theta_dot + theta_ddot * self.dt
        phi_dot_new = phi_dot + phi_ddot * self.dt
        theta_new = theta + theta_dot_new * self.dt
        phi_new = phi + phi_dot_new * self.dt
        
        # Clamp theta to limits
        if self.theta_min is not None and self.theta_max is not None:
            if theta_new < self.theta_min:
                theta_new = self.theta_min
                theta_dot_new = 0.0  # stop at limit
            elif theta_new > self.theta_max:
                theta_new = self.theta_max
                theta_dot_new = 0.0  # stop at limit
        
        # phi wrap to [-pi, pi] for better interpretability (not strictly necessary for dynamics)
        phi_new = (phi_new + np.pi) % (2 * np.pi) - np.pi
        self.state = np.array([theta_new, theta_dot_new, phi_new, phi_dot_new], dtype=float)
        
        # save history for plotting
        self.theta_history.append(theta_new)
        self.thetadot_history.append(theta_dot_new)
        self.phi_history.append(phi_new)
        self.phidot_history.append(phi_dot_new)

        return self.get_state()
    
    
    def simpy_generator(self, env, steps=None):
        """
        SimPy generator for discrete time simulation.
        Yields a step event every self.dt seconds in simulation time.
        Optionally, limit to a number of steps (steps=None for infinite).
        Usage:
            env = simpy.Environment()
            model = FlywheelPendulumModel(...)
            env.process(model.simpy_generator(env, steps=1000))
            env.run()
        """
        n = 0
        while steps is None or n < steps:
            self.step()
            yield env.timeout(self.dt)
            n += 1
    
    