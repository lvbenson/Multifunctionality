import numpy as np

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class InvPendulum():

    def __init__(self):
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.g = 10.0       # gravity
        self.m = 1.0        # mass
        self.l = 1.0        # pole length
        self.theta = 0.0
        self.theta_dot = 0.0
        self.force_mag = 2.0

    def step(self, stepsize, u):
        u = np.clip(u*self.force_mag, -self.max_torque, self.max_torque)[0]
        cost = angle_normalize(self.theta)**2 + .1*self.theta_dot**2 + .001*(u**2)
        self.theta_dot += stepsize * (-3*self.g/(2*self.l) * np.sin(self.theta + np.pi) + 3./(self.m*self.l**2)*u)
        self.theta += stepsize * self.theta_dot
        self.theta_dot = np.clip(self.theta_dot, -self.max_speed, self.max_speed)
        return -cost*stepsize

    def state(self):
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot])
