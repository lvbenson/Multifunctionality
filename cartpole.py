import numpy as np

class Cartpole:
    def __init__(self):
        self.obs_dim = 4
        self.action_dim = 1
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5  # actually half the pole's length
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0
        self.force_mag = 10.0
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

    def step(self, stepsize, action):
        force = action[0] * self.force_mag
        costheta = np.cos(self.theta)
        sintheta = np.sin(self.theta)
        temp = (force + self.polemass_length * self.theta_dot * self.theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        self.x += stepsize * self.x_dot
        self.x_dot += stepsize * xacc
        self.theta += stepsize * self.theta_dot
        self.theta_dot += stepsize * thetaacc
        done = bool(self.x < -self.x_threshold or self.x > self.x_threshold or self.theta < -self.theta_threshold_radians or self.theta > self.theta_threshold_radians)
        if not done:
            return 1.0 * stepsize
        else:
            return 0.0

    def state(self):
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot])
