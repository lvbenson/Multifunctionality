import math
import ctrnn
import numpy as np

# Nervous system
BiasRange = 16                # Range of bias [
WeightRange = 16              # Range of weights [-16, 16]
SensorWeightRange = 16
TimeConstMax = 1        # Maximum time constant parameter
TimeConstMin = 0.1

# Constants
LegLength = 15
MaxLegForce = 0.05
ForwardAngleLimit = math.pi/6.0
BackwardAngleLimit = -math.pi/6.0
MaxVelocity = 6.0
MaxTorque = 0.5
MaxOmega = 1.0

# Neuron roles
FS = 0
FM = 1
BM = 2

class LeggedAgent:

    def __init__(self, ix, iy):
        self.ix = ix
        self.iy = iy
        self.cx = ix
        self.cy = iy
        self.vx = 0.0
        self.footstate = 0
        self.angle = 0.0
        self.omega = 0.0
        self.forwardForce = 0.0
        self.backwardForce = 0.0
        self.jointX = self.cx
        self.jointY = self.cy + 12.5
        self.footX = self.jointX + LegLength * math.sin(self.angle)
        self.footY = self.jointY + LegLength * math.cos(self.angle)

    def reset(self):
        self.cx = self.ix
        self.cy = self.iy
        self.vx = 0.0
        self.footstate = 0
        self.angle = 0.0
        self.omega = 0.0
        self.forwardForce = 0.0
        self.backwardForce = 0.0
        self.jointX = self.cx
        self.jointY = self.cy + 12.5
        self.footX = self.jointX + LegLength * math.sin(self.angle)
        self.footY = self.jointY + LegLength * math.cos(self.angle)

    def state(self):
        #return np.array([self.angle, self.omega, self.footstate])
        return np.array([self.angle, self.omega, self.footstate])

    def step(self, stepsize, u):
        u = (u+1)/2.0
        force = 0.0

        # Update the leg effectors
        if (u[0] > 0.5):
            self.footstate = 1
            self.omega = 0
        else:
            self.footstate = 0

        self.forwardForce = u[1] * MaxLegForce
        self.backwardForce = u[2] * MaxLegForce

        # Compute force applied to the body
        f = self.forwardForce - self.backwardForce
        if (self.footstate == 1.0):
            if ((self.angle >= BackwardAngleLimit and self.angle <= ForwardAngleLimit) or (self.angle < BackwardAngleLimit and f < 0) or (self.angle > ForwardAngleLimit and f > 0)):
                force = f

        # Update the position of the body
        self.vx = self.vx + stepsize * force
        if (self.vx < -MaxVelocity):
            self.vx = -MaxVelocity
        if (self.vx > MaxVelocity):
            self.vx = MaxVelocity
        self.cx = self.cx + stepsize * self.vx

        # Update the leg geometry
        self.jointX = self.jointX + stepsize * self.vx
        if (self.footstate == 1.0):
            angle = math.atan2(self.footX - self.jointX, self.footY - self.jointY)
            self.omega = (angle - self.angle)/stepsize
            self.angle = angle
        else:
            self.vx = 0.0
            self.omega = self.omega + stepsize * MaxTorque * (self.backwardForce - self.forwardForce)
            if (self.omega < -MaxOmega):
                self.omega = -MaxOmega
            if (self.omega > MaxOmega):
                self.omega = MaxOmega
            self.angle = self.angle + stepsize * self.omega
            if (self.angle < BackwardAngleLimit):
                self.angle = BackwardAngleLimit
                self.omega = 0
            if (self.angle > ForwardAngleLimit):
                self.angle = ForwardAngleLimit
                self.omega = 0
            self.footX = self.jointX + LegLength * math.sin(self.angle)
            self.footY = self.jointY + LegLength * math.cos(self.angle)

        # If the foot is too far back, the body becomes "unstable" and forward motion ceases
        if (self.cx - self.footX > 20):
            self.vx = 0.0
