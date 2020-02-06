# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:46:10 2020

@author: Lauren Benson

"""

import mga                  #Optimimizer
import ffann             #Controller
import mountaincar_cont    #Task

import numpy as np
import matplotlib.pyplot as plt


 
# ANN Params
nI = 2
nH1 = 5
nH2 = 5
nO = 1
duration = 30.0
stepsize = 0.1
WeightRange = 15.0 #/nH1
BiasRange = 15.0 #/nH1

noisestd = 0.0 #0.01

# Fitness initialization ranges
trials_theta = 3
theta_range = np.linspace(-np.pi/6, np.pi/6, num=trials_theta)
trials_position = 3
position_range = np.linspace(-1.0, 1.0, num=trials_position)
trials_velocity = 3
velocity_range = np.linspace(-0.05, 0.05, num=trials_velocity)
total_trials = trials_theta * trials_position * trials_velocity

time = np.arange(0.0,duration,stepsize)
DurationSteps = int(duration/stepsize)

# EA Params
popsize = 20
genesize = (nI*nH1) + (nH1*nH2) + (nH1*nO) + nH1 + nH2 + nO
recombProb = 0.5
mutatProb = 0.05 #0.05
demeSize = popsize #2
generations = 10 #50
boundaries = 0 #1
tournaments = generations * popsize

# Fitness function
def fitnessFunction(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    body = mountaincar_cont.MountainCarAgent(0.0)
    fit = 0.0
    for theta in theta_range:
        for velocity in velocity_range:
            for position in position_range:
                body.theta = theta
                body.position = position
                body.velocity = velocity
                for t in time:
                    nn.step(body.state())
                    f = body.step(stepsize,nn.output()+np.random.normal(0.0,noisestd))
                    fit += f
    fitness = fit/(duration*total_trials)
    return fitness



# --------------------------------------------
# Visualize activity of a network
# --------------------------------------------
def plot(g):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    body = mountaincar_cont.MountainCarAgent(0.0)
    out_hist = np.zeros((DurationSteps,nO))
    velocity = np.zeros(DurationSteps)
    time = np.arange(0,duration,stepsize)
    #t = 0.0
    for i in range(DurationSteps):
        nn.step(body.state())
        body.step(stepsize, nn.output() + np.random.normal(0.0,noisestd))
        out_hist[i] = nn.output()
        velocity[i] = body.vx
    plt.plot(time, velocity)
    plt.xlabel('Time')
    plt.ylabel('Outputs')
    plt.axis([0, duration, -0.1, 1.1])
    plt.show()
    plt.plot(time, out_hist)
    plt.xlabel('Time')
    plt.ylabel('Outputs')
    plt.show()

# Evolve and visualize fitness over generations
ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
ga.run(tournaments)
af,bf,genotype = ga.fitStats()
plot(genotype)

# Get best evolved network and show its activity
af,bf,bi = ga.fitStats()