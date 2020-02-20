# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:37:21 2020

@author: Lauren Benson

"""

#simulates three-task script 

from evolve import fitnessFunction
import mga
import numpy as np
import matplotlib.pyplot as plt

# ANN Params
nI = 3+4+3
nH1 = 5 #20
nH2 = 5 #10
nO = 3+1+1 #output activation needs to account for 3 outputs in leggedwalker
WeightRange = 15.0
BiasRange = 15.0

noisestd = 0.0

# Task Params
duration_IP = 10
stepsize_IP = 0.05
duration_CP = 50
stepsize_CP = 0.05
duration_LW = 220.0
stepsize_LW = 0.1
time_IP = np.arange(0.0,duration_IP,stepsize_IP)
time_CP = np.arange(0.0,duration_CP,stepsize_CP)
time_LW = np.arange(0.0,duration_LW,stepsize_LW)

MaxFit = 0.627 #Leggedwalker


# Fitness initialization ranges
#Inverted Pendulum
trials_theta_IP = 6
trials_thetadot_IP = 6
total_trials_IP = trials_theta_IP*trials_thetadot_IP
theta_range_IP = np.linspace(-np.pi, np.pi, num=trials_theta_IP)
thetadot_range_IP = np.linspace(-1.0,1.0, num=trials_thetadot_IP)

#Cartpole
trials_theta_CP = 2
trials_thetadot_CP = 2
trials_x_CP = 2
trials_xdot_CP = 2
total_trials_CP = trials_theta_CP*trials_thetadot_CP*trials_x_CP*trials_xdot_CP
theta_range_CP = np.linspace(-0.05, 0.05, num=trials_theta_CP)
thetadot_range_CP = np.linspace(-0.05, 0.05, num=trials_thetadot_CP)
x_range_CP = np.linspace(-0.05, 0.05, num=trials_x_CP)
xdot_range_CP = np.linspace(-0.05, 0.05, num=trials_xdot_CP)

#Legged walker 
trials_theta = 3
theta_range_LW = np.linspace(-np.pi/6, np.pi/6, num=trials_theta)
trials_omega_LW = 3
omega_range_LW = np.linspace(-1.0, 1.0, num=trials_omega_LW)
total_trials_LW = trials_theta * trials_omega_LW


# EA Params
popsize = 300
genesize = (nI*nH1) + (nH1*nH2) + (nH1*nO) + nH1 + nH2 + nO
recombProb = 0.5
mutatProb = 0.05
demeSize = popsize
generations = 350
boundaries = 0
tournaments = generations * popsize

avghist = []
besthist = []
best_ind = []
reps = 10

for r in range(reps):
    ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
    ga.run(tournaments)
    af,bf,bi = ga.fitStats()
    avghist.append(ga.avgHistory)
    besthist.append(ga.bestHistory)
    best_ind.append(bi)

np.save('avghist',avghist)
np.save('besthist',besthist)
np.save('bi',best_ind)

    
plt.figure()
for i in avghist:
    plt.plot(i)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Average Fitness, 10 networks")
plt.show

plt.figure()
for j in besthist:
    plt.plot(j)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Best Fitness, 10 networks")
plt.show
    
    
    