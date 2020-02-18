import ffann                #Controller
import invpend              #Task
import cartpole             #Task
import leggedwalker         #Task

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ANN Params
nI = 3+4+3
nH1 = 5
nH2 = 5
nO = 3+1+1
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
trials_theta_LW = 3
theta_range_LW = np.linspace(-np.pi/6, np.pi/6, num=trials_theta_LW)
trials_omega_LW = 3
omega_range_LW = np.linspace(-1.0, 1.0, num=trials_omega_LW)
total_trials_LW = trials_theta_LW*trials_omega_LW


def single_neuron_ablations(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    # Task 1
    ip_fit = np.zeros(nI+nH1+nH2)
    body = invpend.InvPendulum()
    for i in range(nI+nH1+nH2): #iterates through each neuron (20 neurons)
        fit = 0.0
        nn.setParameters(genotype,WeightRange,BiasRange)
        nn.ablate(i)
        for theta in theta_range_IP:
            for theta_dot in thetadot_range_IP:
                body.theta = theta
                body.theta_dot = theta_dot
                for t in time_IP:
                    nn.step(np.concatenate((body.state(),np.zeros(4),np.zeros(3))))
                    f = body.step(stepsize_IP, np.array([nn.output()[0]]))
                    fit += f
        fit = fit/(duration_IP*total_trials_IP)
        fit = (fit+7.65)/7 # Normalize to run between 0 and 1
        ip_fit[i]=fit
        
    # Task 2
    cp_fit = np.zeros(nI+nH1+nH2)
    body = cartpole.Cartpole()
    for i in range(nI+nH1+nH2):
        fit = 0.0
        nn.setParameters(genotype,WeightRange,BiasRange)
        nn.ablate(i) #isolates each neuron, sets outdegree to 0
        for theta in theta_range_CP:
            for theta_dot in thetadot_range_CP:
                for x in x_range_CP:
                    for x_dot in xdot_range_CP:
                        body.theta = theta
                        body.theta_dot = theta_dot
                        body.x = x
                        body.x_dot = x_dot
                        for t in time_CP:
                            nn.step(np.concatenate((np.zeros(3),body.state(),np.zeros(3))))
                            f = body.step(stepsize_CP, np.array([nn.output()[1]]))
                            fit += f
        fit = fit/(duration_CP*total_trials_CP)
        cp_fit[i]=fit
    #Task 3
    lw_fit = np.zeros(nI+nH1+nH2)
    body = leggedwalker.LeggedAgent(0.0,0.0)
    for i in range(nI+nH1+nH2):
        fit = 0.0
        nn.setParameters(genotype,WeightRange,BiasRange)
        nn.ablate(i)
        for theta in theta_range_LW:
            for omega in omega_range_LW:
                body.reset()
                body.angle = theta
                body.omega = omega
                for t in time_LW:
                    nn.step(np.concatenate((np.zeros(3),np.zeros(4),body.state())))
                    body.step(stepsize_LW, np.array(nn.output()[2:5]))
                    fit += body.cx/duration_LW
        fit = (fit/total_trials_LW)/MaxFit
        lw_fit[i]=fit
        
   #print('ip fit',ip_fit)
   #print('cp fit',cp_fit)
   #print('lw fit',lw_fit)
    return ip_fit,cp_fit,lw_fit


bi = np.load('best_ind.npy')
ip,cp,lw = single_neuron_ablations(bi)

# 
# plt.scatter(ip, cp, c=lw,cmap='Greens')
# plt.colorbar()
# plt.show()
# 
# =============================================================================
# =============================================================================
