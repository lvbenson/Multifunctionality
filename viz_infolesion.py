# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:05:04 2020

@author: Lauren Benson
"""

import numpy as np
import ffann                #Controller
import invpend              #Task 1
import cartpole             #Task 2
import leggedwalker         #Task 3
import matplotlib.pyplot as plt
import sys


#dir = str(sys.argv[1])
#reps = int(sys.argv[1])
reps = 1

# ANN Params
nI = 3+4+3
nH1 = 5 #20
nH2 = 5 #10
nO = 1+1+3 #output activation needs to account for 3 outputs in leggedwalker
WeightRange = 15.0
BiasRange = 15.0

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

# Fitness function
def performance_analysis(genotype):
    # Common setup
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    Hidden1_Avg = np.zeros((3,nH1))
    Hidden2_Avg = np.zeros((3,nH2))

    # Task 1
    body = invpend.InvPendulum()
    fit = 0.0
    total_steps = len(time_IP)*total_trials_IP
    for theta in theta_range_IP:
        for theta_dot in thetadot_range_IP:
            body.theta = theta
            body.theta_dot = theta_dot
            for t in time_IP:
                nn.step(np.concatenate((body.state(),np.zeros(4),np.zeros(3)))) #arrays for inputs for each task
                Hidden1_Avg[0] += nn.Hidden1Activation/total_steps
                Hidden2_Avg[0] += nn.Hidden2Activation/total_steps
                f = body.step(stepsize_IP, np.array([nn.output()[0]]))
                
                fit += f
    fitness1 = fit/(duration_IP*total_trials_IP)
    fitness1 = (fitness1+7.65)/7 # Normalize to run between 0 and 1
    np.save('ip_hidden_1', Hidden1_Avg[0])
    np.save('ip_hidden_2', Hidden2_Avg[0])


    # Task 2
    body = cartpole.Cartpole()
    fit = 0.0
    total_steps = len(time_CP)*total_trials_CP
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
                        Hidden1_Avg[1] += nn.Hidden1Activation/total_steps
                        Hidden2_Avg[1] += nn.Hidden2Activation/total_steps
                        f = body.step(stepsize_CP, np.array([nn.output()[1]]))
                        fit += f
    fitness2 = fit/(duration_CP*total_trials_CP)
    np.save('cp_hidden_1', Hidden1_Avg[1])
    np.save('cp_hidden_2', Hidden2_Avg[1])

    #Task 3
    body = leggedwalker.LeggedAgent(0.0,0.0)
    fit = 0.0
    total_steps = len(time_LW)*total_trials_LW
    for theta in theta_range_LW:
        for omega in omega_range_LW:
            body.reset()
            body.angle = theta
            body.omega = omega
            for t in time_LW:
                nn.step(np.concatenate((np.zeros(3),np.zeros(4),body.state())))
                Hidden1_Avg[2] += nn.Hidden1Activation/total_steps
                Hidden2_Avg[2] += nn.Hidden2Activation/total_steps
                body.step(stepsize_LW, np.array(nn.output()[2:5]))
            fit += body.cx/duration_LW
    fitness3 = (fit/total_trials_LW)/MaxFit
    np.save('lw_hidden_1', Hidden1_Avg[2])
    np.save('lw_hidden_2', Hidden2_Avg[2])
    return fitness1,fitness2,fitness3,Hidden1_Avg,Hidden2_Avg

def single_neuron_ablations_infolesion(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)

    # Task 1
    ip_fit = np.zeros(nH1+nH2)
    body = invpend.InvPendulum()
    index = 0
    for neuron in range(nI,nI+nH1+nH2): #iterates through each neuron (20 neurons)
        fit = 0.0
        nn.setParameters(genotype,WeightRange,BiasRange)
        
        nn.ablate_infolesion_ip(neuron)
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
        if fit < 0.0:
            fit = 0.0
        ip_fit[index]=fit
        index += 1

    # Task 2
    cp_fit = np.zeros(nH1+nH2)
    body = cartpole.Cartpole()
    index = 0
    for neuron in range(nI,nI+nH1+nH2):
        fit = 0.0
        nn.setParameters(genotype,WeightRange,BiasRange)
        nn.ablate_infolesion_cp(neuron) #isolates each neuron, sets outdegree to 0
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
        if fit < 0.0:
            fit = 0.0
        cp_fit[index]=fit
        index += 1

    #Task 3
    lw_fit = np.zeros(nH1+nH2)
    body = leggedwalker.LeggedAgent(0.0,0.0)
    index = 0
    for neuron in range(nI,nI+nH1+nH2):
        print(neuron)
        fit = 0.0
        nn.setParameters(genotype,WeightRange,BiasRange)
        nn.ablate_infolesion_lw(neuron)
        for theta in theta_range_LW:
            for omega in omega_range_LW:
                body.reset()
                body.angle = theta
                body.omega = omega
                for t in time_LW:
                    nn.step(np.concatenate((np.zeros(3),np.zeros(4),body.state())))
                    body.step(stepsize_LW, np.array(nn.output()[2:5]))
                fit += body.cx/duration_LW # XXX
        fit = (fit/total_trials_LW)/MaxFit
        if fit < 0.0:
            fit = 0.0
        lw_fit[index]=fit
        index += 1

    return ip_fit,cp_fit,lw_fit


#gens = len(np.load(dir+"/average_history_"+dir+"_0.npy"))
gens = len(np.load('average_history_0.npy'))
#gs=len(np.load(dir+"/best_individual_"+dir+"_0.npy"))
gs = len(np.load('best_individual_0.npy'))
af = np.zeros((reps,gens))
bf = np.zeros((reps,gens))
bi = np.zeros((reps,gs))
#for i in range(reps):
    #af[i] = np.load(dir+"/average_history_"+dir+"_"+str(i)+".npy")
af = np.load('average_history_0.npy')
    #bf[i] = np.load(dir+"/best_history_"+dir+"_"+str(i)+".npy")
bf = np.load('best_history_0.npy')
    #bi[i] = np.load(dir+"/best_individual_"+dir+"_"+str(i)+".npy")
bi = np.load('best_individual_0.npy')
#if bf[-1]>0.00005:
print('best overall fitness',bf[-1])
f1,f2,f3,H1,H2=performance_analysis(bi)
ipf,cpf,lwf=single_neuron_ablations_infolesion(bi)
ipp=ipf/f1
cpp=cpf/f2
lwp=lwf/f3
plt.plot(ipp,'ro') #ip is red
plt.plot(cpp,'go') #cp is green
plt.plot(lwp,'bo') #lw is blue
plt.xlabel("Interneurons")
plt.ylabel("Relative performance")
plt.title("Interneuron Lesions")
plt.show()


#numbers of neurons that affect each task, followed by number of neurons affecting two tasks,
#then number of neurons that affect all three tasks


ip_index = 0
cp_index = 0
lw_index = 0
ip_cp_index = 0
ip_lw_index = 0
cp_lw_index = 0
three_index = 0
for (ip_neuron, cp_neuron, lw_neuron) in zip(ipp,cpp,lwp):
    if ip_neuron < 1:                   #ip task neurons
        ip_index = ip_index + 1
    if cp_neuron < 1:                   #cp task neurons
        cp_index = cp_index + 1
        #print('ip neuron:',ip_neuron)
    if lw_neuron < 1:                   #lw task neurons
        lw_index = lw_index + 1
        #print('lw neuron:',lw_neuron)
    if (ip_neuron < 1) and (cp_neuron < 1): #ip AND cp task neurons
        ip_cp_index = ip_cp_index + 1
    if (ip_neuron < 1) and (lw_neuron < 1): #ip AND lw task neurons
        ip_lw_index = ip_lw_index + 1
    if (cp_neuron < 1) and (lw_neuron < 1): #cp AND lw task neurons
        cp_lw_index + cp_lw_index + 1
    if (ip_neuron < 1) and (cp_neuron < 1) and (lw_neuron < 1): #all 3 task neurons
        three_index = three_index + 1
        
print('ip neurons:',ip_index)
print('cp neurons:',cp_index)
print('lw neurons:',lw_index)
print('ip and cp neurons:',ip_cp_index)
print('ip and lw neurons:',ip_lw_index)
print('cp and lw neurons:',cp_lw_index)
print('three task neurons:',three_index)




plt.plot(af.T,'y')
plt.plot(bf.T,'b')
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.title("Evolution")
plt.show()
