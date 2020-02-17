import mga                  #Optimimizer
import ffann                #Controller
import invpend              #Task 1
import cartpole             #Task 2
import leggedwalker         #Task 3

import numpy as np
import sys
#sys.argv[0] is the name of the script
number = 0




##viz = int(sys.argv[1])
#savedata = int(sys.argv[3])
#number = int(sys.argv[3])
#viz = 1
#savedata = 0
#number = 0

# ANN Params
nI = 3+4+3
nH1 = 5 #20
nH2 = 5 #10
nO = 3 #output activation needs to account for 3 outputs in leggedwalker
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
popsize = 5
genesize = (nI*nH1) + (nH1*nH2) + (nH1*nO) + nH1 + nH2 + nO
recombProb = 0.5
mutatProb = 0.05
demeSize = popsize
generations = 5
boundaries = 0
tournaments = generations * popsize

# Fitness function
def fitnessFunction(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    # Task 1
    body = invpend.InvPendulum()
    fit = 0.0
    for theta in theta_range_IP:
        for theta_dot in thetadot_range_IP:
            body.theta = theta
            body.theta_dot = theta_dot
            for t in time_IP:
                nn.step(np.concatenate((body.state(),np.zeros(4),np.zeros(3)))) #arrays for inputs for each task             
                #f = body.step(stepsize_IP, nn.output() + np.random.normal(0.0,noisestd))
                f = body.step(stepsize_IP, np.concatenate(((nn.output() + np.random.normal(0.0,noisestd)),np.zeros(2))))
                fit += f
    fitness1 = fit/(duration_IP*total_trials_IP)
    fitness1 = (fitness1+7.65)/7 # Normalize to run between 0 and 1
    #save data from fitnesses of tasks 
    np.save('Fitness_invpend',fitness1)
    #np.load('Fitness_invpend.npy')
    
    # Task 2
    body = cartpole.Cartpole()
    fit = 0.0
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
                        f = body.step(stepsize_IP, nn.output() + np.random.normal(0.0,noisestd))
                        #f = body.step(stepsize_IP, nn.output())
                        f = body.step(stepsize_CP, np.concatenate(((nn.output() + np.random.normal(0.0,noisestd)),np.zeros(2))))
                        fit += f
    fitness2 = fit/(duration_CP*total_trials_CP)
    #return fitness1*fitness2
    #np.save('Fitness_cartpole',fitness2)
    #np.load('Fitness_cartpole.npy')
    
    #Task 3
    body = leggedwalker.LeggedAgent(0.0,0.0)
    fit = 0.0
    for theta in theta_range_LW:
        for omega in omega_range_LW:   
            body.reset()
            body.angle = theta
            body.omega = omega
            for t in time_LW:
                nn.step(np.concatenate((np.zeros(3),np.zeros(4),body.state())))
                body.step(stepsize_LW, nn.output() + np.random.normal(0.0,noisestd))
                #body.step(stepsize_LW, nn.output())
                #body.step(stepsize_LW, np.concatenate(((nn.output() + np.random.normal(0.0,noisestd)),np.zeros(2))))
            fit += body.cx/duration_LW
    fitness3 = (fit/total_trials_LW)/MaxFit
    np.save('Fitness_legged',fitness3)
    #np.load('Fitness_legged.npy')
    return fitness1*fitness2*fitness3


# Evolve and visualize fitness over generations
ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
ga.run(tournaments)
ga.showFitness()

# Get best evolved network and show its activity
af,bf,bi = ga.fitStats()
#print(ga.avgHistory)
#print(ga.bestHistory)



ah = ga.avgHistory
bh = ga.bestHistory


np.save('average_history',ah)
np.load('average_history.npy')
np.save('best_history',bh)
np.load('best_history.npy')


#np.save("bestfit"+str(number)+".npy",ga.bestHistory)
#np.save("avgfit"+str(number)+".npy",ga.avgHistory)



#Experiment_1avgmean = np.mean(np.array(ah),axis=0)
#Experiment_1bestmean = np.mean(np.array(bh),axis=0)

#Exp1 = np.column_stack((Experiment_1avgmean, Experiment_1bestmean)
#np.savetxt('Experiment1.csv',Exp1)
#header = "Ex1_avg_mean, Ex1_best_mean"
#np.savetxt('Ex1Total.csv', Exp1, delimiter=',')


#np.save('average_history1',ah)
#np.load('average_history1.npy')
#np.save('best_history1',bh)
#np.load('best_history1.npy')

#f=open("average_history2.txt", "a+")
#f.write(ah)

#nf=open("best_history2.txt","a+")
#f.write(bh)

#np.save(ga.avgHistory)
#np.save(ga.bestHistory)
#np.save(bi)


# Evolve and visualize fitness over generations
#ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations, boundaries)
#ga.run(tournaments)

# Get best evolved network and show its activity
#af,bf,bi = ga.fitStats()

#np.save(ga.bestHistory)
#np.save(ga.avgHistory)
#np.save("theta"+str(number)+".npy",theta_hist)
#np.save("fitmap"+str(number)+".npy",fitmap)
#np.save(bi)








#theta_hist, fitmap = evaluate(bi)

## Instead of plotting, save data to file
# =============================================================================
# if viz:
#     ga.showFitness()
#     ga.showAge()
#     #plt.plot(theta_hist.T)
# =============================================================================
    #plt.show()
# =============================================================================
# if savedata:
#     np.save("bestfit"+str(number)+".npy",ga.bestHistory)
#     np.save("avgfit"+str(number)+".npy",ga.avgHistory)
#     #np.save("theta"+str(number)+".npy",theta_hist)
#     np.save("bestgenotype"+str(number)+".npy",bi)
# 
# =============================================================================