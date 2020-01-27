import mga                  #Optimimizer
import ffann             #Controller
import leggedwalker    #Task

import numpy as np
import matplotlib.pyplot as plt
import sys

# =============================================================================
# viz = int(sys.argv[1])
# savedata = int(sys.argv[2])
# number = int(sys.argv[3])
# 

MaxFit = 0.627 # http://mypage.iu.edu/~rdbeer/Papers/Beer1999a.pdf
 
 
# ANN Params
nI = 3
nH1 = 5
nH2 = 5
nO = 3
duration = 220.0
stepsize = 0.1
WeightRange = 15.0 #/nH1
BiasRange = 15.0 #/nH1

noisestd = 0.0 #0.01

# Fitness initialization ranges
trials_theta = 3
theta_range = np.linspace(-np.pi/6, np.pi/6, num=trials_theta)
trials_omega = 3
omega_range = np.linspace(-1.0, 1.0, num=trials_omega)
total_trials = trials_theta * trials_omega

time = np.arange(0.0,duration,stepsize)
DurationSteps = int(duration/stepsize)

# EA Params
popsize = 50
genesize = (nI*nH1) + (nH1*nH2) + (nH1*nO) + nH1 + nH2 + nO
recombProb = 0.5
mutatProb = 0.05 #0.05
demeSize = popsize #2
generations = 25 #50
boundaries = 0 #1
tournaments = generations * popsize

# Fitness function
def fitnessFunction(genotype):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    body = leggedwalker.LeggedAgent(0.0,0.0)
    fit = 0.0
    for theta in theta_range:
        for omega in omega_range:
            body.reset()
            body.angle = theta
            body.omega = omega
            for t in time:
                nn.step(body.state())
                body.step(stepsize, nn.output() + np.random.normal(0.0,noisestd))
            fit += body.cx/duration
    return (fit/total_trials)/MaxFit

# --------------------------------------------
# Visualize activity of a network
# --------------------------------------------
def plot(g):
    nn = ffann.ANN(nI,nH1,nH2,nO)
    nn.setParameters(genotype,WeightRange,BiasRange)
    body = leggedwalker.LeggedAgent(0.0,0.0)
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
#
# ## Instead of plotting, save data to file
# if viz:
#     ga.showFitness()
# if savedata:
#     np.save("bestfit"+str(number)+".npy",ga.bestHistory)
#     np.save("avgfit"+str(number)+".npy",ga.avgHistory)
#     np.save("theta"+str(number)+".npy",theta_hist)
#     np.save("fitmap"+str(number)+".npy",fitmap)
#     np.save("bestgenotype"+str(number)+".npy",bi)
