import numpy as np

# According to conversations with Abe,
# the optimal behavior should be around
# -130. This is with a timestep of 0.05
# and a duration of 10 seconds. Therefore,
# the normalized optimal cost should be 0.65.
# I'm getting closer to 0.94 (188).

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return x * (x > 0)

class ANN:

    def __init__(self, NIU, NH1U, NH2U, NOU):
        self.nI = NIU #input
        self.nH1 = NH1U #hidden layer 1
        self.nH2 = NH2U #hidden layer 2
        self.nO = NOU #output
        self.wIH1 = np.zeros((NIU,NH1U)) #weight input/hidden layer 1
        self.wH1H2 = np.zeros((NH1U,NH2U)) #weight hidden layer 1/hidden layer 2
        self.wH2O = np.zeros((NH2U,NOU)) #weight hidden layer 2/output
        self.bH1 = np.zeros(NH1U) #bias hidden layer 1
        self.bH2 = np.zeros(NH2U) #bias hidden layer 2
        self.bO = np.zeros(NOU) #bias output
        self.Hidden1Activation = np.zeros(NH1U)
        self.Hidden2Activation = np.zeros(NH2U)
        self.OutputActivation = np.zeros(NOU)
        self.Input = np.zeros(NIU)

# =============================================================================
#     def setParametersSTD(self, genotype, std):
#         k = 0
#         for i in range(self.nI):
#             for j in range(self.nH1):
#                 self.wIH1[i][j] = genotype[k]*(std/(self.nI*self.nH1))
#                 k += 1
#         for i in range(self.nH1):
#             for j in range(self.nH2):
#                 self.wH1H2[i][j] = genotype[k]*(std/(self.nH1*self.nH2))
#                 k += 1
#         for i in range(self.nH2):
#             for j in range(self.nO):
#                 self.wH2O[i][j] = genotype[k]*(std/(self.nH2*self.nO))
#                 k += 1
#         for i in range(self.nH1):
#             self.bH1[i] = genotype[k]*(std/self.nH1)
#             k += 1
#         for i in range(self.nH2):
#             self.bH2[i] = genotype[k]*(std/self.nH2)
#             k += 1
#         for i in range(self.nO):
#             self.bO[i] = genotype[k]*(std/self.nO)
#             k += 1
# 
    def setParameters(self, genotype, WeightRange, BiasRange):
         k = 0
         for i in range(self.nI):
             for j in range(self.nH1):
                 self.wIH1[i][j] = genotype[k]*WeightRange
                 k += 1
         for i in range(self.nH1):
             for j in range(self.nH2):
                 self.wH1H2[i][j] = genotype[k]*WeightRange
                 k += 1
         for i in range(self.nH2):
             for j in range(self.nO):
                 self.wH2O[i][j] = genotype[k]*WeightRange
                 k += 1
         for i in range(self.nH1):
             self.bH1[i] = genotype[k]*BiasRange
             k += 1
         for i in range(self.nH2):
             self.bH2[i] = genotype[k]*BiasRange
             k += 1
         for i in range(self.nO):
             self.bO[i] = genotype[k]*BiasRange
             k += 1
 
    def ablate(self, neuron): # Set outgoing connections to 0
        if (neuron<self.nI): #if neuron is an input neuron
            i = neuron
            for j in range(self.nH1):
                self.wIH1[i][j] = 0.0 #set every connection from input neuron to hidden layer 1 to 0
        if (neuron >= self.nI and neuron < self.nI+self.nH1): #if neuron is in nH1
            i = neuron-self.nI
            for j in range(self.nH2):
                self.wH1H2[i][j] = 0.0 #set every connection from neuron in hidden layer 1 to hidden layer 2 to 0
        if (neuron >= self.nI+self.nH1 and neuron < self.nI+self.nH1+self.nH2): #if neuron is in nH2
            i = neuron-(self.nI+self.nH1)
            for j in range(self.nO):
                self.wH2O[i][j] = 0.0 #set every connection from neuron in hidden layer 2 to output to 0
                
    def step(self,Input):
        self.Input = np.array(Input)
        self.Hidden1Activation = relu(np.dot(self.Input.T,self.wIH1)+self.bH1)
        self.Hidden2Activation = relu(np.dot(self.Hidden1Activation,self.wH1H2)+self.bH2)
        self.OutputActivation = sigmoid(np.dot(self.Hidden2Activation,self.wH2O)+self.bO)
        return self.OutputActivation

    def output(self):
        #print("output",self.OutputActivation)
        #print("ffann output",self.OutputActivation*2 - 1)
        return self.OutputActivation*2 - 1
    

    def states(self):
        return np.concatenate((self.Input,self.Hidden1Activation,self.Hidden2Activation,self.OutputActivation))
