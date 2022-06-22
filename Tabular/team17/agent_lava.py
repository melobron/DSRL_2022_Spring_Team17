import numpy as np
from lava_grid import ZigZag6x10
import csv

def sampleDirichletMat(alpha):
    S = alpha.shape[0]
    A = alpha.shape[2]
    theta = np.zeros((S,S,A))
    scale = 1

    theta = np.random.gamma(alpha,scale=scale)
    theta = theta / np.tile(np.sum(theta,axis=0),[S,1,1])

    return theta

def sampleNormalGammaMat( mu0, nMu0, tau0, nTau0, nObs, muObs, varObs ):
    lambda0 = nMu0
    alpha0 = nTau0 / 2
    beta0 = alpha0 / tau0

    mu = (lambda0*mu0 + nObs*muObs) / (lambda0+nObs)
    lmbda = lambda0 + nObs
    alpha = alpha0 + nObs/2
    beta = beta0 + 0.5*(nObs*varObs + lambda0*nObs*(muObs-mu0)**2 / (lambda0+nObs))

    tau = np.random.gamma(alpha,1/beta)
    mu = mu + 1/(np.sqrt(lmbda*tau)) * np.random.randn(mu0.shape[0], mu0.shape[1])

    return mu, tau

def bellman(oldVal, probs, rewards):
    S = rewards.shape[0]
    A = rewards.shape[1]
    superSmall = 1.e-8
    qVals = np.ones((S,A))

    for state in range(S):
        for action in range(A):
            qVals[state, action] = rewards[state,action] + np.dot(probs[:,state,action], oldVal)
    
    qVals = qVals+superSmall*np.random.randn(S,A)
    newVal = np.amax(qVals, axis = 1)
    newPol = np.argmax(qVals, axis = 1)

    return newVal, newPol

def dpValueIteration(probs, rewards, tau):
    S = rewards.shape[0]
    A = rewards.shape[1]
    oldVal = np.zeros((S,1))
    complete = False
    err = 1
    nIter = 1
    flag = 1

    for i in range(tau):
        value, policy = bellman(oldVal, probs, rewards)
        oldVal = value
    
    return value, policy

class agent():
    
    def __init__(self, env):
        # self.sample_actions = [3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        self.s1 = 1
        self.num_states = 60
        self.num_actions = 4
        self.M = 3000   #number of episode
        self.tau = 100   #length of episode

        self.vEps = np.zeros((self.num_states, self.num_actions))
        self.vTot = np.zeros((self.num_states, self.num_actions))
        self.vLim = np.maximum(np.ones((self.num_states, self.num_actions)),
                            self.vTot)
        self.pEmp = np.zeros((self.num_states, self.num_states, self.num_actions))
        self.rMean = np.zeros((self.num_states, self.num_actions))
        self.rVar = np.zeros((self.num_states, self.num_actions))
        self.T = self.M * self.tau

        self.pSample = np.zeros((self.num_states, self.num_states, self.num_actions))
        self.muSample = np.zeros((self.num_states, self.num_actions))
        self.varSample = np.zeros((self.num_states, self.num_actions))

        self.rewards = np.zeros((self.T, 1))
        self.states = np.zeros((self.T, 1))
        self.actions = np.zeros((self.T, 1))
        self.values = np.zeros((self.M, self.num_states))
        self.pols = np.zeros((self.M, self.num_states))

        self.mu0 = np.ones((self.num_states, self.num_actions))
        self.nMu0 = np.ones((self.num_states, self.num_actions))
        self.tau0 = np.ones((self.num_states, self.num_actions))
        self.nTau0 = np.ones((self.num_states, self.num_actions))

        self.alpha0 = 1/self.num_states * np.ones((self.num_states, self.num_states, self.num_actions))
        
        self.data = []
        
        return
    
    def epi_update(self, episode):
        self.vEps = np.zeros((self.num_states, self.num_actions))
        self.vLim = np.maximum(np.ones((self.num_states, self.num_actions)), self.vTot)

        self.alpha = self.alpha0 + self.pEmp
        self.pSample = sampleDirichletMat(self.alpha)   
        self.muSample, self.varSample = sampleNormalGammaMat(self.mu0, self.nMu0, self.tau0, self.nTau0, self.vTot, self.rMean, self.rVar)

        value, policy = dpValueIteration(self.pSample,self.muSample,self.tau)

        self.values[episode] = value
        self.pols[episode] = policy
    
    def update(self, s, action, ns, reward, episode, done):
        state = np.argmax(s)
        
        self.vEps[state, action] += 1
        self.vTot[state, action] += 1
        nCount = self.vTot[state, action]

        self.rVar[state, action] = ((nCount-1)*self.rVar[state,action] + (reward-self.rMean[state,action])**2)/nCount
        self.rMean[state, action] = ((nCount-1)*self.rMean[state,action] + reward)/nCount

        ns = np.argmax(ns)

        if not done:
            self.pEmp[ns, state, action] += 1
        
        self.data.append([state, action, ns, reward])
    
    def give_data(self):
        return self.data

    def action(self, episode, state):
        state = np.argmax(state)
        
        action = int(self.pols[episode,state])
        # return self.sample_actions.pop(0)
        return action