# Author - Aditya Venkataraman <adityav@cs.wisc.edu>
import numpy as np
import pprint as pp
import math

from operator import itemgetter

def argmax(lis):
    '''
    given any list, return argument of max element
    '''
    index, element = max(enumerate(lis), key=itemgetter(1))
    return index

class Hmm():
    ''' an HMM is characterized by three things <A,B,pi>
    A = Transition matrix, NxN where N = no. of possible states for X (hidden variable)
    A = {aij} where aij = P (qj at t+1 | qi at t)
    B = {bj(k)}, NxM sensor observation matrix. M = no. of possible observations for given x.
    bj(k) = P (observation k at t | state qj at t)
    pi = Initial state distribution
    '''
    def __init__ (self, A_init, B_init, pi_init):
        # verify the inputs
        self.pi = {}
        self.A = {}
        self.B = {}

        fail = 0
        for k in A_init.keys():
            if sum(A_init[k].values()) != 1:
                fail = 1
        for k in B_init.keys():
            if sum(B_init[k].values()) != 1:
                fail = 1

        if sum(pi_init.values()) != 1:
            fail = 1

        if fail:
            pp.pprint("Erroneous inputs")
            exit(-1)


        self.A = A_init
        self.B = B_init
        self.pi = pi_init

        # useful lengths
        self.N = len(self.pi.keys())
        self.M = len(self.B[0])

        pp.pprint("Initial A:")
        pp.pprint(self.A)
        pp.pprint("Initial B:")
        pp.pprint(self.B)
        pp.pprint("Initial pi:")
        pp.pprint(self.pi)
        pp.pprint("Initial N = %d, M = %d" % (self.N, self.M))

    def pickSample (self, probs):
        '''
        given a multinomial distribution, it will pick a sample and return index
        '''
        sample_dist = np.random.multinomial(1, probs, size=1)
        return argmax(sample_dist[0])

    def generateObservations (self, num_obs):
        ''' generate num_obs observations of this HMM
        '''
        X = [] # list of Xs
        O = [] # list of obs

        for n in range(num_obs):
            if n == 0: #initial case
                X.append(self.pickSample(self.pi.values())) # pick an initial value based on pi
            else:
                # update hidden variable based on transition model
                X.append(self.pickSample(self.A[X[n-1]].values()))
            # pick an observation based on observation model
            O.append(self.pickSample(self.B[X[n]].values()))
        print X
        return O

    def generateAlpha (self, obs):
        '''
        Given a set of observations, this will calculate the forward messages - alpha & scaling factors c
        '''
        T = len(obs)
        alpha = {} # T x N matrix
        c = {} # HMM scaling. w/o scaling, alpha can underflow for large T

        for t in range(T):
            alpha[t] = []
            if t == 0:
                # initialize alpha
                c[t] = 0
                for i in range(self.N):
                    alpha[t].append(self.pi[i]*self.B[i][obs[t]])

                c[t] = sum(alpha[t])
                c[t] = float(1/c[t])

                for i in range(self.N):
                    alpha[t][i] = c[t]*alpha[t][i]

            else:
                c[t] = 0
                for i in range(self.N):
                    alpha[t].append(0)
                    for j in range(self.N):
                        alpha[t][i] = alpha[t][i] + alpha[t-1][j]*self.A[j][i]
                    alpha[t][i] = alpha[t][i]*self.B[i][obs[t]]
                    c[t] = c[t] + alpha[t][i]

                c[t] = float(1/c[t])
                for i in range(self.N):
                    alpha[t][i] = c[t]*alpha[t][i]


        #print "alpha is ", alpha
        return alpha,c


    def generateBeta(self, obs, c):
        '''
        Given a set of observations & scaling factors, this will generate the backward messages - beta
        '''
        T = len(obs)
        beta = {}

        for t in reversed(range(T)):
            beta[t] = []
            if t == T-1: #initial
                beta[t] = [ c[T-1] for i in range(self.N)]
            else:
                for i in range(self.N):
                    beta[t].append(0)
                    for j in range(self.N):
                        beta[t][i] = beta[t][i] + self.A[i][j]*self.B[j][obs[t+1]]*beta[t+1][j]

                    #scale
                    beta[t][i] = c[t]*beta[t][i]

        return beta

    def generateGammas (self, obs):
        '''
        given a set of obs, returns tuple of gamma and digamma
        '''

        alpha,c = self.generateAlpha(obs)
        beta = self.generateBeta(obs,c)
        filtered_prob = self.likelihoodOfObservations(obs)
        T = len(obs)

        gamma = {}
        digamma = {}
        for t in range(T-1): # t=0 to T-2
            denom = 0
            digamma[t] = {}
            gamma[t] = []
            for i in range(self.N):
                for j in range(self.N):
                    denom = denom + alpha[t][i]*self.A[i][j]*self.B[j][obs[t+1]]*beta[t+1][j]

            for i in range(self.N):
                digamma[t][i] = {}
                gamma[t].append(0)
                for j in range(self.N):
                    digamma[t][i][j] = (alpha[t][i]*self.A[i][j]*self.B[j][obs[t+1]]*beta[t+1][j])/denom
                    gamma[t][i] = gamma[t][i] + digamma[t][i][j]

        return (gamma,digamma)

    def likelihoodOfObservations (self, obs):
        '''
        given a set of observations and knowing the HMM model, we calculate log of likelihood of these observations from the given HMM. the log is calculated to avoid underflow for large T

        P(obs|HMM) = sum over all hidden state (alpha_T_1 (i))

        T = no. of observations

        '''
        T = len(obs)
        alpha,c = self.generateAlpha(obs)
        logProb = 0

        for i in range(T):
            logProb = logProb + math.log(c[i])
        logProb = -1*logProb
        return logProb


    def mostLikelyStateSequence (self, obs):
        '''
        given a set of observations and knowing the HMM model, we return the most likely state sequence that led to this set of observations
        '''
        gamma,digamma = self.generateGammas(obs)
        T = len(obs)

        # likely state sequence
        x = []
        for t in range(T-1):
            x.append(argmax(gamma[t]))

        return x

    def learnModel (self, obs):
        '''
        given a set of observations, we want to learn the HMM model. Return A,B,pi
        '''



