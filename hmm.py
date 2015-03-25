import numpy as np
import pprint as pp
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

        pp.pprint("Initial A:")
        pp.pprint(self.A)
        pp.pprint("Initial B:")
        pp.pprint(self.B)
        pp.pprint("Initial C:")
        pp.pprint(self.pi)

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
        return O
