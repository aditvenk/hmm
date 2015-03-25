import pprint as pp

class Hmm():
    ''' an HMM is characterized by three things <A,B,pi>
    A = Transition matrix, NxN where N = no. of possible states for X (hidden variable)
    A = {aij} where aij = P (qj at t+1 | qi at t)
    B = {bj(k)}, NxM sensor observation matrix. M = no. of possible observations for given x.
    bj(k) = P (observation k at t | state qj at t)
    pi = Initial state distribution
    '''
    pi = {}
    A = {}
    B = {}

    def __init__ (self, A_init, B_init, pi_init):
        # verify the inputs
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


        A = A_init
        B = B_init
        pi = pi_init

        pp.pprint("Initial A:")
        pp.pprint(A)
        pp.pprint("Initial B:")
        pp.pprint(B)
        pp.pprint("Initial C:")
        pp.pprint(pi)

    def generateObservations (self, num_obs):
        ''' generate num_obs observations of this HMM
        '''

