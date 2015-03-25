#! /usr/bin/py
import numpy as np
import sys
import pprint as pp

def enum (**enums):
    return type('Enum', (), enums)

NW = enum(jammed=0, not_jammed=1)
Ob = enum(ok=0, loss=1)


class network_hmm():
    ''' an HMM is characterized by three things <A,B,pi>
    A = Transition matrix, NxN where N = no. of possible states for X (hidden variable)
    A = {aij} where aij = P (qj at t+1 | qi at t)
    B = {bj(k)}, NxM sensor observation matrix. M = no. of possible observations for given x.
    bj(k) = P (observation k at t | state qj at t)
    pi = Initial state distribution

    In Network HMM, N = 2 (jammed, not jammed)
    M = 2 {packet OK, packet LOSS}
    pi = {0.05, 0.95} (jammed, not jammed)
    '''
    pi = {}
    A = {}
    B = {}

    def __init__ (self, A_init, B_init, pi_init):
        A = A_init
        B = B_init
        pi = pi_init

        pp.pprint("Initial A:")
        pp.pprint(A)
        pp.pprint("Initial B:")
        pp.pprint(B)
        pp.pprint("Initial C:")
        pp.pprint(pi)


def main():
    # build the hmm
    A_init = {}
    A_init[NW.jammed] = {}
    A_init[NW.jammed][NW.jammed] = 0.9
    A_init[NW.jammed][NW.not_jammed] = 0.1
    A_init[NW.not_jammed] = {}
    A_init[NW.not_jammed][NW.jammed] = 0.05
    A_init[NW.not_jammed][NW.not_jammed] = 0.95

    B_init = {}
    B_init[NW.jammed] = {}
    B_init[NW.not_jammed] = {}
    B_init[NW.jammed][Ob.ok] = 0.2
    B_init[NW.jammed][Ob.loss] = 0.8
    B_init[NW.not_jammed][Ob.ok] = 0.95
    B_init[NW.not_jammed][Ob.loss] = 0.05

    pi_init = {}
    pi_init[NW.jammed] = 0.05
    pi_init[NW.not_jammed] = 0.95

    n = network_hmm( A_init = A_init, B_init = B_init, pi_init = pi_init)


if __name__ == "__main__":
    main()