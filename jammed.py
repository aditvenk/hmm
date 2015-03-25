#! /usr/bin/py
# Author - Aditya Venkataraman <adityav@cs.wisc.edu>

import numpy as np
import sys
import pprint as pp
import hmm

def enum (**enums):
    return type('Enum', (), enums)


def main():
    # build the hmm
    '''
    In Network HMM, N = 2 (jammed, not jammed)
    M = 2 {packet OK, packet LOSS}
    pi = {0.05, 0.95} (jammed, not jammed)
    '''
    NW = enum(jammed=0, not_jammed=1)
    Ob = enum(ok=0, loss=1)

    print "state notations: "
    print "Hidden variables: jammed = 0, not_jammed = 1"
    print "Observations: ok = 0, loss = 1"
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

    n = hmm.Hmm( A_init = A_init, B_init = B_init, pi_init = pi_init)

    obs = n.generateObservations(6)
    #print obs

    # Finding the likelihood of given observations from HMM (Filtering)
    print n.filtering(obs)

if __name__ == "__main__":
    main()
