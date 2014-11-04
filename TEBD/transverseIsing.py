"""
transverseIsing.py
Use TEBD to compute ground state of transverse field Ising model.
2014-08-29
"""

import numpy as np
import scipy.linalg
from cmath import *
from mpstest15 import *

def main():
    print(pairBlock(1,1))
    a = np.random.rand(3,3)
    show(a,"a")
    show(np.linalg.eig(a)[0],"eignenvalues of a")
    ea = scipy.linalg.expm(a)
    show(ea,"e^a")
    show(np.linalg.eig(ea)[0],"eigenvalues of e^a")
    show(np.log(np.linalg.eig(ea)[0]),"log of ea")

def tebdIsing(J,h,N,t):
    """ TEBD algorithm """
    # Intiate random state
    state = randomMPSOBC(N,3,2)
    pairs = [pairBlock(J,h) for i xrange(N/2)]
    Theta = []
    for i in xrange(N/2):
        theta = np.tensordot(state[i],state[i+1],(2,0))
        theta = np.tensordot(theta,pairBlock(J,h,t),((1,2),(2,3)))
        Theta.append(theta)
    
    
def pairBlock(J,h,t):
    pairHamiltonian = -J*np.kron(pauli(1),pauli(1))-\
           (np.kron(pauli(3),pauli(0))+np.kron(pauli(0),pauli(3)))*h/2
    block = scipy.linalg.expm(-pairHamiltonian*t)
    return np.reshape(block,(2,2,2,2))

def pauli(i):
    if i == 0:
        return np.eye(2,dtype=complex)
    elif i == 1:
        return np.array([[0,1],[1,0]],dtype=complex)
    elif i == 2:
        return np.array([[0,-1j],[1j,0]],dtype=complex)
    elif i == 3:
        return np.array([[1,0],[0,-1]],dtype=complex)

if __name__ == "__main__":
    main()
