"""
tebdIsing.py
Use TEBD to compute ground state of transverse field Ising model.
2014-08-29
"""

import numpy as np
from cmath import *
from mpstest16 import *
import time

def main():
    test0()

def test0():
    (J,muB,N,t,nsteps,chi) = (0.01,10,3,0.1,100,5)
    for N in xrange(3,6):
        test2(J,muB,N,t,nsteps,chi)

def test1():
    print(getU(1,1,1))
    a = np.random.rand(3,3)
    show(a,"a")
    show(np.linalg.eig(a)[0],"eignenvalues of a")
    ea = expm(a)
    show(ea,"e^a")
    show(np.linalg.eig(ea)[0],"eigenvalues of e^a")
    show(np.log(np.linalg.eig(ea)[0]),"log of eigenvalues of e^a")
    print(s(0),"I")
    print(s(1),"X")
    print(s(2),"Y")
    print(s(3),"Z")

def test2(J,muB,N,t,nsteps,chi):
    print("\nStarting TEBD for N = %d Ising model with parameters:"%N)
    print("    (J,muB,N) = (%.4f,%.4f,%d)"%(J,muB,N))
    print("    (t,nsteps,chi) = (%f,%d,%d)"%(t,nsteps,chi))
    startTime = time.clock()
    mps = tebdIsing(J,muB,N,t,nsteps,chi)
    runTime = time.clock()-startTime
    print("Simulation completed in %f seconds"%runTime)
    gs = getStateOBC(mps)
    gs = gs/np.linalg.norm(gs) # Normalize state
##    show(gs,"Ground State")
    print("Ground state =")
    displayState(gs)
    startTime = time.clock()
    energy = getEnergyBruteForce(J,muB,N,gs).real
    print("Energy = %f, Energy per spin = %f"%(energy,energy/N))
    runTime = time.clock() - startTime
    print("Energy calculation completed in %f seconds"%runTime)

def test3():
    print(pairHamiltonianMPO())

def isingH(J,muB,N):
    """
    Full matrix representation of ising model.
    """
    pairs = np.zeros((2**N,2**N),dtype=complex)
    for i in xrange(N-1):
        pairs += pauli([1,1],[i,i+1],N)
    fields = np.zeros((2**N,2**N),dtype=complex)
    for i in xrange(N):
        fields += pauli([3],[i],N)
    return -J*pairs-muB*fields

def tebdIsing(J,muB,N,t,nsteps,chi):
    """
    Run TEBD algorithm on 1D N-spin transverse field Ising model.
    Uses open boundary conditions and imaginary time evolution.

    Parameters
    ----------
    J : float
        Pair-wise interaction energy.
    muB : float
        Magnetic energy in transverse B-field.
    N : int        
        Number of spins.
    t : float
        Timestep of each iteration.
    nsteps : int
        Number of time evolution iterations simulated.
    
    Returns
    -------
    groundState : list
        MPS representation of ground state.
    energies : (nsteps) ndarray
        Energies at each timestep.
    """
    
    # Initiate system with random MPS state.
    d = 2
    state = randomMPSOBC(N,chi,d)

    # Initilise list of energies at each iteration
    
    # Compute time evolution operators.
    U = getU(J,muB,t) # Acts on pairs of spins in middle
    Ub = getUb(J,muB,t) # Acts on only one boundary spin

    # Run iteration nstep times
    for step in xrange(nsteps):
        # First half evolution
        # Evolve first two spins
        state[0],state[1] = leftPair(state[0],state[1],U,chi,d)
        # Evolve middle spins
        for i in xrange(2,N-2,2):
            state[i],state[i+1] = middlePair(state[i],state[i+1],U,chi,d)
        # Evolve last spin pair (or single spin if odd)
##        show(state[-1].shape,"state[-1]")
        if N % 2 and N > 2: # N is odd
            state[-1] = rightSingle(state[-1],Ub)
##            print("odd")
        elif N > 2: # N is even
            state[-2],state[-1] = rightPair(state[-2],state[-1],U,chi,d)
##        show(state[-1].shape,"state[-1]")
        
        # Second half evolution
        # Evolve first spin
        state[0] = leftSingle(state[0],Ub)
        # Evolve middle spins
        for i in xrange(1,N-2,2):
            state[i],state[i+1] = middlePair(state[i],state[i+1],U,chi,d)
##        show(state[-1].shape,"state[-1]")
        state[-1] = rightSingle(state[-1],Ub)
##        show(state[-1].shape,"state[-1]")
##        # Evolve last spin (or spin pair if odd)
##        show(state[-1].shape,"state[-1]")
        if N % 2 and N > 2: # N is odd
            state[-2],state[-1] = rightPair(state[-2],state[-1],U,chi,d)
        elif N > 2: # N is even and greater than 2
            state[-1] = rightSingle(state[-1],Ub)
##        energies.append(getEnergy(state))
    return state

def middlePair(A,B,U,chi,d):
    """
    Evolve a pair of spins in middle.
    """
    lbd = A.shape[0] # Left bond dimension
    rbd = B.shape[2] # Right bond dimension
    theta = np.tensordot(A,U,axes=(1,2))
    theta = np.tensordot(theta,B,axes=((1,4),(0,1)))
    theta = np.reshape(theta,(lbd*d,rbd*d))
    (a,b) = efficientSVD(theta,chi)
    a = np.reshape(a,(lbd,d,a.shape[1]))
    b = np.reshape(b,(b.shape[0],d,rbd))
    return (a,b)

def leftPair(A,B,U,chi,d):
    """
    Evolve a pair of spins on left.
    """
    rbd = B.shape[2] # Right bond dimension
    theta = np.tensordot(A,U,axes=(0,2))
    theta = np.tensordot(theta,B,axes=((0,3),(0,1)))
    theta = np.reshape(theta,(d,d*rbd))
    (a,b) = efficientSVD(theta,chi)
    b = np.reshape(b,(b.shape[0],d,rbd))
    return (a,b)

def rightPair(A,B,U,chi,d):
    """
    Evolve a pair of spins on right.
    """
    lbd = A.shape[0] # Left bond dimension
##    show(A.shape,"A")
##    show(B.shape,"B")
##    show(U.shape,"U")
    theta = np.tensordot(A,U,axes=(1,2))
##    show(theta.shape,"A*U")
    theta = np.tensordot(theta,B,axes=((1,4),(0,1)))
##    show(theta.shape,"A*U*B")
    theta = np.reshape(theta,(lbd*d,d))
    (a,b) = efficientSVD(theta,chi)
    a = np.reshape(a,(lbd,d,a.shape[1]))
    return (a,b)

def leftSingle(A,Ub):
    """
    Evolve a single spin on left end.
    """
##    show(A.shape,"leftSingleA")
##    show(Ub.shape,"leftSingleUb")
    return np.tensordot(Ub,A,axes=(1,0))

def rightSingle(A,Ub):
    """
    Evolve a single spin on right end.
    """
    return np.tensordot(A,Ub,axes=(1,1))

def pairHamiltonianMPO():
    XX = np.kron(s(1),s(1))
    (a,b) = efficientSVD(XX,10)
    (a,b) = (np.reshape(a,(2,2,4)),np.reshape(b,(4,2,2)))
##    print(np.reshape(np.tensordot(a,b,(-1,0)),(4,4)))
    return (a,b)

def getEnergy(J,muB,N,mps):
    energy = (-J*localEnergy(mps)-muB*fieldEnergy(mps))/\
             innerProductOBC(mps,mps)
    return energy

def localEnergy(mps):
    """
    Energy of local interactions.
    """
    N = len(mps)
    hamiltonian = [np.reshape(s(1),(2,2,1))]
    hamiltonian += [np.reshape(s(1),(1,2,2,1))] * (N-2)
    hamiltonian += [np.reshape(s(1),(1,2,2))]
    oddEnergy = operatorInnerOBC(mps,hamiltonian,mps)
    evenEnergy = operatorInnerOBC(mps,hamitonian,mps)
    return oddEnergy + evenEnergy

def fieldEnergy(mps):
    """
    Energy of spins in magnetic field.
    """
    N = len(mps)
    hamiltonian = [np.reshape(s(3),(2,2,1))]
    hamiltonian += [np.reshape(s(3),(1,2,2,1))]
    hamiltonian += [np.reshape(s(3),(1,2,2))]
    return operatorInnerOBC(mps,hamiltonian,mps)

def getEnergyBruteForce(J,muB,N,state):
    """
    Energy of state by brute force with 2**N by 2**N Hamiltonian matrix.
    E = <a|H|a>.

    Parameters
    ----------
    state : (2**N,) ndarray
        State vector of system.

    Returns
    -------
    energy : complex
        Energy of the system.
    """
    return np.dot(np.conj(state),np.dot(isingH(J,muB,N),state))
    
def getUb(J,muB,t):
    """
    Time evolution operators acting on boundaries.

    Parameters
    ----------
    J : float
        Pair-wise interaction energy.
    muB : float
        Magnetic energy of each spin with dipole moment mu in field B.
    t : float
        Timestep of each iteration.

    Returns
    -------
    startU : (2,2) ndarray
        Non-unitary evolution operator acting on single qubit at boundary.
    """
    return expm(-muB/2*s(1)*t)

def getU(J,muB,t):
    """
    Time evolution operator acting on 2 spins.

    Parameters
    ----------
    J : float
        Pair-wise interaction energy.
    muB : float
        Magnetic energy of each spin with dipole moment mu in field B.
    t : float
        Timestep of each iteration.

    Returns
    -------
    U : (2,2,2,2) ndarray
        Non-unitary time evolution operator.        
    """
    hamiltonian = -J*np.kron(s(3),s(3))-\
           (np.kron(s(1),s(0))+np.kron(s(0),s(1)))*muB/2
    U = expm(-hamiltonian*t)
    return np.reshape(U,(2,2,2,2))

def s(i):
    """
    The Pauli Matrices I,X,Y,Z.
    s(0) = I, s(1) = X, s(2) = Y, s(4) = Z.

    Parameters
    ----------
    i : index of Pauli Matrix.

    Returns
    -------
    s : (2,2) ndarray
        Pauli matrix with complex elements.
    """
    if i == 0:
        return np.eye(2,dtype=complex)
    elif i == 1:
        return np.array([[0,1],[1,0]],dtype=complex)
    elif i == 2:
        return np.array([[0,-1j],[1j,0]],dtype=complex)
    elif i == 3:
        return np.array([[1,0],[0,-1]],dtype=complex)

def pauli(paulis,positions,N):
    mat = 1+0j
    identity = s(0)
    for i in xrange(N):
        if i in positions:
            mat = np.kron(mat,s(paulis[positions.index(i)]))
        else:
            mat = np.kron(mat,identity)
    return mat

def expm(A):
    """
    Matrix exponential by eigen-decomposition.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix to be exponentiated

    Returns
    -------
    expm : (N, N) ndarray
        Matrix exponential of A

    """
    s,vr = np.linalg.eig(A)
    vri = np.linalg.inv(vr)
    return np.dot(np.dot(vr,np.diag(np.exp(s))),vri)

def displayState(state):
    display = ""
    N = int(np.log2(state.size))
    for i in xrange(state.size):
        display += " + %.4f*exp(%d"%(abs(state[i]),np.degrees(phase(state[i])))
        display += u'\u00b0' + "i)|" + format(i,"0"+str(N)+"b") + ">"
        if i % 2:
            display += "\n"
    print(display[:-1])
    
if __name__ == "__main__":
    main()
