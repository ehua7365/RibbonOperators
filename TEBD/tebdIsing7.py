"""
tebdIsing7.py
Use TEBD to compute ground state of transverse field Ising model.
2014-09-29
"""

import numpy as np
from cmath import *
from mpstest17 import *
import matplotlib.pyplot as plt
import time

def main():
    test0()

def test0():
    (J,hx,hz) = (1,1,0)
    (nsteps,chi) = (32,5)
    Nvalues = []
    energies = []
    tebdTimes = []
    energyTimes = []
    runTime = time.clock()
    for N in xrange(3,32):
        Nvalues.append(N)
        energy,t1,t2 = test2(J,hx,hz,N,8.0/nsteps,nsteps,chi)
        energies.append(energy)
        tebdTimes.append(t1)
        energyTimes.append(t2)
    runTime = time.clock() - runTime
    print("%d TEBD simulations in %f seconds"%(len(Nvalues),runTime))
    plotFit(Nvalues,energies,10,r"$N$",r"$\left\langle\psi\right|H\left|\psi\right\rangle$")
    plotFit(np.log(Nvalues)[5:],np.log(tebdTimes)[5:],1,r"$\log(N)$",r"$\log(t)$")
    plotFit(np.log(Nvalues)[5:],np.log(energyTimes)[5:],1,r"$\log(N)$",r"$\log(t)$")

def plotFit(x,y,degree,xname,yname):
    plt.plot(x,y,'+')
    z = np.polyfit(x,y,degree)
    print(z)
    p = np.poly1d(z)
    xp = np.linspace(min(x),max(x),100)
    plt.plot(xp,p(xp),'-')
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title("Plot of "+yname+" vs. "+xname)
    plt.show()

def test2(J,hx,hz,N,t,nsteps,chi):
    print("\nStarting TEBD for N = %d Ising model with parameters:"%N)
    print("    (J,hx,hz,N) = (%.4f,%.4f,%.4f,%d)"%(J,hx,hz,N))
    print("    (t,nsteps,chi) = (%f,%d,%d)"%(t,nsteps,chi))
##    startTime = time.clock()
    tebdTime = time.clock()
    mps = tebdIsing(J,hx,hz,N,t,nsteps,chi)
    tebdTime = time.clock() - tebdTime
##    runTime = time.clock()-startTime
##    gs = getStateOBC(mps)
##    gs = gs/np.linalg.norm(gs) # Normalize state
##    show(gs,"Ground State")
##    print("Ground state =")
##    displayState(gs)
##    startTime = time.clock()
##    energyBF = getEnergyBruteForce(J,hx,hz,N,gs).real
##    print("Energy = %f, Energy per spin = %f"%(energyBF,energyBF/N))
##    runTime = time.clock() - startTime
##    print("Brute force energy computed in %f seconds"%runTime)
    energyTime = time.clock()
    energy = getEnergy(J,hx,hz,N,mps).real
    print("Energy = %f, Energy per spin = %f"%(energy,energy/N))
    energyTime = time.clock() - energyTime
    print("MPO energy computed in %f seconds"%energyTime)
    return energy/N,tebdTime,energyTime

def isingH(J,hx,hz,N):
    """
    Full matrix representation of ising model hamiltonian.
    """
    X = 1
    Z = 3
    pairs = np.zeros((2**N,2**N),dtype=complex)
    for i in xrange(N-1):
        pairs += pauli([Z,Z],[i,i+1],N)
    fieldsx = np.zeros((2**N,2**N),dtype=complex)
    for i in xrange(N):
        fieldsx += pauli([X],[i],N)
    fieldsz = np.zeros((2**N,2**N),dtype=complex)
    for i in xrange(N):
        fieldsz += pauli([Z],[i],N)
    return -J*pairs-hx*fieldsx-hz*fieldsz

def tebdIsing(J,hx,hz,N,t,nsteps,chi):
    """
    Run TEBD algorithm on 1D N-spin transverse field Ising model.
    Uses open boundary conditions and imaginary time evolution.

    Parameters
    ----------
    J : float
        Pair-wise interaction energy.
    hx : float
        Magnetic energy due to x-direction field.
    hz : float
        Magnetic energy due to z-direction field.
    N : int        
        Number of spins.
    t : float
        Timestep of each iteration.
    nsteps : int
        Number of time evolution iterations simulated.
    chi : int
        Upper bound for MPS bond dimensions.
    
    Returns
    -------
    groundState : list
        MPS representation of ground state.
    energies : (nsteps) ndarray
        Energies at each timestep.
    """
    
    # Initiate system with random MPS state.
    d = 2
    state = randomMPSOBC(N,chi,d,real=True)
##    state = allEqual(N)
##    print("Intial state")
##    displayState(getStateOBC(state))
    print("TEBD Algorithm Started")
    print("(J,hx,hz,N) = (%f,%f,%f,%d)"%(J,hx,hz,N))
    print("(t,nsteps) = (%f,%d)"%(t,nsteps))
    startTime = time.clock()
    # Initilise list of energies at each iteration
    
    # Compute time evolution operators.
    U = getU(J,hx,hz,t) # Acts on pairs of spins in middle
    Ub = getUb(J,hx,hz,t) # Acts on only one boundary spin

    energies = []
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
##        if innerProductOBC(state,state) > 1e100:
##            state = [s/1e10 for s in state]
##        show(sum([np.sum(np.abs(s)) for s in state]),"sum state")
##        show(innerProductOBC(state,state),"<a|a>")
##        energies.append(getEnergy(J,hx,hz,N,state).real)
    print("TEBD finished in %f seconds"%(time.clock()-startTime))
##    plt.plot(energies)
##    plt.show()
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
    Z = s(3)
    ZZ = np.kron(Z,Z)
    (a,b) = efficientSVD(ZZ,10)
    (a,b) = (np.reshape(a,(2,2,4)),np.reshape(b,(4,2,2)))
##    print(np.reshape(np.tensordot(a,b,(-1,0)),(4,4)))
    return (a,b)

def getEnergy(J,hx,hz,N,mps):
    """
    Energy <a|H|a> of a state |a> by transfer matrices.

    Parameters
    ----------
    J : float
        Coupling constant.
    hx : float
        Magnetic energy due to x-direction field.
    hz : float
        Magnetic energy due to z-direction field.
    N : int
        Number of spins.
    mps : list
        MPS representation of state.
    """
    # Local energy
    I = np.reshape(s(0),(1,2,2,1))
    X = np.reshape(s(1),(1,2,2,1))
    Z = np.reshape(s(3),(1,2,2,1))
    localEnergy = 0
    for i in xrange(0,N-1):
        hamiltonian = [I for x in xrange(N)]
##        show((hamiltonian),"Hamiltonian MPO")
##        show(N,"N")
##        show(i,"i")
##        show(hamiltonian[i],"hamiltonian[i]")
##        show(hamiltonian[i+1],"hamiltonian[i+1]")
        hamiltonian[i] = Z
        hamiltonian[i+1] = Z
        hamiltonian[0] = np.reshape(hamiltonian[0],(2,2,1))
        hamiltonian[-1] = np.reshape(hamiltonian[-1],(1,2,2))
        localEnergy += operatorInnerOBC(mps,hamiltonian,mps)
    # x-direction magnetic energy
    xFieldEnergy = 0
    for i in xrange(N):
        hamiltonian = [I for x in xrange(N)]
        hamiltonian[i] = X
        hamiltonian[0] = np.reshape(hamiltonian[0],(2,2,1))
        hamiltonian[-1] = np.reshape(hamiltonian[-1],(1,2,2))
        xFieldEnergy += operatorInnerOBC(mps,hamiltonian,mps)
    # x-direction magnetic energy
    zFieldEnergy = 0
    for i in xrange(N):
        hamiltonian = [I for x in xrange(N)]
        hamiltonian[i] = Z
        hamiltonian[0] = np.reshape(hamiltonian[0],(2,2,1))
        hamiltonian[-1] = np.reshape(hamiltonian[-1],(1,2,2))
        zFieldEnergy += operatorInnerOBC(mps,hamiltonian,mps)
    return (-J*localEnergy-hx*xFieldEnergy-hz*zFieldEnergy)/\
             innerProductOBC(mps,mps)

def getEnergyBruteForce(J,hx,hz,N,state):
    """
    Energy of state vector using brute force.
    E = <a|H|a>.

    Parameters
    ----------
    J : float
        Coupling constant.
    hx : float
        Magnetic energy due to x-direction field.
    hz : float
        Magnetic energy due to z-direction field.
    N : int
        Number of spins.
    state : (2**N,) ndarray
        State vector of system.

    Returns
    -------
    energy : complex
        Energy of the system.
    """
    H = isingH(J,hx,hz,N)
    (energies,states) = np.linalg.eig(H)
    minEnergy = min(energies)
    i = list(energies).index(minEnergy)
    print("Sorted Eigenvalues of Hamiltonian:")
    print(np.round(np.sort(energies.real),decimals=4))
    gs = states[:,i]
    gs = np.reshape(gs,gs.size)
    print("Brute force ground state")
    displayState(gs)
    print("Brute force hamiltonian eigendecomposition energy")
    print(energies[i].real)
    return np.dot(np.conj(state),np.dot(H,state))
    
def getUb(J,hx,hz,t):
    """
    Time evolution operators acting on boundaries.

    Parameters
    ----------
    J : float
        Pair-wise interaction energy.
    hx : float
        Magnetic energy in x-direction.
    hz : float
        Magnetic energy in z-direction.
    t : float
        Timestep of each iteration.

    Returns
    -------
    startU : (2,2) ndarray
        Non-unitary evolution operator acting on single qubit at boundary.
    """
    X = s(1)
    Z = s(3)
    hamiltonian = -hx*X*0.5 - hz*Z*0.5
    return expm(-hamiltonian*t)

def getU(J,hx,hz,t):
    """
    Time evolution operator acting on 2 spins.

    Parameters
    ----------
    J : float
        Pair-wise interaction energy.
    hx : float
        Magnetic energy of each spin with dipole moment mu in field B.
    t : float
        Timestep of each iteration.

    Returns
    -------
    U : (2,2,2,2) ndarray
        Non-unitary time evolution operator.        
    """
    I = s(0)
    X = s(1)
    Z = s(3)
    hamiltonian = -J*np.kron(Z,Z)\
           -(np.kron(X,I)+np.kron(I,X))*hx*0.5\
           -(np.kron(Z,I)+np.kron(I,Z))*hz*0.5
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
    """
    N-qubit Pauli operator given paulis and positions.

    Parameters
    ----------
    paulis : list
        List of integers denoting type of Pauli matrix at each site.
    positions : list
        List of positions for each corresponding element in paulis.
    N : int
        Total number of sites.

    Returns
    -------
    pauli : (2**N,2**N) ndarray
        Pauli operator as 2^N by 2^N matrix.
    """
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
    """
    Print state vector as readable form in Dirac notation.

    Parameters
    ----------
    state : (2**N,) ndarray
        State vector of state to be displayed.

    Returns
    -------
        None
    """
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
