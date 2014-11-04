"""
mpstest16.py
A test of manipulating matrix product states with numpy.
There is an upper bound chi for bond dimensions in getMPSOBC()
Variable bond dimension.
2014-08-29
"""

import numpy as np
import matplotlib.pyplot as plt
from cmath import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def main():
    test2()

def test1():
    """ Test functions just for a simple case.
    """
    mps0 = randomMPSOBC(5,3,2)
    state0 = getStateOBC(mps0)
    mps1 = getMPSOBC(state0,4)
    state1 = getStateOBC(mps1)
    print("Test completed. State Fidelity = %f. MPS Fidelity = %f"
          %(np.absolute(fidelity(state0,state1)),
            np.absolute(fidelityMPS(mps0,mps1))))

def test2():
    """
    Test inner product functionality.
    """
    (N,chi,d) = (5,5,2)
    mpsA = randomMPSOBC(N,chi,d)
    mpsB = randomMPSOBC(N,chi,d)
    mpoU = randomMPOOBC(N,chi,d)
    a = getStateOBC(mpsA)
    b = getStateOBC(mpsB)
    U = getOperatorOBC(mpoU)
    show(np.dot(np.conj(a),np.dot(U,b))*1e6,"Brute Force <a|U|b>")
    show(operatorInnerOBC(mpsA,mpoU,mpsB)*1e6,"Transfer <a|U|b>")
    show(np.dot(np.conj(a),b)*1e6,"Expected <a|I|b>")
    show(operatorInnerOBC(mpsA,trivialMPO(N),mpsB)*1e6,"Transfer <a|I|b>")
    show(np.dot(np.conj(a),np.dot(getOperatorOBC(trivialMPO(N))\
                                        ,b))*1e6,"Brute Force <a|I|b>")
    show(innerProductOBC(mpsA,mpsB)*1e6,"Transfer <a|b>")
##    show(np.dot(np.conj(a),a)*1e6,"Expected <a|a>")
##    show(innerProductOBC(mpsA,mpsA)*1e6,"Transfer <a|a>")
##    show(np.dot(np.conj(b),b)*1e6,"Expected <b|b>")
##    show(innerProductOBC(mpsB,mpsB)*1e6,"Transfer <b|b>")

    XX = np.kron(s(1),s(1))
    (a,b) = efficientSVD(XX,10)
    (a,b) = (np.reshape(a,(2,2,4)),np.reshape(b,(4,2,2)))
    mpoXX = [a,b]
    show(getOperatorOBC(mpoXX),"XX")

    show(getOperatorOBC(allXMPO(N)),"All X")

    show(getOperatorOBC(trivialMPO(N)),"Trivial MPO")

def test3():
    """ Test MPS conversion functions by computing fidelity between
        generated MPS and orginal, with new and old bond dimensions
        chi0 and chi1 varied.
    """
    print("*** Started testing MPS ***")
    N = 8
    d = 2
    nTrials = 10
    # Points to plot on 3d graph
    (X,Y,Z) = ([],[],[])
    chi0min,chi0max = 1,10
    chi1min,chi1max = 1,10
    for chi0 in xrange(chi0min,chi0max+1):
        for chi1 in xrange(chi1min,chi1max+1):
            F = 0
            # Run random test for 20 points and take average fidelity
            for i in xrange(nTrials):
                mps0 = randomMPSOBC(N,chi0,d) # Make random MPS
                state0 = getStateOBC(mps0) # Convert to state
                mps1 = getMPSOBC(state0,chi1) # Convert back to MPS with new bond dimension
                state1 = getStateOBC(mps1) # Convert back to state
                F += np.absolute(fidelityMPS(mps0,mps1)) # Compute fidelity and add to sum
                # F += fidelity(state0,state1) # Uncomment this to try with vectors
            X.append(chi0)
            Y.append(chi1)
            Z.append(F/nTrials)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    minZ = min(Z)
    show(X.shape,"X.shape")
    show(Y.shape,"Y.shape")
    show(Z.shape,"Z.shape")
    # Plot the surface
    meshShape = (chi0max-chi0min+1,chi1max-chi1min+1)
    X = np.reshape(X,meshShape)
    Y = np.reshape(Y,meshShape)
    Z = np.reshape(Z,meshShape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False,alpha=0.5)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=minZ, cmap=cm.coolwarm)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, aspect=5)

    ax.set_xlabel('chi0')
    ax.set_ylabel('chi1')
    ax.set_zlabel('fidelity')
    plt.show()
    print("*** Finished testing MPS ***")

def fidelityMPS(A,B):
    """
    Fidelity of two MPS representations
    f = <A|B><B|A>/(<A|A><B|B>).

    Parameters
    ----------
    A : list
        MPS representation of |A>.
    B : list
        MPS representation of |B>.

    Returns
    -------
    fidelity : complex
        Fidelity of two states as complex number. Should be real.
    """
    return innerProductOBC(A,B)*innerProductOBC(B,A)\
           /innerProductOBC(A,A)/innerProductOBC(B,B)

def fidelity(a,b):
    """
    Fidelity of two state vectors
    f = <a|b><b|a>/(<a|a><b|b>).

    Parameters
    ----------
    a : (n,) ndarray
        State vector of |a>.
    b : (n,) ndarray
        State vector of |b>.

    Returns
    -------
    fidelity : complex
        Fidelity of two states as complex number. Should be real.
    """
    return np.inner(np.conj(a),b)*np.inner(np.conj(b),a)\
           /np.inner(np.conj(a),a)/np.inner(np.conj(b),b)

def randomMPSOBC(N,chi,d):
    """ Random normalised MPS with OBC.

    Parameters
    ----------
    N : int
        Number of qudits.
    chi: int
        Bond dimension of between each pair of matrix product tensors.
    d : int
        Number of states of each qudit.

    Returns
    -------
    mpsobc : list
        List of matrix product tensors each with shape (chi,d,chi) except
        for the first and last which have shapes (d,chi) and (chi,d)
        respectively.
    """
    A = [randomComplex((d,chi))]
    for i in xrange(N-2):
        A.append(randomComplex((chi,d,chi)))
    A.append(randomComplex((chi,d)))
    return A

def getStateOBC(A):
    """
    State vector of MPS with open boundary conditions.

    Parameters
    ----------
    A : list
        MPS representation with open boundary conditions as list of
        N ndarrays each with shape upper-bounded by (chi,d,chi)

    Returns:
    state : (d^N,) ndarray
        State vector of MPS in computational basis.
    """
    N = len(A) # Number of spins
    c = A[0]
    for i in xrange(1,N):
        c = np.tensordot(c,A[i],axes=(-1,0))
    return np.reshape(c,c.size)

def getMPSOBC(state,chi):
    """
    Matrix product state representation of a state with bond
    dimension chi and open boundary conditions.

    Parameters
    ----------
    state : (2^N,) ndarray
        State vector of N-qubit system.
    chi : int
        Upper bound for the bond dimension between each qubit.

    Returns
    -------
    mpsobc : list
        List of matrix products as ndarrays each with maximum shape
        (chi,2,chi) except for the first and last elements which have
        maximum shape (2,chi) and (chi,2) respectively due to having
        open boundary conditions.
    """
    d = 2 # Qubits have 2 states each
    N = int(np.log2(len(state))) # Number of qubits

    c = np.reshape(state,cShape(d,N)) # State amplitudes tensor c.
    A = [] # List of N matrices of MPS, each of shape (chi,d,chi)

    # Start left end with a vector of size (d,chi)
    c = np.reshape(c,(d,d**(N-1))) # Reshape c
    (ap,c) = efficientSVD(c,chi)
    A.append(ap) # Contract and append to A

    # Sweep through the middle, creating matrix products each with
    # shape (chi,d,chi)
    for i in xrange(1,N-2):
        c = np.reshape(c,(d*A[-1].shape[-1],c.size/(d*A[-1].shape[-1])))
        (a,c) = efficientSVD(c,chi)
        a = np.reshape(a,(A[-1].shape[-1],d,c.shape[0]))
        A.append(a)
    
    # Finish right end with the remaining vector
    c = np.reshape(c,(c.size/d,d))
    (a,c) = efficientSVD(c,chi)
    a = np.reshape(a,(A[-1].shape[-1],d,c.shape[0]))
    A.append(a)
    A.append(c)

    return A

def show(a,name):
    """ Convenient space-saving print function.
    """
    print(name)
    print(np.round(np.absolute(a),2))

def printIntermediate(A,c,state,d,N):
    """ Testing method which compares contracted state from matrix
        products produced so far with original state
    """
    print("Printing intermediate with %d elements in A."%len(A))
    prod = A[0]
    for i in xrange(1,len(A)):
        prod = np.tensordot(prod,A[i],axes=(-1,0))
    prod = np.tensordot(prod,c,axes=(-1,0))
    prod = np.reshape(prod,cShape(d,N))
    fid = fidelity(state,np.reshape(prod,d**N))
    dif = (np.sum(np.absolute(prod-np.reshape(state,cShape(d,N)))))
    print("Difference = %f; fidelity = %f;"%(dif,np.absolute(fid)))

def innerProductOBC(mpsA,mpsB):
    """
    Inner product <A|B> using transfer matrices,
    where A and B are MPS representations of }A> and }B>
    with open boundary conditions (OBC).

    Parameters
    ----------
    mpsA : list
        MPS representation of state |A>.
    mpsB : list
        MPS representation of state |B>.

    Returns
    -------
    inner : complex
        Inner product <A|B> of two states.
    """
    # Take adjoint of |A> to get <A|
    A = []
    for a in mpsA:
        A.append(np.conj(a))
    
    B = mpsB
    N = len(A) # Number of qubits
    d = A[1].shape[1] # d = 2 for qubits

    # Construct list of transfer matrices by contracting pairs of
    # tensors from A and B.
    transfer = []
    t = np.tensordot(A[0],B[0],axes=(0,0))
    t = np.reshape(t,A[0].shape[1]*B[0].shape[1])
    transfer.append(t)
    for i in xrange(1,N-1):
        t = np.tensordot(A[i],B[i],axes=(1,1))
        t = np.transpose(t,axes=(0,2,1,3))
        t = np.reshape(t,(A[i].shape[0]*B[i].shape[0],
                          A[i].shape[2]*B[i].shape[2]))
        transfer.append(t)
    t = np.tensordot(A[N-1],B[N-1],axes=(1,1))
    t = np.reshape(t,A[N-1].shape[0]*B[N-1].shape[0])
    transfer.append(t)
    # Contract the transfer matrices.
    prod = transfer[0]
    for i in xrange(1,N-1):
        prod = np.dot(prod,transfer[i])
    prod = np.dot(prod,transfer[N-1])
    return prod

def randomState(d,N):
    """ Random N-qudit state vector.
    Parameters
    ----------
    d : int
        Number of states in each qudit.
    N : int
        Number of qudits.

    Returns
    -------
    state : (d**N,) ndarray
        Random state vector.
    """
    state = (np.random.rand(d**N)-.5) + (np.random.rand(d**N)-.5)*1j
    state = state/np.linalg.norm(state)
    return state

def efficientSVD(A,chi):
    """
    Efficient form of SVD.
    Performs Schmidt Decomposition with maximum bond dimension chi.

    Parameters
    ----------
    A : (n,m) ndarray
        Matrix to be decomposed.
    chi : int
        Upper bound for bond dimension.

    Returns
    -------
    (U,V) : 2-tuple
        Tuple of two nd-arrays such that A = UV approximately or exactly
        depending on chi.
    """
    (m,n) = A.shape
    (a,sv,b) = np.linalg.svd(A)
    s = np.diag(sv[:min(chi,m)])
    return (np.dot(a[:,:min(chi,n)],s),b[:min(chi,m),:])

def cShape(d,N):
    """ Returns the shape of c tensor representation.
        I.e. simply just (d,d,...,d) N times.
    """
    return tuple([d for i in xrange(N)])

def randomComplex(size):
    """ Random normalized array of complex numbers.
    """
    sample = (np.random.random_sample(size)-.5)\
             +1j*(np.random.random_sample(size)-.5)
    return sample/np.linalg.norm(sample)

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

def allXMPO(N):
    mpo = [np.reshape(s(1),(2,2,1))]
    for i in xrange(1,N-1):
        mpo.append(np.reshape(s(1),(1,2,2,1)))
    mpo.append(np.reshape(s(1),(1,2,2)))
    return mpo

def trivialMPO(N):
    mpo = [np.reshape(s(0),(2,2,1))]
    for i in xrange(1,N-1):
        mpo.append(np.reshape(s(0),(1,2,2,1)))
    mpo.append(np.reshape(s(0),(1,2,2)))
    return mpo

def randomMPOOBC(N,chi,d):
    mpo = [randomComplex((d,d,chi))]
    for i in xrange(1,N-1):
        mpo.append(randomComplex((chi,d,d,chi)))
    mpo.append(randomComplex((chi,d,d)))
    return mpo

def getOperatorOBC(mpo):
    N = len(mpo)
    U = mpo[0]
    for i in xrange(1,N):
##        show(U.shape,"U at i = %d"%i)
        U = np.tensordot(U,mpo[i],axes=(-1,0))
    # Permutation of indices for transpose
##    p = [[i,i+N] for i in xrange(N)]
##    p = [item for sublist in p for item in sublist]
##    show(N,"N")
##    show(U.shape,"U.shape")
##    show(p,"p")
##    p = inversePermutation(p)
    p = [i for i in xrange(0,2*N,2)] + [i for i in xrange(1,2*N,2)]
    U = np.transpose(U,axes=p)
    dim = int(np.sqrt(U.size))
    U = np.reshape(U,(dim,dim))
    return U

def inversePermutation(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def operatorInnerOBC(mpsA,mpoU,mpsB):
    """
    Compute amplitude <A|U|B> using transfer matrices,
    where mpsA and mpsB are MPS representations of }A> and }B>
    with open boundary conditions (OBC) and mpoU is MPO representation
    of operator.

    Parameters
    ----------
    mpsA : list
        MPS representation of state |A>.
    mpoU : list
        MPO representation of operator U.
    mpsB : list
        MPS representation of state |B>.

    Returns
    -------
    product : complex
        Inner product <A|B> of two states.
    """
    # Take adjoint of |A> to get <A|
    A = []
    for a in mpsA:
        A.append(np.conj(a))

    B = mpsB
    N = len(A) # Number of qubits
    d = A[1].shape[1] # d = 2 for qubits

    U = mpoU

    # Construct list of transfer matrices by contracting pairs of
    # tensors from A and B.
    transfer = []
    # Left boundary case
    t = np.tensordot(A[0],U[0],axes=(0,0))
    t = np.tensordot(t,B[0],axes=(1,0))
    transfer.append(t)
    # Middle cases
    for i in xrange(1,N-1):
        t = np.tensordot(A[i],U[i],axes=(1,1))
        t = np.tensordot(t,B[i],axes=(3,1))
##        t = np.transpose(t,axes=(0,3,1,4,2,5))
        t = np.transpose(t,axes=(0,2,4,1,3,5))
        transfer.append(t)
    # Right boundary case
    t = np.tensordot(A[N-1],U[N-1],axes=(1,1))
    t = np.tensordot(t,B[N-1],axes=(2,1))
    transfer.append(t)
    # Contract the transfer matrices.
    prod = transfer[0]
    for i in xrange(1,N):
        prod = np.tensordot(prod,transfer[i],axes=((0,1,2),(0,1,2)))
    return prod

if __name__ == "__main__":
    main()
