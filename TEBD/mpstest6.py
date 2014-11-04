"""
mpstest6.py
A test of manipulating matrix product states with numpy.
2014-08-25
"""

import numpy as np
import matplotlib.pyplot as plt
from cmath import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def main():
    #test1()
    #test2()
    test3()
    #test4()
    #test5()

def test1():
    print("*** MPS tests started ***")
    (N,chi,d) = (7,10,2)
    A = randomMPS(N,chi,d)
    state = getState(A)
    state = state/np.sqrt(np.dot(np.conj(state),state))
    prod = np.dot(np.conj(state),state)
    approxA = getMPS(state,2)
    approxState = getState(approxA)
    approxProd = np.dot(np.conj(approxState),approxState)
    relErr = approxProd/prod - 1
    S = entropy(state)
    print("State total %d elements"%state.size)
    print("MPS total %d elements"%A.size)
    print("(N,chi,d) = (%d,%d,%d)"%(N,chi,d))
    print("Expected:        (%f,%f)"%polar(prod))
    print("SVD:             (%f,%f)"%polar(innerProduct(approxA,approxA)))
    print("Product:         (%f,%f)"%polar(approxProd))
    print("Relative error:  %f"%np.absolute(relErr))
    print("Entropy:         %f"%S)
    print("")
    # state = np.ones(d**N)/np.sqrt(2)**N
    # state = np.zeros(2**10)
    # state[0] = 1/np.sqrt(2)
    # state[-1] = 1/np.sqrt(2)
    state = np.random.rand(d**N)
    state = state/np.linalg.norm(state)
    
    mps = getMPS(state,4)
    print("Expected: (%f,%f)"%polar(np.inner(state,state)))
    print("MPS:      (%f,%f)"%polar(innerProduct(mps,mps)))
    print("*** MPS tests finished ***\n")

def test2():
    print("*** Started testing MPS approximation ***")
    (N,chi,d) = (5,3,2)
    A = randomMPS(N,chi,d)
    a = getState(A)
    for newChi in xrange(1,12):
        newA = getMPS(a,newChi)
        print(fidelityMPS(A,newA))
        newa = getState(newA)
        print(fidelity(a,newa))
        print(fidelity(a,a))
        print(fidelityMPS(A,A))
    print("*** Finished testing MPS approximation ***")

def test3():
    print("*** Started testing MPS ***")
    N = 5
    d = 2
    X = []
    Y = []
    Z = []
    for chi0 in xrange(1,8):
        for chi1 in xrange(1,8):
            F = 0
            for i in xrange(20):
                mps = randomMPS(N,chi0,d)
                state = getState(mps)
                newmps = getMPS(state,chi1)
                state1 = getState(newmps)
                F += fidelityMPS(mps,newmps)
            X.append(chi0)
            Y.append(chi1)
            Z.append(F/20)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
    ax.set_xlabel('chi0')
    ax.set_ylabel('chi1')
    ax.set_zlabel('fidelity')
    plt.show()
    print("*** Finished testing MPS ***")

def test4():
    print("*** Started testing fidelity ***")
    d = 2
    N = 5
    for i in xrange(10):
        mpsa = randomMPS(N,5,d)
        a = getState(mpsa)
        mpsb = getMPS(a,2)
        b = getState(mpsb)
        print(fidelity(a,b))
        print(fidelityMPS(mpsa,mpsb))
    print("*** Finished testing fidelity ***")

def test5():
    print("*** Started testing MPS ***")
    N = 5
    d = 2
    X = []
    Y = []
    Z = []
    for chi0 in xrange(1,8):
        for chi1 in xrange(1,8):
            F = 0
            for i in xrange(5):
                mps = randomMPS(N,chi0,d)
                state0 = getState(mps)
                newmps = getMPS(state0,chi1)
                state1 = getState(newmps)
                F += fidelity(state0,state1)
            X.append(chi0)
            Y.append(chi1)
            Z.append(F/20)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
    ax.set_xlabel('chi0')
    ax.set_ylabel('chi1')
    ax.set_zlabel('fidelity')
    plt.show()
    print("*** Finished testing MPS ***")

def closeness(a,b):
    return np.inner(np.conj(a),a)-np.inner(np.conj(b),b)

def correlation(A,B):
    return innerProduct(A,B)*innerProduct(B,A)/innerProduct(A,A)/innerProduct(B,B)

def fidelityMPS(A,B):
    """ Fidelity of two MPS """
    return innerProduct(A,B)*innerProduct(B,A)\
           /innerProduct(A,A)/innerProduct(B,B)

def fidelity(a,b):
    """ Fidelity of two states """
    return np.inner(np.conj(a),b)*np.inner(np.conj(b),a)\
           /np.inner(np.conj(a),a)/np.inner(np.conj(b),b)

def randomMPS(N,chi,d):
    """ Returns a random MPS given parameters N, chi, d."""
    A = []
    for i in xrange(N):
        A.append((np.random.rand(chi,d,chi)-.5)+1j*(np.random.rand(chi,d,chi)-.5))
        #A.append(np.random.rand(chi,d,chi))
        
    return np.array(A)

def bellState():
    return np.array([1,0,0,1],dtype=complex)/np.sqrt(2)

def getState(A):
    """ State vector of a MPS."""
    N = len(A)
    chi = A[0].shape[0]
    d = A[0].shape[1]
    c = A[0]
    for i in xrange(1,N):
        c = np.tensordot(c,A[i],axes=(-1,0))
    c = np.trace(c,axis1=0,axis2=-1)
    return np.reshape(c,d**N)

def getMPS(state,chi):
    """ MPS of a state."""
    d = 2 # Qubits have 2 states each
    N = int(np.log2(len(state))) # Number of qubits

    c = np.reshape(state,cShape(d,N)) # State amplitudes tensor.
    A = [] # List of N matrices of MPS, each of shape (chi,d,chi)

    # Start left end with a vector of size (d,chi)
    c = np.reshape(c,(d,d**(N-1))) # Reshape c
    (ap,sv,c) = np.linalg.svd(c) # Apply SVD
    s = np.zeros((d,chi),dtype=complex) # Construct shape of singular value matrix s
    s[:d,:d] = np.diag(sv[:chi]) # Fill s with singular values
    # Trim c or fill rest of c with zeros
    newc = np.zeros((chi,d**(N-1)),dtype=complex)
    newc[:min(chi,d**(N-1)),:] = c[:chi,:]
    c = newc
    A.append(np.tensordot(ap,s,axes=(-1,0))) # Contract and append to A

    # Sweep through the middle, creating matrix products each with
    # shape (chi,d,chi)
    for i in xrange(1,N-2):
        c = np.reshape(c,(d*chi,d**(N-i-1)))
        (ap,sv,c) = np.linalg.svd(c)
        s = np.zeros((d*chi,chi),dtype=complex)
        s[:min(chi,len(sv)),:min(chi,len(sv))] = np.diag(sv[:chi])
        A.append(np.reshape(np.dot(ap,s),(chi,d,chi)))
        newc = np.zeros((chi,d**(N-i-1)),dtype=complex)
        newc[:min(chi,len(sv)),:] = c[:chi,:]
        c = newc
    
    # Finish right end with the remaining vector
    c = np.reshape(c,(d*chi,d))
    (ap,sv,c) = np.linalg.svd(c)
    s = np.zeros((chi,d),dtype=complex)
    s[:d,:d] = np.diag(sv[:chi])
    A.append(np.reshape(ap[:chi,:],(chi,d,chi)))
    c = np.dot(s,c)
    A.append(c)

    # Fix up ends by filling first row of correctly shaped zeros with
    # end vectors such that the trace is preserved.
    start = np.zeros((chi,d,chi),dtype=complex)
    start[0,:,:] = A[0]
    A[0] = start
    finish = np.zeros((chi,d,chi),dtype=complex)
    finish[:,:,0] = A[-1]
    A[-1] = finish

    # Return MPS as numpy array with shape (N,chi,d,chi)
    return np.array(A)

def innerProduct(A,B):
    """ Inner product <A|B> using transfer matrices."""
    N = len(A)
    chiA = A.shape[1]
    chiB = B.shape[1]
    d = A.shape[2]
    
    # Take adjoint of |A> to get <A|
    A = np.conj(A)
    # Construct list of transfer matrices by contracting pairs of
    # tensors from A and B.
    transfer = []
    for i in xrange(N):
        t = np.tensordot(A[i],B[i],axes=(1,1))
        t = np.transpose(t,axes=(0,2,1,3))
        t = np.reshape(t,(chiA*chiB,chiA*chiB))
        transfer.append(t)
    # Contract the transfer matrices.
    prod = transfer[0]
    for i in xrange(1,len(transfer)):
        prod = np.tensordot(prod,transfer[i],axes=(-1,0))
    return np.trace(prod)

def operatorInner(A,U,B):
    """ Compute <A|U|B> where A,B are MPS and U is a MPO."""
    N = len(A)
    d = A.shape[2]
    chiA = A.shape[1]
    chiB = B.shape[1]
    chiU = U.shape[1]

    # Take complex conjugate of elements in A to get <A|
    A = np.conj(A)

    # Construct list of transfer matrices
    transfer = []
    for i in xrange(N):
        t = np.tensordot(A[i],U[i],axes=(1,1))
        t = np.tensordot(t,B[i],axes=(3,1))
        t = np.reshape(t,(chiA*chiA*d,chiB*chiB*d))
        transfer.append(t)

    # Take product of transfer matrices
    prod = transfer[0]
    for i in xrange(1,N):
        prod = np.tensordot(prod,transfer[i],axes=(-1,0))
    return np.trace(prod)

def getOperator(mpo):
    """ Contract MPO into matrix representation."""
    N = len(A)
    d = mpo.shape[2]
    chi = mpo.shape[1]

    prod = mpo[0]
    for i in xrange(1,N):
        prod = np.tensordot(prod,mpo[i],axes=(-1,0))
    prod = np.trace(prod,axis1=0,axis2=-1)
    permutation = tuple(range(0,2*N,2) + range(1,2*N,2))
    prod = np.transpose(prod,perutation)
    return np.reshape(prod,(d**N,d**N))

def getMPO(U,chi):
    """ Returns MPO of operator U."""
    d = 2
    N = int(np.log2(U.shape[0]))
    mpo = []
    c = np.reshape(U,tuple([i for i in xrange(2*N)]))
    permutation = []
    for i in xrange(N):
        permutation.append(i)
        permutation.append(i+N)
    c = np.transpose(U,tuple(permutation))

    c = np.reshape(c,(d**2,d**(2*(N-1))))
    [up,sv,c] = np.linalg.svd(c)
    return 0

def randomState(d,N):
    state = (np.random.rand(d**N)-.5) + (np.random.rand(d**N)-.5)*1j
    state = state/np.linalg.norm(state)
    return state

def equalDist(N):
    """ Returns state with equal amplitudes."""
    return np.ones(cShape(2,N))/np.sqrt(2)**N

def ghz(N):
    c = np.zeros(2**N)
    c[0] = 1/np.sqrt(2)
    c[-1] = 1/np.sqrt(2)
    return np.reshape(c,cShape(2,N))

def Z(N):
    sz = np.array([[1,0],[0,-1]])
    z = np.identity(2)
    for i in xrange(N):
        z = np.kron(z,sz)
    return z

def tp(factors):
    """ Returns tensor product of list of matrices."""
    prod = factors[0]
    for i in xrange(1,len(factors)):
        prod = np.kron(prod,factors)
    return prod

def cShape(d,N):
    """ Returns the shape of c tensor representation."""
    return tuple([d for i in xrange(N)])

def densityMatrix(state):
    p = np.absolute(state)
    rho = np.outer(p,p)
##    print np.linalg.det(rho)
    return rho

def entropy(state):
    """ Von Neumann Entropy of pure state by SVD. """
    c = np.reshape(state,(2,np.size(state)/2))
    d = np.linalg.svd(c)[1]
    p = np.abs(d)**2
    S = 0
    for x in p:
        if x != 0:
            S += x*np.log(x)
    return -S

def matFunction(f,A):
    """ Function of a matrix. """
    (D,P) = np.linalg.eig(A)
    return np.dot(P,np.dot(np.diag(f(D)),np.linalg.inv(P)))

if __name__ == "__main__":
    main()
