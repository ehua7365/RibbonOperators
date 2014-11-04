"""
mpstest3.py
A test of manipulating matrix product states with numpy.
2014-08-16
"""

import numpy as np
from cmath import *

def main():
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

    print("*** Started testing MPS approximation ***")
    print("*** Finished testing MPS approximation ***")

def randomMPS(N,chi,d):
    """ Returns a random MPS given parameters N, chi, d."""
    A = []
    for i in xrange(N):
        A.append(np.random.rand(chi,d,chi)+1j*np.random.rand(chi,d,chi))
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
    A = [] # List of matrix products of the MPS

    # Start left end with a vector of size (d,chi)
    c = np.reshape(c,(d,d**(N-1))) # Reshape c
    (ap,sv,c) = np.linalg.svd(c) # Apply SVD
    s = np.zeros((d,chi)) # Construct shape of singular value matrix s
    s[:d,:d] = np.diag(sv) # Fill s with singular values
    A.append(np.tensordot(ap,s,axes=(-1,0))) # Contract and append to A
    c = c[:chi,:] # Trim remainng c

    # Sweep through the middle, creating matrix products each with
    # shape (chi,d,chi)
    for i in xrange(1,N-2):
        c = np.reshape(c,(d*chi,d**(N-i-1)))
        (ap,sv,c) = np.linalg.svd(c)
        s = np.zeros((d*chi,chi))
        s[:chi,:chi] = np.diag(sv[:chi])
        A.append(np.reshape(np.dot(ap,s),(chi,d,chi)))
        c = c[:chi,:]
    
    # Finish right end with the remaining vector
    c = np.reshape(c,(d*chi,d))
    (ap,sv,c) = np.linalg.svd(c)
    s = np.zeros((chi,d))
    s[:d,:d] = np.diag(sv)
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
    for i in xrange(1,len(transfer)):
        prod = np.tensordot(prod,transfer[i],axes=(-1,0))
    A = np.conj(A)
    return np.trace(prod)

def getOperator(mpo):
    """ Contract MPO into matrix representation."""
    N = len(A)
    d = mpo.shape[2]
    return 0

def getMPO(O):
    """ Returns MPO of operator O."""
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
    return -np.sum(np.log(p**p))

##def entropy(rho):
##    """ Entropy of density matrix. """
##    eigenvalues = np.linalg.eig(rho)[0]
####    return -np.trace(np.dot(rho,matFunction(np.log,rho)))
####    return -np.dot(eigenvalues,log(eigenvalues))
##    return -np.sum(np.log(eigenvalues**eigenvalues))

##def log(x):
##    length = len(x)
##    ret = np.zeros(length,dtype=complex)
##    for i in xrange(length):
##        if np.absolute(x[i]) > 1e-9:
##            ret[i] = np.log(x[i])
##    return ret

def matFunction(f,A):
    """ Function of a matrix. """
    (D,P) = np.linalg.eig(A)
    return np.dot(P,np.dot(np.diag(f(D)),np.linalg.inv(P)))

if __name__ == "__main__":
    main()
