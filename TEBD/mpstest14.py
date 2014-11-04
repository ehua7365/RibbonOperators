"""
mpstest14.py
A test of manipulating matrix product states with numpy.
There is no upper bound chi for getMPSOBC()
2014-08-29
"""

import numpy as np
import matplotlib.pyplot as plt
from cmath import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def main():
    #state0 = np.array(range(2**10))
##    state0 = np.random.rand(2**4)
##    state0 = state0/np.linalg.norm(state0)
    mps0 = randomMPSOBC(5,3,2)
    state0 = getStateOBC(mps0)
    mps1 = getMPSOBC(state0,4)
    state1 = getStateOBC(mps1)
    print("Test completed. State Fidelity = %f. MPS Fidelity = %f"
          %(np.absolute(fidelity(state0,state1)),
            np.absolute(fidelityMPS(mps0,mps1))))

def fidelityMPS(A,B):
    """ Fidelity of two MPS representations
        f = <A|B><B|A>/(<A|A><B|B>).
    """
    return innerProductOBC(A,B)*innerProductOBC(B,A)\
           /innerProductOBC(A,A)/innerProductOBC(B,B)

def fidelity(a,b):
    """ Fidelity of two state vectors
        f = <a|b><b|a>/(<a|a><b|b>).
    """
    return np.inner(np.conj(a),b)*np.inner(np.conj(b),a)\
           /np.inner(np.conj(a),a)/np.inner(np.conj(b),b)

def randomMPSOBC(N,chi,d):
    """ Returns a random MPS given parameters N, chi, d."""
    A = [(np.random.rand(d,chi)-.5)
         +1j*(np.random.rand(d,chi)-.5)]
    for i in xrange(N):
        # Each real part of each value varies between -0.5 and 0.5.
        A.append((np.random.rand(chi,d,chi)-.5)
                 +1j*(np.random.rand(chi,d,chi)-.5))
    A.append((np.random.rand(chi,d)-.5)
             +1j*(np.random.rand(chi,d)-.5))
    return A

def getStateOBC(A):
    """
        State vector of MPS with open boundary conditions.
    """
    N = len(A) # Number of spins
    chi = A[1].shape[0] # Bond dimension
    d = A[1].shape[1] # d = 2 for qubits
    c = A[0]
    for i in xrange(1,N):
        c = np.tensordot(c,A[i],axes=(-1,0))
    return np.reshape(c,d**N)

def getMPSOBC(state,chi):
    """
        Matrix product state representation of a state with bond
        dimension chi and open boundary conditions.
    """
    d = 2 # Qubits have 2 states each
    N = int(np.log2(len(state))) # Number of qubits

    c = np.reshape(state,cShape(d,N)) # State amplitudes tensor c.
    A = [] # List of N matrices of MPS, each of shape (chi,d,chi)

    # Start left end with a vector of size (d,chi)
    c = np.reshape(c,(d,d**(N-1))) # Reshape c
    (ap,c) = efficientSVD(c)
    A.append(np.dot(ap,s)) # Contract and append to A

    printIntermediate(A,c,state,d,N)

    # Sweep through the middle, creating matrix products each with
    # shape (chi,d,chi)
    for i in xrange(1,N-2):
        #c = np.reshape(c,(d,chi,d**(N-i-1)))
        #c = np.transpose(c,(1,0,2))
        print("Executing routine i = %d"%i)
        show(c,"c before reshape")
        c = np.reshape(c,(d*chi,d**(N-i-1)))
        show(c,"c after reshape")
        (ap,sv,c) = np.linalg.svd(c)
        show(ap,"ap")
        show(sv,"sv")
        show(c,"c after svd")
        s = np.zeros((d*chi,chi),dtype=complex)
        s[:min(chi,len(sv)),:min(chi,len(sv))] = np.diag(sv[:chi])
        show(s,"s after construction")
        a = np.dot(ap,s)
        show(a,"a after contracting ap*s")
        a = np.reshape(a,(chi,d,chi))
        show(a,"a after reshape, just before append")
        A.append(a)
        newc = np.zeros((chi,d**(N-i-1)),dtype=complex)
        newc[:min(chi,len(sv)),:] = c[:chi,:]
        show(newc,"newc after construction")
        show(np.reshape(np.dot(a,newc),(d*chi,d**(N-i-1))),"new reconstructed c")
        show(c,"c before reshape")
        c = newc
        printIntermediate(A,c,state,d,N)
    
    # Finish right end with the remaining vector
    c = np.reshape(c,(d*chi,d))
    (ap,sv,c) = np.linalg.svd(c)
    s = np.zeros((chi,d),dtype=complex)
    s[:d,:d] = np.diag(sv[:chi])
    A.append(np.reshape(ap[:chi,:],(chi,d,chi)))
    c = np.dot(s,c)
    A.append(c)

    prod = A[0]
    for i in xrange(1,N):
        prod = np.tensordot(prod,A[i],axes=(-1,0))
    print(np.sum(np.absolute((prod-np.reshape(state,cShape(d,N))))))
    return A

def show(a,name):
    print(name)
    print(np.round(np.absolute(a),2))

def printIntermediate(A,c,state,d,N):
    print("Printing intermediate with %d elements in A."%len(A))
    prod = A[0]
    for i in xrange(1,len(A)):
        prod = np.tensordot(prod,A[i],axes=(-1,0))
    prod = np.tensordot(prod,c,axes=(-1,0))
    prod = np.reshape(prod,cShape(d,N))
    fid = fidelity(state,np.reshape(prod,d**N))
    dif = (np.sum(np.absolute(prod-np.reshape(state,cShape(d,N)))))
    print("Difference = %f; fidelity = %f;"%(dif,np.absolute(fid)))

def innerProductOBC(A,B):
    """ Inner product <A|B> using transfer matrices
        where A and B are MPS representations of }A> and }B>
        with open boundary conditions (OBC).
    """
    N = len(A) # Number of qubits
    chiA = A[1].shape[0] # bond dimension of MPS in A
    chiB = B[1].shape[0] # bond dimension of MPS in B
    d = A[1].shape[1] # d = 2 for qubits
    
    # Take adjoint of |A> to get <A|
    A = np.conj(A)
    # Construct list of transfer matrices by contracting pairs of
    # tensors from A and B.
    transfer = []
    t = np.tensordot(A[0],B[0],axes=(0,0))
    t = np.reshape(t,chiA*chiB)
    transfer.append(t)
    for i in xrange(1,N-1):
        t = np.tensordot(A[i],B[i],axes=(1,1))
        t = np.transpose(t,axes=(0,2,1,3))
        t = np.reshape(t,(chiA*chiB,chiA*chiB))
        transfer.append(t)
    t = np.tensordot(A[N-1],B[N-1],axes=(1,1))
    t = np.reshape(t,chiA*chiB)
    transfer.append(t)
    # Contract the transfer matrices.
    prod = transfer[0]
    for i in xrange(1,N-1):
        prod = np.dot(prod,transfer[i])
    prod = np.dot(prod,transfer[N-1])
    return prod

def randomState(d,N):
    state = (np.random.rand(d**N)-.5) + (np.random.rand(d**N)-.5)*1j
    state = state/np.linalg.norm(state)
    return state

def efficientSVD(A):
    """
    Efficient form of SVD.
    m < n
    """
    (m,n) = A.shape
    (a,sv,b) = np.linalg.svd(A)
    s = np.diag(sv)
    return (np.dot(a,s),b[:m,:])

def cShape(d,N):
    """ Returns the shape of c tensor representation.
        I.e. simply just (d,d,...,d) N times.
    """
    return tuple([d for i in xrange(N)])

if __name__ == "__main__":
    main()
