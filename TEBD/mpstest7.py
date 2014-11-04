"""
mpstest7.py
A test of manipulating matrix product states with numpy.
2014-08-25
"""

import numpy as np
import matplotlib.pyplot as plt
from cmath import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def main():
    test3()

def test3():
    """ Test MPS conversion functions by computing fidelity between
        generated MPS and orginal, with new and old bond dimensions
        chi0 and chi1 varied.
    """
    print("*** Started testing MPS ***")
    N = 5
    d = 2
    # Points to plot on 3d graph
    (X,Y,Z) = ([],[],[])
    for chi0 in xrange(1,8):
        for chi1 in xrange(1,8):
            F = 0
            # Run random test for 20 points and take average fidelity
            for i in xrange(20):
                mps0 = randomMPS(N,chi0,d) # Make random MPS
                state0 = getState(mps0) # Convert to state
                mps1 = getMPS(state0,chi1) # Convert back to MPS with new bond dimension
                state1 = getState(mps1) # Convert back to state
                F += fidelityMPS(mps0,mps1) # Compute fidelity and add to sum
                # F += fidelity(state0,state1) # Uncomment this to try with vectors
            X.append(chi0)
            Y.append(chi1)
            Z.append(F/20)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    # Plot the surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
    ax.set_xlabel('chi0')
    ax.set_ylabel('chi1')
    ax.set_zlabel('fidelity')
    plt.show()
    print("*** Finished testing MPS ***")

def fidelityMPS(A,B):
    """ Fidelity of two MPS representations
        f = <A|B><B|A>/(<A|A><B|B>).
    """
    return innerProduct(A,B)*innerProduct(B,A)\
           /innerProduct(A,A)/innerProduct(B,B)

def fidelity(a,b):
    """ Fidelity of two state vectors
        f = <a|b><b|a>/(<a|a><b|b>).
    """
    return np.inner(np.conj(a),b)*np.inner(np.conj(b),a)\
           /np.inner(np.conj(a),a)/np.inner(np.conj(b),b)

def randomMPS(N,chi,d):
    """ Returns a random MPS given parameters N, chi, d."""
    A = []
    for i in xrange(N):
        # Each real part of each value varies between -0.5 and 0.5.
        A.append((np.random.rand(chi,d,chi)-.5)+1j*(np.random.rand(chi,d,chi)-.5))
    return np.array(A)

def getState(A):
    """ State vector of a MPS by contracting MPS."""
    N = len(A) # Number of spins
    chi = A[0].shape[0] # Bond dimension
    d = A[0].shape[1] # d = 2 for qubits
    c = A[0]
    for i in xrange(1,N):
        c = np.tensordot(c,A[i],axes=(-1,0))
    c = np.trace(c,axis1=0,axis2=-1)
    return np.reshape(c,d**N)

def getMPS(state,chi):
    """ MPS of a state."""
    d = 2 # Qubits have 2 states each
    N = int(np.log2(len(state))) # Number of qubits

    c = np.reshape(state,cShape(d,N)) # State amplitudes tensor c.
    A = [] # List of N matrices of MPS, each of shape (chi,d,chi)

    # Start left end with a vector of size (d,chi)
    c = np.reshape(c,(d,d**(N-1))) # Reshape c
    (ap,sv,c) = np.linalg.svd(c) # Apply SVD
    s = np.zeros((d,chi),dtype=complex) # Construct singular value matrix shape
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
    """ Inner product <A|B> using transfer matrices
        where A and B are MPS representations of }A> and }B>.
    """
    N = len(A) # Number of qubits
    chiA = A.shape[1] # bond dimension of MPS in A
    chiB = B.shape[1] # bond dimension of MPS in B
    d = A.shape[2] # d = 2 for qubits
    
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

def randomState(d,N):
    state = (np.random.rand(d**N)-.5) + (np.random.rand(d**N)-.5)*1j
    state = state/np.linalg.norm(state)
    return state

def cShape(d,N):
    """ Returns the shape of c tensor representation.
        I.e. simply just (d,d,...,d) N times.
    """
    return tuple([d for i in xrange(N)])

if __name__ == "__main__":
    main()
