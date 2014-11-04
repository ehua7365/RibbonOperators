"""
mpstest15.py
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

def main():
    test3()

def test1():
    """ Test functions just for a simple case.
    """
    mps0 = randomMPSOBC(3,3,2)
    state0 = getStateOBC(mps0)
    mps1 = getMPSOBC(state0,4)
    state1 = getStateOBC(mps1)
    print("Test completed. State Fidelity = %f. MPS Fidelity = %f"
          %(np.absolute(fidelity(state0,state1)),
            np.absolute(fidelityMPS(mps0,mps1))))

def test3():
    """ Test MPS conversion functions by computing fidelity between
        generated MPS and orginal, with new and old bond dimensions
        chi0 and chi1 varied.
    """
    print("*** Started testing MPS ***")
    N = 5
    d = 2
    nTrials = 3
    # Points to plot on 3d graph
    (X,Y,Z) = ([],[],[])
    for chi0 in xrange(1,10):
        for chi1 in xrange(1,10):
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
    A = [randomComplex((d,chi))]
    for i in xrange(N-2):
        A.append(randomComplex((chi,d,chi)))
    A.append(randomComplex((chi,d)))
    return A

def getStateOBC(A):
    """
        State vector of MPS with open boundary conditions.
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
    """
    d = 2 # Qubits have 2 states each
    N = int(np.log2(len(state))) # Number of qubits
##    show(N,"N = ")

    c = np.reshape(state,cShape(d,N)) # State amplitudes tensor c.
    A = [] # List of N matrices of MPS, each of shape (chi,d,chi)

    # Start left end with a vector of size (d,chi)
    c = np.reshape(c,(d,d**(N-1))) # Reshape c
    (ap,c) = efficientSVD(c,chi)
    A.append(ap) # Contract and append to A

##    printIntermediate(A,c,state,d,N)

    # Sweep through the middle, creating matrix products each with
    # shape (chi,d,chi)
    for i in xrange(1,N-2):
        #c = np.reshape(c,(d,chi,d**(N-i-1)))
        #c = np.transpose(c,(1,0,2))
##        print("Executing routine i = %d"%i)
##        show(c,"c before reshape")
        c = np.reshape(c,(d*A[-1].shape[-1],c.size/(d*A[-1].shape[-1])))
##        show(c,"c after reshape")
        (a,c) = efficientSVD(c,chi)
##        show(a,"a")
##        show(c,"c after svd")
        a = np.reshape(a,(A[-1].shape[-1],d,c.shape[0]))
##        show(a,"a after reshape, just before append")
        A.append(a)
##        printIntermediate(A,c,state,d,N)
    
    # Finish right end with the remaining vector
    c = np.reshape(c,(c.size/d,d))
    (a,c) = efficientSVD(c,chi)
    a = np.reshape(a,(A[-1].shape[-1],d,c.shape[0]))
    A.append(a)
    A.append(c)

##    for a in A:
##        print(a.shape)
##    prod = A[0]
##    for i in xrange(1,N):
##        prod = np.tensordot(prod,A[i],axes=(-1,0))
##    show(np.sum(np.absolute((prod-np.reshape(state,cShape(d,N))))),
##         "Difference")
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
    """ Inner product <A|B> using transfer matrices
        where A and B are MPS representations of }A> and }B>
        with open boundary conditions (OBC).
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
    """ Random N-qudit state vector
    """
    state = (np.random.rand(d**N)-.5) + (np.random.rand(d**N)-.5)*1j
    state = state/np.linalg.norm(state)
    return state

def efficientSVD(A,chi):
    """
    Efficient form of SVD.
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

if __name__ == "__main__":
    main()
