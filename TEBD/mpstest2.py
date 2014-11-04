"""
mpstest2.py
A test of manipulating matrix product states with numpy.
2014-08-13
"""

import numpy as np

def main():
    A = randomMPS(5,3,2)
    print(A.shape)
    print(getState(A).shape)
    
    print("*** MPS tests started ***")
    (N,chi,d) = (10,3,2)
    A = randomMPS(N,chi,d)
    state = getState(A)
    prod = np.dot(state,state)
    approxA = getMPS(state,chi)
    approxState = getState(approxA)
    approxProd = np.dot(approxState,approxState)
    relErr = approxProd/prod - 1
    print("State total %d elements"%state.size)
    print("MPS total %d elements"%A.size)
    print("(N,chi,d) = (%d,%d,%d)"%(N,chi,d))
    print("Expected:        %f"%prod)
    print("Transfer Matrix: %f"%innerProduct(A,A))
    print("SVD:             %f"%innerProduct(approxA,approxA))
    print("Product:         %f"%approxProd)
    print("Relative error:  %f"%relErr)
    print("*** MPS tests finished ***")
    

def randomMPS(N,chi,d):
    """ Returns a random MPS given parameters N, chi, d."""
    A = []
    for i in xrange(N):
        A.append(np.random.rand(chi,d,chi))
    return np.array(A)

def getState(A):
    """ Returns state given MPS A."""
    N = len(A)
    chi = A[0].shape[0]
    d = A[0].shape[1]
    c = A[0]
    for i in xrange(1,N):
        c = np.tensordot(c,A[i],axes=(-1,0))
    c = np.trace(c,axis1=0,axis2=-1)
    return np.reshape(c,d**N)

def getMPS(state,chi):
    """ Returns MPS of a state."""
    print("Computing getMPS()")
    d = 2
    N = int(np.log2(len(state)))

    c = np.reshape(state,cShape(d,N)) # State amplitudes tensor.
    A = [] # List of matrices for MPS

    # Start left end with a vector
    c = np.reshape(c,(d,d**(N-1)))
    (ap,sv,c) = np.linalg.svd(c)
    s = np.zeros((d,chi))
    s[:d,:d] = np.diag(sv)
    A.append(np.tensordot(ap,s,axes=(-1,0)))
    c = c[:chi,:]
    print("  Step 0")
    print("    Shape A[0] = %s"%str(A[0].shape))
    print("    Shape c = %s"%str(c.shape))
    print("    Total %d elements"%(c.size+A[0].size))

    # Sweep through the middle
    for i in xrange(1,N-2):
        c = np.reshape(c,(d*chi,d**(N-i-1)))
        (ap,sv,c) = np.linalg.svd(c)
        s = np.zeros((d*chi,chi))
        s[:chi,:chi] = np.diag(sv[:chi])
        A.append(np.reshape(np.dot(ap,s),(chi,d,chi)))
        c = c[:chi,:]
        print("  Step %d"%i)
        print("    Shape A[%d] = %s"%(i,str(A[1].shape)))
        print("    Shape c = %s"%str(c.shape))
        print("    Total %d elements"%(c.size+sum([mat.size for mat in A])))

    # Finish right end with vector
    c = np.reshape(c,(d*chi,d))
    (ap,sv,c) = np.linalg.svd(c)
    s = np.zeros((chi,d))
    s[:d,:d] = np.diag(sv)
    A.append(np.reshape(ap[:chi,:],(chi,d,chi)))
    c = np.dot(s,c)
    print("  Step %d"%(N-2))
    print("    Shape A[%d] = %s"%(N-2,str(A[N-2].shape)))
    print("    Shape c = %s"%str(c.shape))
    print("    Total %d elements"%(c.size+sum([mat.size for mat in A])))
    A.append(c)
    print("  Step %d"%(N-1))
    print("    Shape A[%d] = %s"%(N-1,str(A[N-1].shape)))
    print("    Total %d elements"%sum([mat.size for mat in A]))

    # Construct end matrices by filling zeros with end vectors
    start = np.zeros((chi,d,chi))
    start[0,:,:] = A[0]
    A[0] = start
    finish = np.zeros((chi,d,chi))
    finish[:,:,0] = A[-1]
    A[-1] = finish

    # Convert to numpy array
    A = np.array(A)

    print("  Finished MPS construction")
    print("    Total %d elements"%A.size)
    
    return A

def innerProduct(A,B):
    """ Returns inner product of two MPS representations A and B."""
    N = len(A)
    chiA = A.shape[1]
    chiB = B.shape[1]
    d = A.shape[2]
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

def operatorInner(A,O,B):
    """ Returns product <A|O|B>."""
    return 0

def getOperator(mpo):
    """ Returns operator as matrix given MPO."""
    return 0

def getMPO(O):
    """ Returns MPO of operator O."""
    return 0

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

if __name__ == "__main__":
    main()
