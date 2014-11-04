"""
mpstest.py
A test of manipulating matrix product states with numpy.
2014-08-11
"""

import numpy as np

def main():
    print(ghz(2))
    print(equalDist(2))
    chi = 5
    a = np.random.random_integers(0,4,(2,2,2))
    print(a)
    print(np.sum(a))

def getState(A):
    N = A.shape[0]
    chi = A[0].shape[0]
    d = 2

def MPS(ket):
    d = 2
    N = np.log2(len(ket))
    c = np.reshape(ket,cShape(d,N))

def equalDist(N):
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

def cShape(d,N):
    return tuple([d for i in xrange(N)])

if __name__ == "__main__":
    main()
