"""
mpstest.py
A test of manipulating matrix product states with numpy.
2014-08-11
"""

import numpy as np

def main():
    print(ghz(2))
    print(ghz(3))

def ghz(N):
    c = np.zeros(2**N)
    c[0] = 1/np.sqrt(2)
    c[-1] = 1/np.sqrt(2)
    return np.reshape(c,cShape(2,N))
    

def cShape(d,N):
    return tuple([d for i in xrange(N)])

if __name__ == "__main__":
    main()
