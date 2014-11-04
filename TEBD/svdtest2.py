"""
svdtest.py
2014-08-29
"""

import numpy as np
import matplotlib.pyplot as plt
from cmath import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def main():
    (X,Y,Z) = ([],[],[])
    m = 10
    for n in xrange(1,20):
        for chi in xrange(1,25):
            F = 0
            for i in xrange(10):
                F += test(m,n,chi)/10
            X.append(m*n)
            Y.append(chi)
            Z.append(F)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    # Plot the surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
    ax.set_xlabel('N')
    ax.set_ylabel('chi')
    ax.set_zlabel('|difference|')
    plt.show()

def test(m,n,chi):
    c = np.random.rand(m,n)*(1+1j)
    (a,b) = efficientSVD(c,chi)
    difference = c - np.dot(a,b)
##    show(np.linalg.norm(difference),"Norm of difference")
##    show(c.size,"Original total number of elements")
##    show(a.size+b.size,"New total number of elements")
    return np.linalg.norm(difference)

def efficientSVD(A,chi):
    """
    Efficient form of SVD.
    m < n
    """
    (m,n) = A.shape
    (a,sv,b) = np.linalg.svd(A)
    s = np.diag(sv[:min(chi,m)])
    return (np.dot(a[:,:min(chi,n)],s),b[:min(chi,m),:])

def show(a,name):
    print(name)
    print(np.round(np.absolute(a),2))

if __name__ == "__main__":
    main()
