"""
svdtest.py
2014-08-29
"""

import numpy as np

def main():
    test2()

def test1():
    (m,n) = (4,5)
    a = np.random.rand(m,m)
    b = np.random.rand(n,n)
    sv = np.array([4,3,0,0])
    s = np.zeros((m,n))
    s[:m,:m] = np.diag(sv)
    c = np.dot(a,np.dot(s,b))
    (ap,svp,bp) = np.linalg.svd(c)
    sp = np.zeros((m,n))
    sp[:m,:m] = np.diag(svp)
    cp = np.dot(ap,np.dot(sp,bp))
    show(a,"a")
    show(b,"b")
    show(s,"s")
    show(c,"c=a*s*b")
    show(ap,"ap")
    show(sp,"sp")
    show(bp,"bp")
    show(cp,"cp=ap*sp*bp")
    show(sv,"sv")
    show(np.linalg.norm(sv),"norm(sv)")
    show(sv/np.linalg.norm(sv),"normalized sv")
    show(svp,"svp")
    show(np.linalg.norm(svp),"norm(svp)")
    show(svp/np.linalg.norm(svp),"normalized svp")

def test2():
    c = np.random.rand(7,10)*(1+1j)
##    show(c,"c")
    (a,b) = efficientSVD2(c,10)
    show(a,"a")
    show(b,"b")
    difference = c - np.dot(a,b)
##    show(difference,"difference")
    show(np.linalg.norm(difference),"Norm of difference")
    show(c.size,"Original total number of elements")
    show(a.size+b.size,"New total number of elements")

def efficientSVD(A):
    """
    Efficient form of SVD.
    m < n
    """
    (m,n) = A.shape
    (a,sv,b) = np.linalg.svd(A)
    s = np.diag(sv)
    return (np.dot(a,s),b[:m,:])

def efficientSVD2(A,chi):
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
