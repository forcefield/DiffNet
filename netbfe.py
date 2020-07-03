'''
This is an example of how to allocate computational resources for computing 
binding free energies of a set of molecules using a combination of relative
and absolute binding free energy calculations.  
'''

import numpy as np
import cvxopt
from cvxopt import matrix
from diffnet import A_optimize, MLestimate, round_to_integers
from diffnet import covariance, sum_upper_triangle
 
def COX2params():
    '''
    Generate the variance s[i,j] and the free energy results. s[i,i] is the
    variance of the absolute binding free energy for molecule i,
    s[i,j] (i!=j) is the variance of the relative binding free energy
    between molecules i and j.

    Also generate the experimental uncertainties delta[i].

    '''
    nheavy = dict(A1=7, A2=6, B1=9, B2=6, C1=10, C2=10)
    sCOX2 = np.diag( [nheavy['A1'] + nheavy['B1'] + nheavy['C1'],
                  nheavy['A1'] + nheavy['B1'] + nheavy['C2'],
                  nheavy['A1'] + nheavy['B2'] + nheavy['C1'],
                  nheavy['A1'] + nheavy['B2'] + nheavy['C2'],
                  nheavy['A2'] + nheavy['B1'] + nheavy['C1'],
                  nheavy['A2'] + nheavy['B1'] + nheavy['C2'],
                  nheavy['A2'] + nheavy['B2'] + nheavy['C1'],
                  nheavy['A2'] + nheavy['B2'] + nheavy['C2']]) + \
       np.array( [[ 0,  1, 16, 17,  1,  2, 16, 17],
                  [ 1,  0, 17, 16,  2,  1, 17, 16],
                  [16, 17,  0,  1, 16, 17,  1,  2],
                  [17, 16,  1,  0, 17, 16,  2,  1],
                  [ 1,  2, 16, 17,  0,  1, 16, 17],
                  [ 2,  1, 17, 16,  1,  0, 17, 16],
                  [16, 17,  1,  2, 16, 17,  0,  1],
                  [17, 16,  2,  1, 17, 16,  1,  0]], dtype=float)
    sCOX2 = 10.*np.sqrt( sCOX2)
    sCOX2 = matrix( sCOX2)

    K = sCOX2.size[0]

    # Experimental values and error bars
    dG0 = np.array([ -9.9, -8.9, -9.5, -7.2, -9.4, -4.6, -9, -9.6 ])
    delta = np.array([ 0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.2, 0.5 ])

    return dict( s=sCOX2,
                 dG0=dG0,
                 delta=delta)

def mockupBFEresults( n, s, dG0):
    '''
    Generate a mock-up of BFE results.  Return dG and 1/sigma^2, where
    dG[i,i] is the absolute binding free energy result for molecule i,
    and dG[i, j] = dG[i] - dG[j] is the relative binding free energy
    result between molecules i and j.

    '''
    si2 = cvxopt.div( 1., s)
    # isigma2 := \sigma_{ij}^{-2} = n_{ij}/s_{ij}^2
    isigma2 = cvxopt.div( n, si2)

    K = n.size[0]
    dG = matrix( 0., n.size)
    for i in xrange(K):
        if n[i,i] > 0:
            # results for absolute binding free energy.
            dG[i,i] = dG0[i] 
            dG[i,i] += 2*np.sqrt(1/isigma2[i,i])*(np.random.rand() - 0.5)
        for j in xrange(i+1, K):
            if n[i,j] > 0:
                ddG = dG0[i] - dG0[j] 
                ddG += 2*np.sqrt(1./isigma2[i,j])*(np.random.rand() - 0.5)
                dG[i, j] = ddG
                dG[j, i] = -ddG

    return dG, isigma2

def networkBFEalloc( s, N, delta=None):
    '''Use A-optimal to allocate the network of binding free energy
    calculations.

    Args:

    s: KxK symmetric matrix.

    s[i,i] gives the fluctuations in the absolute binding free energy
    calculation for molecule i;

    s[i,j] gives the fluctuations in the relative binding free energy
    calculation between molecule i and j.  s[i,j] = s[j,i].

    delta: length K array.

    delta[i] gives the experimental uncertainty for the measured
    binding free energy for the reference molecule i.

    N: total samples.

    Return:

    n: KxK symmetric matrix. 
    
    n[i,i] gives the allocation to the sampling of absolute binding
    free energy calculation of molecule i.

    n[i,j] gives the allocation of the sampling of relative binding
    free energy calculation between molecule i and j.
    '''
    n = A_optimize( s, N, delta=delta)
    return n

def networkBFEdG( ddG, isigma2, dG0, delta):
    '''Use maximum-likelihood to estimate the individual binding free
    energies given the computed absolute and binding free energy
    values, supplemented by the experimental binding free energies for
    some of the reference molecules.

    Args:

    ddG: KxK matrix, ddG[i,i] is the computed absolute binding free
    energy for molecule i.  ddG[i,j] = dG[i] - dG[j] is the computed
    relative binding free energy between molecules i and j.  dG[i,j] =
    -dG[j,i].

    isignam2: KxK matrix, isigma2[i,j] = 1/sigma[i,j]^2, where
    sigma[i,j] is the standard deviation in the computed free energy
    ddG[i,j].

    dG0: length K array, dG0[i] is the experimental binding free
    energy for molecule i.  dG0[i]=None if the experimental
    value is unavailable for molecule i.

    delta: length K array, delta[i] is the standard deviation in the
    measured dG0[i].

    Return:

    dG: length K array, dG[i] is the ML estimate for the individual
    binding free energy of molecule i.

    '''
    dG, v = MLestimate( ddG, isigma2, dG0, np.sqrt(1./delta))
    return dG

def test_A_optimality_with_reference( s, n, delta, dn=1E-1, ntimes=10):
    '''
    Return True if n is the A-optimal.
    '''
    K = n.size[0]
    cov = covariance( s, n, delta)
    f = np.trace( cov)
    df = np.zeros( ntimes)
    for t in xrange( ntimes):
        zeta = matrix( 1. + 2*dn*(np.random.rand(  K, K) - 0.5))
        n1 = cvxopt.mul( n, zeta)
        n1 = 0.5*(n1 + n1.trans()) # Symmetrize
        tot = sum_upper_triangle( n1)
        n1 /= tot
        Cp = covariance( s, n1, delta)
        fp = np.trace( Cp)
        df[t] = fp - f

    success = np.all( df >= 0)

    if success:
        print 'A-optimality with references passed!'
    else:
        print 'A-optimality with references FAILED!'
        print 'df = ', df

    return success

def unit_test():
    references = [0]
    cox2 = COX2params()
    s = cox2['s']
    dG0 = cox2['dG0']
    delta = cox2['delta']

    K = s.size[0]
    dG0p = [ x for x in dG0 ]
    # The experimental values of the molecules not in the references 
    # will be unavailable.
    if references is not None:
        for i in xrange( K):
            if i not in references:
                dG0p[i] = None
                delta[i] = np.infty
            
    N = 1000.
    n = networkBFEalloc( s, N, delta)

    success = True
    success = success and test_A_optimality_with_reference( s, n, delta)

    nint = round_to_integers( n)
    n = matrix( nint[:], (K,K), tc='d')

    ddG, isigma2 = mockupBFEresults( n, s, dG0)

    dG = networkBFEdG( ddG, isigma2, dG0p, delta)
    
    cov = covariance( s, n, delta)
    err = np.sqrt(np.diag( cov))
    deltaG = np.abs(dG - dG0)
    
    success = success and np.all( deltaG < err)
    if (success):
        print 'ML estimate with references passed!'
    else:
        print 'ML estimate with references FAILED!'
        print 'dG = ', dG
        print 'stderr = ', err
        print 'dG0 = ', dG0
    print 'max(|dG - dG0|) = %g' % np.max( deltaG)

if __name__ == '__main__':
    unit_test()
