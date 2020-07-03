__doc__ = '''
DiffNet
-------

DiffNet is a Python tool for finding optimal allocations of sampling
in computational or experimental measurements of the individual
quantities and their pairwise differences, so as to minimize the covariance
in the estimated quantities.

License
-------

Released as free software.  NO WARRANTY.  Use AS IS.  

  Copyright (C) 2018-2020
  Huafeng Xu
  
Please cite https://doi.org/10.1021/acs.jcim.9b00528 if you use this
in a publication.  '''

import numpy as np
from scipy import linalg
import cvxopt 
from cvxopt import matrix, spmatrix
import heapq
import networkx as nx

import graph

try:
    from scipy.linalg import null_space
except ImportError:
    from scipy.linalg import svd
    def null_space( A, rcond=None):
        u, s, vh = svd(A, full_matrices=True)
        M, N = u.shape[0], vh.shape[1]
        if rcond is None:
            rcond = np.finfo(s.dtype).eps * max(M, N)
        tol = np.amax(s) * rcond
        num = np.sum(s > tol, dtype=int)
        Q = vh[num:,:].T.conj()
        return Q

from A_opt import solution_to_nij, measurement_index
from A_opt import A_optimize_fast, update_A_optimal_sdp, A_optimize_sdp
        
def sum_upper_triangle( x):
    '''
    Return the sum of the upper triangle elements of the square matrix x.
    '''
    if not isinstance(x, matrix): x = matrix( x)
    s = 0.
    for i in xrange( x.size[0]):
        for j in xrange( i, x.size[1]):
            s += x[i,j]
    return s

def lndetC( sij, x, hessian=False):
    '''
    f = ln det C = ln det F^{-1} = -ln det F
    where F = \sum_m^M x_m v_m.v_m^t
    
    By Jacob's formula
    
    df/dx_m = -tr(F^{-1}.(v_m.v_m^t))

    The second derivative is 
    
    d^2 f/dx_a dx_b = -tr( dC/dx_b.v_a.v_a^t)
                    = tr( C.dF/dx_b.C.v_a.v_a^t)
                    = tr( C.v_b.v_b^t.C.v_a.v_a^t)
    
    Return:
    tuple (f, d/dx f) if hessian is false
    tuple (f, d/dx f, d^2/dx^2 f) if hessian is true.
    '''
    if not isinstance( sij, matrix): sij = matrix( sij)
    K = sij.size[0]
    M = K*(K+1)/2
    F = matrix( 0., (K, K))
    for i in xrange( K):
        # n_{ii}*v_{ii}.v_{ii}^t
        F[i,i] += x[i]/(sij[i,i]*sij[i,i])
        for j in xrange( i+1, K):
            m = measurement_index( i, j, K)
            v2 = x[m]/(sij[i,j]*sij[i,j])
            F[i,i] += v2
            F[j,j] += v2
            F[i,j] = F[j,i] = -v2
    C = linalg.inv( F)
    fval = -np.log(linalg.det( F))
    df = matrix( 0., (1, M))
    for i in xrange( K):
        df[i] = -C[i,i]/(sij[i,i]*sij[i,i])
        for j in xrange( i+1, K):
            m = measurement_index( i, j, K)
            df[m] = (2*C[i,j] - C[i,i] - C[j,j])/(sij[i,j]*sij[i,j])
    if not hessian: 
        return (fval, df)
    # Compute the Hessian
    d2f = matrix( 0., (M, M))
    for i in xrange( K):
        for j in xrange( i, K):
            # d^2/dx_i dx_j = C_{ij}^2/(s_{ii}^2 s_{jj}^2)
            d2f[i, j] = C[i,j]*C[i,j]/(sij[i,i]*sij[i,i]*sij[j,j]*sij[j,j])
            d2f[j, i] = d2f[i, j]
        for i2 in xrange( K):
            for j2 in xrange( i2+1, K):
                m2 = measurement_index( i2, j2, K)
                # d^2/dx_id_x(i',j') = (C_{ii'}-C_{ji'})^2/(s_{i'i'}^2 s_{ij}^2)
                dC = C[i2,i] - C[j2,i]
                d2f[i, m2] = dC*dC/(sij[i,i]*sij[i,i]*sij[i2,j2]*sij[i2,j2])
                d2f[m2, i] = d2f[i, m2]
        for j in xrange( i+1, K):
            m = measurement_index( i, j, K)
            invs2 = 1/(sij[i,j]*sij[i,j])
            for i2 in xrange( i, K):
                for j2 in xrange( i2+1, K):
                    m2 = measurement_index( i2, j2, K)
                    # d^2/dx_{ij}dx_{i'j'} = 
                    # (C_{ii'}+C_{jj'}-C_{ji'}-C_{ij'})^2/(s_{i'j'}^2 s_{ij}^2)
                    dC = C[i,i2] + C[j,j2] - C[j,i2] - C[i,j2]
                    d2f[m,m2] = dC*dC*invs2/(sij[i2,j2]*sij[i2,j2])
                    d2f[m2,m] = d2f[m,m2]
    return (fval, df, d2f)
 
def A_optimize( sij, nadd=1., nsofar=None, delta=None, 
                only_include_measurements=None,
                method='conelp'):
    '''
    Find the A-optimal of the difference network that minimizes the trace of
    the covariance matrix.  This corresponds to minimizing the average error.

    In an iterative optimization of the difference network, the
    optimal allocation is updated with the estimate of s_{ij}, and we
    need to allocate the next iteration of sampling based on what has
    already been sampled for each pair.

    Args:

    sij: KxK symmetric matrix, where the measurement variance of the
    difference between i and j is proportional to s[i][j]^2 =
    s[j][i]^2, and the measurement variance of i is proportional to
    s[i][i]^2.

    nadd: float, Nadd gives the additional number of samples to be collected in
    the next iteration.

    nsofar: KxK symmetric matrix, where nsofar[i,j] is the number of samples
    that has already been collected for (i,j) pair.

    delta: a length K vector.  delta[i] is the measurement uncertainty on the
    quantity x[i] from an independent experiment; if no independent experiment
    provides a value for x[i], delta[i] can be set to None or numpy.infty.

    only_include_measurements: set of pairs, if not None, indicate which 
    pairs should be considered in the optimal network.  Any pair (i,j) not in 
    the set will be excluded in the allocation (i.e. dn[i,j] = 0).  The pair
    (i,j) in the set must be ordered so that i<=j. 

    method: 'conelp' (DEFAULT, with fast KKT solver), 'sdp'
    (Semidefinite programming).

    Return:

    KxK symmetric matrix of float, the (i,j) element of which gives the
    number of samples to be allocated to the measurement of (i,j) difference
    in the next iteration.

    '''
    if 'conelp'==method:
        nij = A_optimize_fast( sij, nadd, nsofar, delta, 
                               only_include_measurements)
    else:
        if delta is not None:
            raise ValueError, 'Currently delta values are only supported in A-optimal by the conelp method.'
        if nsofar is None:
            nij = A_optimize_sdp( sij)
            nij *= nadd
        else:
            nij = update_A_optimal_sdp( sij, nadd, nsofar, 
                                        only_include_measurements)
    return nij
    
def D_optimize( sij):
    '''
    Find the D-optimal of the difference network that minimizes the log of 
    the determinant of the covariance matrix.  This corresponds to minimize
    the volume of the confidence ellipsoid for a fixed confidence level.

    Args: 

    sij: KxK symmetric matrix, where the measurement variance of the
    difference between i and j is proportional to s[i][j]^2 =
    s[j][i]^2, and the measurement variance of i is proportional to
    s[i][i]^2.

    Return:

    nij: symmetric matrix, where n[i][j] is the fraction of measurements
    to be performed for the difference between i and j, satisfying  
    \sum_i n[i][i] + \sum_{i<j} n[i][j] = 1.

    The implementation follows Chapter 7.5 (Experimental design) of Boyd,
    Convex Optimization (http://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
    and http://cvxopt.org/userguide/solvers.html#problems-with-nonlinear-objectives.
    '''
    if not isinstance( sij, matrix): sij = matrix( sij)
    assert( sij.size[0]==sij.size[1])
    K = sij.size[0]
    M = K*(K+1)/2

    def F( x=None, z=None):
        if x is None:
            x0 = matrix( [ sij[i,i] for i in xrange( K) ] + 
                         [ sij[i,j] for i in xrange( K) 
                           for j in xrange( i+1, K) ], (M, 1))
            return (0, x0)
        if z is None:
            return lndetC( sij, x)
        f, df, d2f = lndetC( sij, x, True)
        return (f, df, z[0]*d2f)

    # The constraint n_m >= 0, formulated as G.x <= h
    G = matrix( np.diag( -np.ones( M)))
    h = matrix( np.zeros( M))

    # The constraint \sum_m n_m = 1.
    A = matrix( [1.]*M, (1, M))
    b = matrix( 1., (1, 1))

    sol = cvxopt.solvers.cp( F, G, h, A=A, b=b)

    n = solution_to_nij( sol, K)
    return n

def constant_relative_error( si):
    '''
    Construct a difference network with constant relative error, such that 
    s_{ij} = s_i - s_j, from the given $s_i$.

    Return:

    sij = s_i - s_j
    '''
    K = len(si)
    si = np.sort( si)
    sij = np.diag( si)
    for i in xrange( K):
        for j in xrange( i+1, K):
            sij[i,j] = sij[j,i] = si[j] - si[i]
    return matrix( sij)

def A_optimize_const_relative_error( si):
    '''
    Find the A-optimal of the difference network where s_{ij} = |s_i - s_j|.
    '''
    K = len(si)
    si = np.sort( si)

    nij = np.zeros( (K, K), dtype=float)
    N = nij[0,0] = np.sqrt( K)*si[0]
    for i in xrange(K-1):
        nij[i+1, i] = nij[i, i+1] = np.sqrt(K - (i+1))*(si[i+1] - si[i])
        N += nij[i, i+1]
    
    nij = matrix( nij/N)
    assert( abs(sum_upper_triangle( nij) - 1) < 1e-10)
    return nij

def D_optimize_const_relative_error( si):
    '''
    Find the D-optimal of the difference network where s_{ij} = |s_i - s_j|.
    '''
    K = len(si)
    si = np.sort( si)

    iK = 1./K
    nij = np.zeros( (K, K), dtype=float)
    nij[0,0] = iK
    for i in xrange(K-1):
        nij[i,i+1] = nij[i+1,i] = iK

    return matrix( nij)
        
def E_optimize( sij):
    '''
    Find the E-optimal of the difference network that minimizes the largest 
    eigenvalue of the covariance matrix.  This is equivalent to minimizing 
    the diameter of the confidence ellipsoid.

    Args: 

    sij: KxK symmetric matrix, where the measurement variance of the
    difference between i and j is proportional to s[i][j]^2 =
    s[j][i]^2, and the measurement variance of i is proportional to
    s[i][i]^2.

    Return:

    nij: symmetric matrix, where n[i][j] is the fraction of measurements
    to be performed for the difference between i and j, satisfying  
    \sum_i n[i][i] + \sum_{i<j} n[i][j] = 1.

    The implementation follows Chapter 7.5 (Experimental design) of Boyd,
    Convex Optimization (http://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
    and https://cvxopt.org/userguide/coneprog.html#semidefinite-programming
    '''
    # We solve the dual problem, which can be cast as a semidefinite
    # programming (SDP) problem.
    if not isinstance( sij, matrix): sij = matrix( sij)
    assert( sij.size[0] == sij.size[1])
    K = sij.size[0]
    M = K*(K+1)/2
    # x = ( n, t ) where t \in R
    # We will minimize t = c.x
    c = matrix( [ 0. ]*M + [1.] )
    
    # Subject to the following constraints
    # \sum_{m=1}^M n_m v_m.v_m^t - t I >= 0
    
    # G matrix, of dimension (K*K, M+1).
    # G[i*K + j] = (v_m.v_m^t)[i,j]
    G = matrix( 0., (K*K, M+1))
    h = matrix( 0., (K, K))
    for i in xrange( K):
        G[i*(K+1), i] = 1./(sij[i,i]*sij[i,i])
        G[i*(K+1), M] = 1.  # The column-major identity matrix for t.
        for j in xrange( i+1, K):
            m = measurement_index( i, j, K)
            v2 = 1./(sij[i,j]*sij[i,j])
            G[j*K + i, m] = G[i*K + j, m] = -v2
            G[i*(K+1), m] = G[j*(K+1), m] = v2

    # G.(x,t) >= 0 <=> -G.(x,t) + s = 0 & s >= 0
    G *= -1.
    
    # The constraint n >= 0.
    G0 = matrix( np.diag(np.concatenate( [ -np.ones( M), np.zeros( 1) ])))
    h0 = matrix( np.zeros( M + 1))

    # The constraint \sum_m n_m = 1.
    A = matrix( [1.]*M + [0.], (1, M + 1) )
    b = matrix( 1., (1, 1) )

    sol = cvxopt.solvers.sdp( c, G0, h0, [ G ], [ h ], A, b)
    n = solution_to_nij( sol, K)

    return n

def Dijkstra_shortest_path( sij):
    '''
    Find the shortest path tree from the origin to every node, where the
    distance between nodes (i, j) are given by sij[i,j], and the distance
    between node i and the origin is sij[i,i].

    This implementation follows https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm.
    '''
    K = sij.size[0]
    dist = np.inf*np.ones( K+1, dtype=float)  
    prev = -1*np.ones( K+1, dtype=int)

    # The origin has index K.
    dist[K] = 0
    q = [(0, K)]

    while q:
        d, u = heapq.heappop( q)
        for v in xrange( K):
            suv = sij[u,v] if u != K else sij[v,v]
            dp = d + suv
            if dp < dist[v]:
                dist[v] = dp
                prev[v] = u
                heapq.heappush( q, (dp, v))
    
    return dist, prev

def E_optimal_tree( sij):
    '''
    Construct a tree where each weighted edge represents a difference 
    measurement--and the weight is the fraction of measurements allocated
    to the corresponding edege--so that the measurements minimizes the 
    largest eigenvalue of the covariance matrix.

    Args: 

    sij: KxK symmetric matrix, where the measurement variance of the
    difference between i and j is proportional to s[i][j]^2 =
    s[j][i]^2, and the measurement variance of i is proportional to
    s[i][i]^2.

    Return:

    nij: symmetric matrix, where n[i][j] is the fraction of measurements
    to be performed for the difference between i and j, satisfying  
    \sum_i n[i][i] + \sum_{i<j} n[i][j] = 1.
    '''
    if not isinstance( sij, matrix): sij = matrix( sij)
    # a[i] = \sum_{e\elem E_i s_e}, where E_i is the shortest path
    # from origin to i.  prev[i] is the node preceding i in E_i.
    K = sij.size[0]
    a, prev = Dijkstra_shortest_path( sij)
    suma2 = np.sum( np.square( a))
    lamb = 1./suma2
    
    # For each node, compute \sum_{j \elem T_i} a_j, where j runs over
    # the set of nodes in the subtree T_i rooted at i, including i itself.
    suma = np.zeros( K, dtype=float)
    for v in xrange( K):
        suma[v] += a[v]
        u = prev[v]
        # Follow up the tree until the root at the origin
        while u != K:  # not the origin
            suma[u] += a[v]  # Add a[v] to each node u for v \elem T_u
            u = prev[u]

    # n_{i\mu_i} = s_{i\mu_i} \lambda \sum_{j\elem T_i} a_j
    nij = matrix( 0., (K, K))
    for i in xrange( K):
        j = prev[i]
        if j!=K: # not the origin
            nij[i,j] = sij[i,j]*suma[i]
            nij[j,i] = nij[i,j]
        else:
            nij[i,i] = sij[i,i]*suma[i]
    nij *= lamb

    return nij

OPTIMIZERS = dict(
    D = D_optimize, A = A_optimize, E = E_optimize, Etree = E_optimal_tree
)

def optimize( sij, optimalities=[ 'D', 'A', 'E', 'Etree' ],
              cutoff=1e-4):
    '''
    Find the optimal allocation of sampling to the measurements 
    of differences, according to the specified types of optimalities.

    Args:
    
    sij: KxK symmetric matrix, where the measurement variance of the
    difference between i and j is proportional to s[i][j]^2 =
    s[j][i]^2, and the measurement variance of i is proportional to
    s[i][i]^2.

    optimalities: list of choices, including 'D', 'A', 'E', and 'Etree'.
    This specifies the types of optimalities to solve (D-optimal, A-optimal,
    E-optimal, E-optimal as a tree).

    cutoff: float, if n_{ij} < cutoff s_{ij}/\sum_{i<=j} s_{ij}, we set
    n_{ij} = 0

    Return:

    results: dict, where the keys are 'D', 'A', 'E', 'Etree'.
    n = results[key] is a symmetric matrix, where n[i][j] is the fraction of 
    measurements to be performed for the difference between i and j, 
    satisfying  
    \sum_i n[i][i] + \sum_{i<j} n[i][j] = 1.
    '''
    nijc = cutoff*sij/np.sum( np.triu( sij))
    results = dict()
    if not isinstance( sij, matrix): sij = matrix( sij)
    for o in optimalities:
        try:
             nij = np.array( OPTIMIZERS[o]( sij))
             nij[ nij < nijc ] = 0.
             results[o] = matrix( nij)
        except ValueError:
            results[o] = None
    return results

def MLestimate( xij, invsij2, x0=None, di2=None):
    '''
    Given the measurements and their associated statistical errors, 
    estimate the individual quantities.

    Args:

    xij: KxK antisymmetricmatrix, where xij[i][j] = xi - xj is the
    measured difference between xi and xj for j!=i, and xij[i][i] = xi
    is the measured value of xi.  If a measurement is absent, the
    corresponding xij element can be an arbitrary real number.

    invsij2: KxK symmetric matrix, where invsij2[i][j] =
    1/\sigma_{ij}^2 is the inverse of the statistical variance for
    measurement xij[i][j].  If a measurement is absent, the
    corresponding invsij2 element should be set to 0 (as the
    statistical variance of the measurement is infinity).

    x0: K-element array, where x0[i] is the input a priori value of the i'th
    quantity.  x0[j]=None indicates that there is no input value for the i'th
    quantity.
    
    di2: length K array, di2[i] = 1/delta[i]^2, where delta[i] is the 
    uncertainty in the input a priori value of x0[i].  di2[i]=0 if x0[i]=None.

    Return:

    x: a K element array, where x[i] is the estimate for the quantity xi.
    In the case that the Fisher matrix is not full-ranked, the returned x
    minimizes \sum_j (x[j] - x0[j])^2 where j goes over all indices for which
    x0[j] is not None.  This minimizes the RMSE of the estimates from the 
    a priori input.
    v: a Kxd matrix, where v is the null space of the Fisher information matrix
    F v[:,i] = 0 for i=0,1,...,d

    '''
    xij = np.array( xij)
    invsij2 = np.array( invsij2)
    # z_i = \sigma_i^{-2} x_i + \sum_{j\neq i} \sigma_{ij}^{-2} x_{ij}
    z = np.sum( invsij2*xij, axis=1)  
    if di2 is not None and x0 is not None:
        for i in xrange( len(z)):
            if x0[i] is not None and x0[i] is not np.nan and di2[i] != 0:
                z[i] += di2[i]*x0[i]
    
    # F[i][i] = \sigma_i^{-2} + \delta_i^{-2} + \sum_{k\neq i} \sigma_{ik}^{-2} 
    Fd = np.diag( np.sum( invsij2, axis=1))
    if di2 is not None:
        # F[i][i] += 1/delta[i]^2
        Fd += np.diag( di2)
    # F[i][j] = -\sigma_{ij}^{-2} for i\neq j
    F = -invsij2 + np.diag( np.diag( invsij2)) + Fd
    rcond = np.finfo(F.dtype).eps * max(xij.shape[0], xij.shape[1])
    x, residuals, rank, sv = linalg.lstsq( F, z, rcond)
    v = null_space( F, rcond)
    assert( rank + v.shape[1] == xij.shape[0])

    if (v.shape[1]==0 or x0 is None): return x, v
    if type(x0)==list: x0 = np.array( x0)
    # Find l so as to minimize
    # \sum_j (x_j + l_a v_{ja} - x_{0j})^2
    dx = (x0[x0!=None].astype(float) - x[x0!=None])
    vp = v[x0!=None,:]
    lv, _, _, _ = linalg.lstsq( vp, dx)
    # vv = vp.dot( vp)
    # dxv = dx.dot( vp)
    # lv = linalg.solve( vv, dxv)
    x += v.dot( lv)

    return x, v

def covariance( sij, nij, delta=None):
    '''Compute the covariance matrix of the difference network.

    Args:
    
    sij:  KxK symmetric matrix, where the measurement variance of the
    difference between i and j is proportional to s[i][j]^2 =
    s[j][i]^2, and the measurement variance of i is proportional to
    s[i][i]^2.

    nij:  symmetric matrix, where n[i][j] is the fraction of measurements
    to be performed for the difference between i and j, satisfying
    \sum_i n[i][i] + \sum_{i<j} n[i][j] = 1.

    delta: length K vector, delta[i] is the standard deviation on the
    input reference value of x[i] (measured by an
    independentexperiment).  delta[i] = numpy.infty if there is no reference
    value for x[i].

    Return:

    KxK symmetric matrix for the covariance matrix of the corresponding 
    difference network.

    The covariance matrix is the inverse of the Fisher information matrix:

    F_{i,i} = n_{ii}/s_{ii}^2 + \sum_{k!= i} n_{ik}/s_{ij}^2
    F_{i,j} = -n_{ij}/s_{ij}^2

    '''
    K = sij.size[0]
    f = cvxopt.mul( nij, cvxopt.div( 1., sij)**2)
    F = np.diag( np.sum(f, axis=1) )
    for k in xrange(K): f[k,k] = 0
    F -= f
    if delta is not None:
        di2 = np.square( 1/delta)
        F += np.diag( di2)
    C = linalg.inv( F)
    return C

def round_to_integers( n):
    '''
    Round the allocations n_{e} to nearest integers i(n_e) = \floor{n_e} or
    \ceil{n_e} so that
    
    \sum_e i(n_e) = \sum_e n_e
    '''
    K = matrix( n).size[0]
    nsum = 0
    nceilsum = 0
    edges = []
    nint = np.zeros( (K, K), dtype=int)
    for i in xrange(K):
        for j in xrange(i, K):
            nsum += n[i,j]
            nint[i,j] = nint[j,i] = int(np.ceil( n[i,j]))
            nceilsum += nint[i,j]
            heapq.heappush( edges, (-n[i,j], (i,j)))
    nsum = int(np.floor(nsum + 0.5))
    surplus = nceilsum - nsum
    for k in xrange( surplus):
        d, (i, j) = heapq.heappop( edges)
        nint[i,j] -= 1
        nint[j,i] = nint[i,j]
    
    return nint

def MST_optimize( sij, allocation='std'):
    '''
    Measure the differences through a minimum spanning tree.

    Args:
    sij: KxK symmetric matrix, where the measurement variance of the
    difference between i and j is proportional to s[i][j]^2 =
    s[j][i]^2, and the measurement variance of i is proportional to
    s[i][i]^2.

    allocation: string - can be 'std', 'var', or 'n'. 
    if 'std', allocate n_{ij} \propto s_{ij},
    if 'var', allocate n_{ij} \propto s_{ij}^2
    if 'n', allocate n_{ij} = const
    n_{ij} = 0 for all (i,j) that are not part of the minimum spanning tree.

    Return:
    n: KxK symmetric matrix, where n[i][j] is the fraction of measurements
    to be performed for the difference between i and j, satisfying  
    \sum_i n[i][i] + \sum_{i<j} n[i][j] = 1.
    '''
    K = sij.size[0]
    G = graph.diffnet_to_graph( sij)
    T = nx.minimum_spanning_tree( G)
    n = matrix( 0., (K, K))
    for i, j, data in T.edges( data=True):
        weight = data['weight']
        if allocation == 'var':
            weight *= weight
        elif allocation == 'n':
            weight = 1.
        if i=='O':
            n[j,j] = weight
        elif j=='O':
            n[i,i] = weight
        else:
            n[i,j] = n[j,i] = weight
    s = sum_upper_triangle( n)
    n *= (1./s)  # So that \sum_{i<=j} n_{ij} = 1
    return n

def sparse_A_optimal_network( sij, nadd=1., nsofar=None, 
                              n_measure=0, connectivity=2,
                              sparse_by_fluctuation=True):
    '''
    Construct a sparse A-optimal network, so that (approximately) only
    max_measure different measurements will receive resource
    allocations, while guaranteeing the given degree of connectivity.

    Args:

    sij: KxK symmetric matrix, where the measurement variance of the
    difference between i and j is proportional to s[i][j]^2 =
    s[j][i]^2, and the measurement variance of i is proportional to
    s[i][i]^2.
    nadd: float, nadd gives the additional number of samples to be collected in
    the next iteration.
    nsofar: KxK symmetric matrix, where nsofar[i,j] is the number of samples
    that has already been collected for (i,j) pair.
    n_measure: int, the number of measurements to receive allocations.  
    The actual number of measurements with non-zero allocation might exceed
    this number in order to guarantee the connectivity. If it is zero, the
    number of measurements will be determined by the connectivity requirement.
    connectivity: int, ensure that the resulting difference network is k-edge
    connected.
    sparse_by_fluctuation: bool, if True, generate the sparse network by 
    minimizing \sum_e s_e in the k-connected spanning subgraph.
    
    Return:

    KxK symmetric matrix of float, the (i,j) element of which gives the
    number of samples to be allocated to the measurement of (i,j) difference
    in the next iteration.
    
    '''
    K = sij.size[0]

    if nsofar is None:
        nsofar = np.zeros( (K, K), dtype=float)
    if not sparse_by_fluctuation:
        # First, get the dense optimal network
        nij = update_A_optimal_sdp( sij, nadd, nsofar)
        def weight( i, j, epsilon=1e-10):
            n = nij[i,j]
            large = 1/epsilon
            if n > epsilon:
                return 1./n
            else:
                return large
    else:
        def weight( i,j):
            return sij[i,j]

    # Next, get the k-connected graph that approximately minimizes the 
    # sum of 1/n_{ij}.
    G = nx.Graph()
    G.add_nodes_from( range( K))
    G.add_node( 'O')
    edges = []
    
    for i in xrange(K):
        edges.append( ('O', i, weight( i,i)))
        for j in xrange(i+1, K):
            edges.append( (i, j, weight(i,j)))
    edges = list(nx.k_edge_augmentation( G, k=connectivity, partial=True))
    
    # Include only the edges that guarantee k-connectivity and nothing else 
    only_include_measurements = set([])
    for i, j in edges:
        if 'O'==i:
            only_include_measurements.add( (j,j))
        elif 'O'==j:
            only_include_measurements.add( (i,i))
        else:
            if i<j:
                only_include_measurements.add( (i,j))
            else:
                only_include_measurements.add( (j,i))
    
    # If there is additional allowance for the number of measurements,
    # add the remaining ones with the largest allocations from the dense
    # network.
    if (len(only_include_measurements) < n_measure):
        indices = []
        for i in xrange(K):
            for j in xrange(i, K):
                if (i,j) in only_include_measurements:
                    continue
                heapq.heappush( indices, (weight(i,j), (i,j)))
        addition = []
        for m in xrange(n_measure - len(only_include_measurements)):
            _w, (i,j) = heapq.heappop( indices)
            addition.append( (i,j))
        only_include_measurements.update( addition)
    
    nij = update_A_optimal_sdp( sij, nadd, nsofar, only_include_measurements)

    return nij

