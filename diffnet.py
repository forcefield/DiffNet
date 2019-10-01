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

  Copyright (C) 2018-2019 
  Huafeng Xu
'''

import numpy as np
from scipy import linalg
import cvxopt 
from cvxopt import matrix
import heapq

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
        
def measurement_index( i, j, K):
    '''
    The measurement index m = (i,j) corresponds to the serial index
    K + (K-1) + ... (K-i) + j - i - 1 = (i+1)*K - i(i+3)/2 + j - 1.
    The measurement for m = (i,i) are the first i=0,1,...,K positions.

    Args:
    i < j: the indices for the difference measurement of (i, j) 
    K: the total number of quantities of interests: 0<=i, j<K

    Return:
    The serial index of the measurement (i,j).
    '''
    return (i+1)*K - i*(i+3)/2 + j - 1

def sum_upper_triangle( x):
    '''
    Return the sum of the upper triangle elements of the square matrix x.
    '''
    s = 0.
    for i in xrange( x.size[0]):
        for j in xrange( i, x.size[1]):
            s += x[i,j]
    return s

def solution_to_nij( sol, K):
    '''
    Get the KxK n[i][j] symmetric matrix for fractions of measurements from the
    CVXOPT solution.
    '''
    if sol['status'] != 'optimal':
        raise ValueError, sol['status']
    x = sol['x']
    # print x
    n = matrix( 0., (K, K))
    for i in xrange(K):
        n[i,i] = x[i]
        for j in xrange(i+1, K):
            m = measurement_index( i, j, K)
            n[i,j] = n[j,i] = x[m]
    return n

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

def A_optimize( sij):
    '''
    Find the A-optimal of the difference network that minimizes the trace of
    the covariance matrix.  This corresponds to minimizing the average error.

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
    assert( sij.size[0] == sij.size[1])
    K = sij.size[0]
    M = K*(K+1)/2
    # x = ( n, u ), where u=(u_1,u_2,...,u_K) is the dual variables.
    # We will minimize \sum_k u_k = c.x
    c = matrix( [0.]*M + [1.]*K )
    
    # Subject to the following constraints
    # \sum_{m=1}^M n_m [ [ v_m.v_m^t, 0 ], [0, 0] ]
    # + u_k [ [0, 0], [0, 1] ] + [ [0, e_k], [e_k^t, 0] ] >= 0
    # for k = 1,2,...,K
    # where M = K*(K+1)/2 are the number of types of measurements.
    # m index the measurements, m = (i,j).
    # v_m is a length K measurement vector, where 
    #     v_{(i,i), a} = s_{ii}^{-1}\delta_{i,a}
    #     v_{(i,j), a} = s_{ij]^{-1}\delta_{i,a} - s_{ij}^{-1}\delta_{j,a}
    # The matrix U_m = v_m.v_m^t is
    # U_{(i,i), (a,b)} = s_{ii}^{-2}\delta_{i,a}\delta_{i,b}
    # U_{(i,j), (a,b)} 
    #     = s_{ij}^{-2}(\delta_{i,a}\delta_{i,b} + \delta_{j,a}\delta_{j,b}) 
    #     - s_{ij}^{-2}(\delta_{i,a}\delta_{j,b} + \delta_{j,a}\delta_{i,b})
    
    # G matrix, of dimension ((K+1)*(K+1), (M+K)).  Each column is a
    # column-major vector representing the KxK matrix of U_m augmented
    # by a length K vector, hence the dimension (K+1)x(K+1).
    Gs = [ matrix( 0., ((K+1)*(K+1), (M+K))) for k in xrange( K) ]
    hs = [ matrix( 0., (K+1, K+1)) for k in xrange( K) ]
    
    for i in xrange( K):
        # The index of matrix element (i,i) in column-major representation
        # of a (K+1)x(K+1) matrix is i*(K+1 + 1) 
        Gs[0][i*(K+2), i] = 1./(sij[i,i]*sij[i,i])
        for j in xrange( i+1, K):
            m = measurement_index( i, j, K)
            # The index of matrix element (i,j) in column-major representation
            # of a (K+1)x(K+1) matrix is j*(K+1) + i
            v2 = 1./(sij[i,j]*sij[i,j])
            Gs[0][j*(K+1) + i, m] = Gs[0][i*(K+1) + j, m] = -v2
            Gs[0][i*(K+2), m] = Gs[0][j*(K+2), m] = v2

    # G.(x, u) + e >=0 <=> -G.(x, u) <= e
    Gs[0] *= -1.

    for k in xrange( K):
        if (k>0): Gs[k][:,:M] = Gs[0][:,:M]
        # for the term u_k [ [0, 0], [0, 1] ]
        Gs[k][-1, M+k] = -1.
        
        hs[k][k,-1] = hs[k][-1,k] = 1.

    # The constraint n >= 0, as G0.x <= h0
    G0 = matrix( np.diag(np.concatenate( [ -np.ones( M), np.zeros( K) ])))
    h0 = matrix( np.zeros( M + K))

    # The constraint \sum_m n_m = 1.
    A = matrix( [1.]*M + [0.]*K, (1, M + K) )
    b = matrix( 1., (1, 1) )
    
    sol = cvxopt.solvers.sdp( c, G0, h0, Gs, hs, A, b)
    n = solution_to_nij( sol, K)

    return n

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

def MLestimate( xij, invsij2, x0=None):
    '''
    Given the measurements and their associated statistical errors, 
    estimate the individual quantities.

    Args:

    xij: KxK matrix, where xij[i][j] = xi - xj is the measured difference 
    between xi and xj for j!=i, and xij[i][i] = xi is the measured value
    of xi.  If a measurement is absent, the corresponding xij element can 
    be an arbitrary real number.

    invsij2: KxK matrix, where invsij2[i][j] = 1/\sigma_{ij}^2 is the inverse
    of the statistical variance for measurement xij[i][j].  If a measurement
    is absent, the corresponding invsij2 element should be set to 0 (as the
    statistical variance of the measurement is infinity).

    x0: K-element array, where x0[i] is the input a priori value of the i'th
    quantity.  x0[j]=None indicates that there is no input value for the i'th
    quantity.
    
    Return:

    x: a K element array, where x[i] is the estimate for the quantity xi.
    In the case that the Fisher matrix is not full-ranked, the returned x
    minimizes \sum_j (x[j] - x0[j])^2 where j goes over all indices for which
    x0[j] is not None.  This minimizes the RMSE of the estimates from the 
    a priori input.
    v: a Kxd matrix, where v is the null space of the Fisher information matrix
    F v[:,i] = 0 for i=0,1,...,d

    '''
    # z_i = \sigma_i^{-2} x_i + \sum_{j\neq i} \sigma_{ij}^{-2} x_{ij}
    z = np.sum( invsij2*xij, axis=1)  
    # F[i][i] = \sigma_i^{-2} + \sum_{k\neq i} \sigma_{ik}^{-2} 
    Fd = np.diag( np.sum( invsij2, axis=1))
    # F[i][j] = -\sigma_{ij}^{-2} for i\neq j
    F = -invsij2 + np.diag( np.diag( invsij2)) + Fd
    x, residuals, rank, sv = linalg.lstsq( F, z)
    v = null_space( F)
    assert( rank + v.shape[1] == xij.shape[0])

    if (v.shape[1]==0 or x0 is None): return x, v
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

def covariance( sij, nij):
    '''
    Compute the covariance matrix of the difference network.

    Args:
    
    sij:  KxK symmetric matrix, where the measurement variance of the
    difference between i and j is proportional to s[i][j]^2 =
    s[j][i]^2, and the measurement variance of i is proportional to
    s[i][i]^2.
    nij:  symmetric matrix, where n[i][j] is the fraction of measurements
    to be performed for the difference between i and j, satisfying
    \sum_i n[i][i] + \sum_{i<j} n[i][j] = 1.

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
    C = linalg.inv( F)
    return C

def diffnet_iterate( Nsofar, Nadd, nopt):
    '''
    In an iterative optimization of the difference network, the
    optimal allocation is updated with the estimate of s_{ij}, and we
    need to allocate the next iteration of sampling based on what has
    already been sampled for each pair.

    Args:

    Nsofar: KxK symmetric matrix, where Nsofar[i,j] is the number of samples
    that has already been collected for (i,j) pair.
    Nadd: int, Nadd gives the additional number of samples to be collected in
    the next iteration.
    nopt: KxK symmetric matrix, where nopt[i,j] is the optimal fraction of 
    samples to be allocated to the measurement of the difference between 
    i and j. 

    Return:

    KxK symmetric matrix of integers, the (i,j) element of which gives the
    number of samples to be allocated to the measurement of (i,j) difference
    in the next iteration.
    '''
    K = nopt.size[0]
    Nnext = np.zeros( (K, K), dtype=int)
    Ntotal = sum_upper_triangle( matrix(Nsofar))
    Nopt = np.asarray( nopt*(Ntotal + Nadd), dtype=int)
    # If a pair has already been sampled more than its optimal allocation, 
    # move its allocation to other pairs in proportion to the latter's 
    # allocation.
    extra = 0
    normalize = 0
    for i in xrange( K):
        for j in xrange( i, K):
            if Nopt[i,j] < Nsofar[i,j]:
                extra += (Nsofar[i,j] - Nopt[i,j])
            else:
                normalize += (Nopt[i,j] - Nsofar[i,j])
    scale = 1. - extra/float(normalize)
    for i in xrange( K):
        for j in xrange( i, K):
            if Nopt[i,j] > Nsofar[i,j]:
                Nnext[i,j] = int((Nopt[i,j] - Nsofar[i,j])*scale)
                Nnext[j,i] = Nnext[i,j]
    return Nnext
    

def check_optimality( sij, nij, optimality='A', delta=1E-1, ntimes=10):
    '''
    Return True if nij is the optimal.
    '''
    K = sij.size[0]
    C = covariance( sij, nij)
    fC = dict(
        A = np.trace( C), 
        D = np.log( linalg.det( C)),
        E = np.max( linalg.eig( C)[0]).real,
        Etree = np.max( linalg.eig( C)[0]).real
    )
    df = np.zeros( ntimes)
    for t in xrange( ntimes):
        zeta = matrix( 1. + 2*delta*(np.random.rand(  K, K) - 0.5))
        nijp = cvxopt.mul( nij, zeta)
        nijp = 0.5*(nijp + nijp.trans()) # Symmetrize
        s = sum_upper_triangle( nijp)
        nijp /= s
        Cp = covariance( sij, nijp)
        if (optimality=='A'):
            fCp = np.trace( Cp)
        elif (optimality=='D'):
            fCp = np.log( linalg.det( Cp))
        elif (optimality=='E' or optimality=='Etree'):
            fCp = np.max( linalg.eig( Cp)[0]).real
        df[t] = fCp - fC[optimality]
    print df
    return np.all( df >= 0)

def check_hessian( dF, d2F, x0):
    '''
    Check the Hessian for correctness.

    Returns:
    err: float - the square root of the sum of squres of the difference
    between finite difference approximation and the analytical results
    at the point x0.
    '''
    from scipy.optimize import check_grad

    N = len(x0)
    esqr = 0.
    for i in xrange( N):
        def func( x):
            return dF(x)[i]
        def dfunc( x):
            return d2F(x)[i,:]
        e = check_grad( func, dfunc, x0)
        esqr += e*e
    return np.sqrt(esqr)

def fabricate_measurements( K=10, sigma=0.1, noerror=True, disconnect=False):
    x0 = np.random.rand( K)
    xij = np.zeros( (K, K))
    invsij2 = 1/(sigma*sigma)*np.random.rand( K, K)
    invsij2 = 0.5*(invsij2 + np.transpose( invsij2))
    sij = np.sqrt( 1./invsij2)
    if noerror: sij *= 0.
    for i in xrange(K):
        xij[i][i] = x0[i]
        for j in xrange(i+1, K):
            xij[i][j] = x0[i] - x0[j] + sij[i][j]*(np.random.rand() - 0.5)
            xij[j][i] = -xij[i][j]

    if (disconnect >= 1):
        # disconnect the origin and thus eliminate the individual measurements
        for i in xrange(K): invsij2[i][i] = 0
    if (disconnect >= 2):
        # disconnect the network into the given number of disconnected
        # components.
        for i in xrange( K):
            c1 = i % disconnect
            for j in xrange( i+1, K):
                c2 = j % disconnect
                if (c1 != c2):
                    invsij2[i][j] = invsij2[j][i] = 0
        
    return x0, xij, invsij2

def check_MLest( K=10, sigma=0.1, noerr=True, disconnect=False):
    x0, xij, invsij2 = fabricate_measurements( K, sigma, noerr, disconnect)
    if (not disconnect):
        xML, vML = MLestimate( xij, invsij2)
    else:
        xML, vML = MLestimate( xij, invsij2, 
                               np.concatenate( [x0[:disconnect+1], 
                                                [None]*(K-disconnect-1)]))
    # Compute the RMSE between the input quantities and the estimation by ML.
    return np.sqrt(np.sum(np.square(xML - x0))/K)
    
def unitTest( tol=1.e-4):
    if (True):
        sij = matrix( [[ 1.5, 0.1, 0.2, 0.5],
                       [ 0.1, 1.1, 0.3, 0.2],
                       [ 0.2, 0.3, 1.2, 0.1],
                       [ 0.5, 0.2, 0.1, 0.9]])
    elif (False):
        sij = np.ones( (4, 4), dtype=float)
        sij += np.diag( 4.*np.ones( 4))
        sij = matrix( sij)
    else:
        sij = matrix ( [[ 1., 0.1, 0.1 ],
                        [ 0.1, 1., 0.1 ],
                        [ 0.1, 0.1, 1.2 ]])

    from scipy.optimize import check_grad

    def F( x):
        return lndetC( sij, x)[0]
    
    def dF( x):
        return np.array( lndetC( sij, x)[1])[0]
        
    def d2F( x):
        return np.array( lndetC( sij, x, True)[2])

    K = sij.size[0]
    
    x0 = np.random.rand( K*(K+1)/2)
    err = check_grad( F, dF, x0)
    print 'Gradient check for ln(det(C)) error=%g:' % err,
    if (err < tol):
        print 'Passed!'
    else:
        print 'Failed!'

    err = check_hessian( dF, d2F, x0)
    print 'Hessian check for ln(det(C)) error=%g:' % err,
    if (err < tol):
        print 'Passed!'
    else:
        print 'Failed!'

    print 'Testing ML estimator'
    for disconnect, label in [
            (False, 'Full-rank'), 
            (1, 'No individual measurement'), 
            (2, '2-disconnected') ]:
        err = check_MLest( K, disconnect=disconnect)
        print '%s: RMSE( x0, xML) = %g' % (label, err),
        if (err < tol): 
            print 'Passed!'
        else:
            print 'Failed!'

    results = optimize( sij)
    for o in [ 'D', 'A', 'E', 'Etree' ]:
        nij = results[o]
        C = covariance( sij, nij)
        print '%s-optimality' % o
        print 'n (sum=%g):' % sum_upper_triangle( nij)
        print nij
        D = np.log(linalg.det( C))
        A = np.trace( C)
        E = np.max(linalg.eig(C)[0]).real
        print 'C: (ln(det(C))=%.4f; tr(C)=%.4f; max(eig(C))=%.4f)' % \
            ( D, A, E )
        print C
        if (check_optimality( sij, nij, o)):
            print '%s-optimality check passed!' % o
        else:
            print '%s-optimality check failed!' % o
        
if __name__ == '__main__':
    unitTest()
