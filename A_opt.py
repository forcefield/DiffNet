import numpy as np
import cvxopt
from cvxopt import blas, lapack, solvers, matrix, spmatrix, misc

def upper_index( i, j, K):
    '''
    Return the position of the tuple (i,j) in the sequence (0,0), (0,1), ... 
    (0,K-1), (1,1), ... (K-1,K-1).  i.e., 0<i<=j<K.
    '''
    return K*i - i*(i+1)/2 + j

def solution_to_nij( sol, K, measure_indices=None):
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
        if measure_indices is not None:
            m = measure_indices.get( (i,i), None)
            if m is not None:
                n[i,i] = x[m]
        else:
            n[i,i] = x[i]
        for j in xrange(i+1, K):
            if measure_indices is not None:
                m = measure_indices.get( (i,j), None)
                if m is None: continue
            else:
                m = measurement_index( i, j, K)
            n[i,j] = n[j,i] = x[m]
    return n

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

def conelp_solution_to_nij( x, K):
    '''
    '''
    n = matrix( 0.0, (K, K))
    p = 0
    for i in xrange(K):
        for j in xrange(i, K):
            n[i,j] = n[j,i] = x[p]
            p += 1
    return n

def cngrnc(r, x, n, xstart=0, alpha = 1.0):
    """
    Congruence transformation
    
    x := alpha * r'*x*r.
    
    r is a square matrix, and x is a symmetric matrix.
    """

    # return alpha*r.T*matrix(x, (n,n))*r
    
    # Scale diagonal of x by 1/2.
    nsqr = n*n
    xend = xstart+nsqr
    x[xstart:xend:n+1] *= 0.5
    
    # a := tril(x)*r
    a = +r
    tx = matrix(x[xstart:xend], (n,n))
    blas.trmm(tx, a, side = 'L')
    
    # x := alpha*(a*r' + r*a')
    blas.syr2k(r, a, tx, trans = 'T', alpha = alpha)
    x[xstart:xend] = tx[:]

def pairwise_diff( x, y, n):
    '''Compute pairwise difference x[:,i] - x[:,j] and store in y[:,k],
    where k is the index of (i,j) in the array (0,0), (0,1), ...,
    (1,1), ..., (k-1,k-1). y[:,(i,i)] = x[:,i]

    '''
    k = 0
    r = x.size[0]
    for i in xrange(n):
        #y[:,k] = x[:,i]
        blas.copy( x, y, n=r, offsetx=i*r, offsety=k*r)
        k+=1
        for j in xrange(i+1, n):
            #y[:,k] = x[:,i] - x[:,j]
            blas.copy( x, y, n=r, offsetx=i*r, offsety=k*r)
            blas.axpy( x, y, alpha=-1, n=r, offsetx=j*r, offsety=k*r)
            k+=1

def congruence_matrix( r, W, offset=0):
    '''
    Let r, x be KxK matrices, and let y = r^t x r.  Let vec(x) be the
    column major vector of matrix x.  Find the W matrix so that 
    
    vec(y) = W vec(x)

    y_{ij} = (r^t)_{ia} (x r)_{aj} = r_{ai} x_{ab} r_{bj}
           = r_{ai} r_{bj} x_{ab}

    So W_{(i,j),(a,b)} = r_{ai} r_{bj}
 
    '''
    K = r.size[0]
    ij = 0
    for i in xrange( K):
        for j in xrange( K):
            ab = 0
            for a in xrange( K):
                for b in xrange( K):
                    W[offset+ij, offset+ab] = r[a,i]*r[b,j]
                    ab += 1
            ij += 1
    return W

def symmetrize_matrix( x, n, xstart=0):
    '''
    Given a sequence x representing the lower triangle of a symmetric square
    nxn matrix, fill in the upper triangle with symmetric values.
    '''
    for a in xrange(n):
        for b in xrange(a+1, n):
            x[xstart+b*n+a] = x[xstart+a*n+b]

def tri2symm( x, n):
    '''
    Convert the sequence x[0,0], x[0,1], x[0,2], ..., x[0,n-1],
    x[1,1], ... x[n-1,n-1] from the upper triangle of a square
    symmetric matrix to the full symmetric matrix.
    '''
    y = matrix( 0., (n, n))
    p = 0
    for i in xrange( n):
        y[i:,i] = x[p:p+n-i]
        y[i, i:] = y[i:, i].T
        p += (n-i)
    return y

def Fisher_matrix( si2, nij):
    '''
    Return the Fisher information matrix.
    
    Args:

    si2: KxK symmetric matrix, si2[i,j] = 1/s[i,j]^2

    nij: KxK symmetric matrix of n[i,j].
    '''
    K = si2.size[0]
    F = -cvxopt.mul(nij, si2)
    ones = matrix( 1., (K, 1))
    d = matrix( 0., (K, 1))
    blas.symv( F, ones, d, alpha=-1.)
    F[::K+1] = d[:]
    return matrix( F, (K,K))

def sumdR2_aligned( Ris, K):
    '''In constructing the KKT equation, the coefficient for n_{ab} in
    the row (i,j) is given by
    
        R_{ai}^2 if a=b and i=j
        (R_{ai} - R_{bi})^2  if a!=b and i=j
        (R_{ai} - R_{aj})^2  if a=b and i!=j
        (R_{ai} + R_{bj} - R_{bi} - R_{aj})^2  if a!=b and i!=j
      * s_{ab}^{-2} * s_{ij}^{-2}

    summed over K of R matrices.

    This function computes the coefficients without the s_{ab}^{-2}s_{ij}^{-2}.
    '''
    ddR2 = matrix( 0., (K*(K+1)/2, K*(K+1)/2))

    # First, line up all the R matrices so that we can take the pairwise 
    # difference of their columns in one go:
    #
    #  R1[:,1]  R1[:,2]  ... R1[:,K]
    #  R2[:,1]  R2[:,2]  ... R2[:,K]
    #  ...
    #  RK[:,1]  R2[:,2]  ... RK[:,K]
    KR = matrix( 0., (K*K, K))
    start = 0
    for i in xrange(K):
        KR[start:start+K, :] = Ris[i][:K,:K]
        start += K
    
    dR = matrix( 0., (K*K, K*(K+1)/2))
    pairwise_diff( KR, dR, K)

    # Now, rearrange dR into 
    #
    # dR1[1,(1,1)] dR2[1,(1,1)] ... dRK[1,(1,1)] dR1[1,(1,2)] ... dRK[1,(K,K)]
    # dR1[2,(1,1)] dR2[2,(1,1)] ... dRK[2,(1,1)] dR1[2,(1,2)] ... dRK[2,(K,K)]
    # ...
    # dR1[K,(1,1)] dR2[K,(1,1)] ... dRK[K,(1,1)] dR1[K,(1,2)] ... dRK[K,(K,K)]
    #
    # Notice that this new matrix has the same column-major vector order as dR
    # itself.
    dR = matrix( dR[:], (K, K*K*(K+1)/2))
    
    # This scales with K^5 in memory!  If this becomes memory-bound, we need
    # to break this into subgroups.
    ddRs = matrix( 0., (K*K*(K+1)/2, K*(K+1)/2))
    # ddRs := ddR1[(1,1),:]
    #         ddR2[(1,1),:]
    #         ...
    #         ddRK[(1,1),:]
    #         ddR1[(1,2),:]
    #         ddR2[(1,2),:]
    #         ...
    #         ddRK[(1,2),:]
    #         ...
    #         ddR1[(K,K),:]
    #         ddR2[(K,K),:]
    #         ...
    #         ddRK[(K,K),:]
    # each ddRi is a K*(K+1)/2 by K*(K+1)/2 matrix
    pairwise_diff( dR.T, ddRs, K)
    ddRs = ddRs**2

    # Now sum up every K rows
    # ddR2 := \sum_i ddR[i]^2
    ddR2 = matrix( 0., (K*(K+1)/2, K*(K+1)/2))
    start = 0
    bsize = K*(K+1)/2
    for i in xrange(K):
        ddR2 += ddRs[i::K, :]
    return ddR2

def sumdR2( Ris, K):
    ddR2 = matrix( 0., (K*(K+1)/2, K*(K+1)/2))
    for i in xrange(K):
        Ri = Ris[i]
        # dR[:(a,b)] = R[:,a] - R[:,b]
        dR = matrix( 0., (K, K*(K+1)/2))
        pairwise_diff( Ri[:K,:K], dR, K)
        # ddR[:(ap,bp)] = dR'[:,ap] - dR'[:,bp]
        ddR = matrix( 0., (K*(K+1)/2, K*(K+1)/2))
        pairwise_diff( dR.T, ddR, K)
        ddR = ddR**2
        ddR2 += ddR
        # blas.axpy( ddR, ddR2, alpha=1., n=K*K*(K+1)*(K+1)/4)
    return ddR2

def Aopt_KKT_solver( si2, W):
    '''
    Construct a solver that solves the KKT equations associated with the cone 
    programming for A-optimal:

    / 0   At   Gt  \ / x \   / p \
    | A   0    0   | | y | = | q |
    \ G   0  -Wt W / \ z /   \ s /

    Args:

    si2: symmetric KxK matrix, si2[i,j] = 1/s_{ij}^2
    '''
    K = si2.size[0]

    ds = W['d']
    dis = W['di']  # dis[i] := 1./ds[i]
    rtis = W['rti']
    ris = W['r']

    d2s = ds**2
    di2s = dis**2

    # R_i = r_i^{-t}r_i^{-1} 
    Ris = [ matrix(0.0, (K+1, K+1)) for i in xrange(K) ]
    for i in xrange(K):
        blas.gemm( rtis[i], rtis[i], Ris[i], transB = 'T')

    ddR2 = sumdR2( Ris, K)

    # upper triangular representation si2ab[(a,b)] := si2[a,b]
    si2ab = matrix( 0., (K*(K+1)/2, 1))
    p = 0
    for i in xrange(K):
        si2ab[p:p+(K-i)] = si2[i:,i]
        p += (K-i)

    si2q = matrix( 0., (K*(K+1)/2, K*(K+1)/2))
    blas.syr( si2ab, si2q)
    
    sRVR = cvxopt.mul( si2q, ddR2)

    #  We first solve for K(K+1)/2 n_{ab}, K u_i, 1 y
    nvars = K*(K+1)/2 + K # + 1  We solve y by elimination of n and u.
    Bm = matrix( 0.0, (nvars, nvars))

    # The LHS matrix of equations
    # 
    # d_{ab}^{-2} n_{ab} + vec(V_{ab})^t . vec( \sum_i R_i* F R_i*) 
    # + \sum_i vec(V_{ab})^t . vec( g_i g_i^t) u_i + y 
    # = -d_{ab}^{-2}l_{ab} + ( p_{ab} - vec(V_{ab})^t . vec(\sum_i L_i*)
    # 

    # Coefficients for n_{ab}
    Bm[:K*(K+1)/2,:K*(K+1)/2] = cvxopt.mul( si2q, ddR2)

    row = 0
    for a in xrange(K):
        for b in xrange(a, K):
            Bm[row, row] += di2s[row] # d_{ab}^{-2} n_{ab}
            row += 1
    assert(K*(K+1)/2 == row)

    # Coefficients for u_i

    # The LHS of equations
    # g_i^t F g_i + R_{i,K+1,K+1}^2 u_i = pi - L_{i,K+1,K+1}
    dg = matrix( 0., (K, K*(K+1)/2))
    g = matrix( 0., (K, K))
    for i in xrange(K):
        g[i,:] = Ris[i][K,:K]
    # dg[:,(a,b)] = g[a] - g[b] if a!=b else g[a]
    pairwise_diff( g, dg, K)
    dg2 = dg**2
    # dg2 := s[(a,b)]^{-2} dg[(a,b)]^2
    for i in xrange( K):
        dg2[i,:] = cvxopt.mul( si2ab.T, dg2[i,:])

    Bm[K*(K+1)/2:K*(K+1)/2+K,:-K] = dg2
    # Diagonal coefficients for u_i.
    uoffset = K*(K+1)/2
    for i in xrange(K):
        RiKK = Ris[i][K,K]
        Bm[uoffset+i,uoffset+i] = RiKK**2

    # Compare with the default KKT solver.
    TEST_KKT = False
    if (TEST_KKT):
        Bm0 = matrix( 0., Bm.size)
        blas.copy( Bm, Bm0)
        G, h, A = Aopt_GhA( si2)
        dims = dict( l = K*(K+1)/2,
                     q = [],
                     s = [K+1]*K )
        default_solver = misc.kkt_ldl( G, dims, A)(W)

    ipiv = matrix( 0, Bm.size)
    lapack.sytrf( Bm, ipiv)
    # TODO: IS THIS A POSITIVE DEFINITE MATRIX?
    # lapack.potrf( Bm)

    # oz := (1, ..., 1, 0, ..., 0)' with K*(K+1)/2 ones and K zeros
    oz = matrix( 0., (Bm.size[0], 1))
    oz[:K*(K+1)/2] = 1.
    # iB1 := B^{-1} oz
    iB1 = matrix( oz[:], oz.size)
    lapack.sytrs( Bm, ipiv, iB1)
    # lapack.potrs( Bm, iB1)

    #######
    # 
    #  The solver
    #
    #######
    def kkt_solver( x, y, z):
        
        if (TEST_KKT):
            x0 = matrix( 0., x.size)
            y0 = matrix( 0., y.size)
            z0 = matrix( 0., z.size)
            x0[:] = x[:]
            y0[:] = y[:]
            z0[:] = z[:]

            # Get default solver solutions.
            xp = matrix( 0., x.size)
            yp = matrix( 0., y.size)
            zp = matrix( 0., z.size)
            xp[:] = x[:]
            yp[:] = y[:]
            zp[:] = z[:]
            default_solver( xp, yp, zp)
            offset = K*(K+1)/2
            for i in xrange(K):
                symmetrize_matrix( zp, K+1, offset)
                offset += (K+1)*(K+1)

        # pab = x[:K*(K+1)/2]  # p_{ab}  1<=a<=b<=K
        # pis = x[K*(K+1)/2:]  # \pi_i   1<=i<=K

        # z_{ab} := d_{ab}^{-1} z_{ab}
        # \mat{z}_i = r_i^{-1} \mat{z}_i r_i^{-t}
        misc.scale( z, W, trans='T', inverse='I')

        l = z[:]

        # l_{ab} := d_{ab}^{-2} z_{ab}
        # \mat{z}_i := r_i^{-t}r_i^{-1} \mat{z}_i r_i^{-t} r_i^{-1}
        misc.scale( l, W, trans='N', inverse='I')

        # The RHS of equations
        # 
        # d_{ab}^{-2}n_{ab} + vec(V_{ab})^t . vec( \sum_i R_i* F R_i*) 
        # + \sum_i vec(V_{ab})^t . vec( g_i g_i^t) u_i + y 
        # = -d_{ab}^{-2} l_{ab} + ( p_{ab} - vec(V_{ab})^t . vec(\sum_i L_i*)
        # 
        ###

        # Lsum := \sum_i L_i
        moffset = K*(K+1)/2
        Lsum = np.sum( np.array(l[moffset:]).reshape( (K, (K+1)*(K+1))), axis=0)
        Lsum = matrix( Lsum, (K+1, K+1))
        Ls = Lsum[:K,:K]
        
        x[:K*(K+1)/2] -= l[:K*(K+1)/2]

        dL = matrix( 0., (K*(K+1)/2, 1))
        ab = 0
        for a in xrange(K):
            dL[ab] = Ls[a,a]
            ab += 1
            for b in xrange(a+1,K):
                dL[ab] = Ls[a,a] + Ls[b,b] - 2*Ls[b,a]
                ab += 1

        x[:K*(K+1)/2] -= cvxopt.mul( si2ab, dL)

        # The RHS of equations
        # g_i^t F g_i + R_{i,K+1,K+1}^2 u_i = pi - L_{i,K+1,K+1}
        x[K*(K+1)/2:] -= l[K*(K+1)/2+(K+1)*(K+1)-1::(K+1)*(K+1)]

        # x := B^{-1} Cv
        lapack.sytrs( Bm, ipiv, x)
        # lapack.potrs( Bm, x)

        # y := (oz'.B^{-1}.Cv[:-1] - y)/(oz'.B^{-1}.oz)
        y[0] = (blas.dotu( oz, x) - y[0])/blas.dotu( oz, iB1)
        # x := B^{-1} Cv - B^{-1}.oz y
        blas.axpy( iB1, x, -y[0])

        # Solve for -n_{ab} - d_{ab}^2 z_{ab} = l_{ab}
        # We need to return scaled d*z.
        # z := d_{ab} d_{ab}^{-2}(n_{ab} + l_{ab})
        #    = d_{ab}^{-1}n_{ab} + d_{ab}^{-1}l_{ab}
        z[:K*(K+1)/2] += cvxopt.mul( dis, x[:K*(K+1)/2])
        z[:K*(K+1)/2] *= -1.

        # Solve for \mat{z}_i = -R_i (\mat{l}_i + diag(F, u_i)) R_i
        #                     = -L_i - R_i diag(F, u_i) R_i
        # We return 
        # r_i^t \mat{z}_i r_i = -r_i^{-1} (\mat{l}_i +  diag(F, u_i)) r_i^{-t} 
        ui = x[-K:]
        nab = tri2symm( x, K)

        F = Fisher_matrix( si2, nab)
        offset = K*(K+1)/2
        for i in xrange( K):
            start, end = i*(K+1)*(K+1), (i+1)*(K+1)*(K+1)
            Fu = matrix( 0.0, (K+1, K+1))
            Fu[:K,:K] = F
            Fu[K,K] = ui[i]
            Fu = matrix( Fu, ((K+1)*(K+1), 1))
            # Fu := -r_i^{-1} diag( F, u_i) r_i^{-t} 
            cngrnc( rtis[i], Fu, K+1, alpha=-1.)
            # Fu := -r_i^{-1} (\mat{l}_i + diag( F, u_i )) r_i^{-t}
            blas.axpy( z[offset+start:offset+end], Fu, alpha=-1.)
            z[offset+start:offset+end] = Fu

        if (TEST_KKT):
            offset = K*(K+1)/2
            for i in xrange(K):
                symmetrize_matrix( z, K+1, offset)
                offset += (K+1)*(K+1)
            dz = np.max(np.abs(z - zp))
            dx = np.max(np.abs(x - xp))
            dy = np.max(np.abs(y - yp))
            tol = 1e-5
            if dx > tol:
                print 'dx='
                print dx
                print x
                print xp
            if dy > tol:
                print 'dy='
                print dy
                print y
                print yp
            if dz > tol:
                print 'dz='
                print dz
                print z
                print zp
            if dx > tol or dy > tol or dz > tol:
                for i, (r, rti) in enumerate( zip(ris, rtis)):
                    print 'r[%d]=' % i
                    print r
                    print 'rti[%d]=' % i
                    print rti
                    print 'rti.T*r='
                    print rti.T*r
                for i, d in enumerate( ds):
                    print 'd[%d]=%g' % (i, d)
                print 'x0, y0, z0='
                print x0
                print y0
                print z0
                print Bm0

    ###
    #  END of kkt_solver.
    ###
        
    return kkt_solver

def Aopt_Gfunc( si2, x, y, alpha=1.0, beta=0.0, trans='N'):
    '''
    Compute 

    y := alpha G x + beta y if trans=='N'

    and
    
    y := alpha G^t x + beta y if trans!='N'

    Let x = (n11, n12, ... nKK, u1, u2, ..., uK)^t
    a G x + b y = -a ( n11, n12, ..., nKK, 
                        vec( F(n), 0, 
                                0, u1 ),
                        vec( F(n), 0,
                                0, u2 ),
                        ... ) + b y

    Let x = (x11, x12, ...., xKK, x_1, x_2, ... x_K), where
    x_i are (K+1)x(K+1) matrices.
    a Gt x + b y = -a ( x11 + vec( V11).vec( sum_i x_i),
                        x12 + vec( V12).vec( sum_i x_i),
                        ...
                        xKK + vec( VKK).vec( sum_i x_i),
                        x_{1,K+1,K+1},
                        ...
                        x_{K,K+1,K+1} ) + b y
    '''
    K = si2.size[0]
    hkkp1 = K*(K+1)/2
    kp1sqr = (K+1)*(K+1)
    if 'N'==trans:
        u = alpha*x[-K:]
        n = tri2symm( x[:-K], K)
        F = Fisher_matrix( si2, alpha*n)
        y[:hkkp1] = -alpha*x[:hkkp1] + beta*y[:hkkp1]
        Fu = matrix( 0., (K+1, K+1))
        Fu[:K,:K] = F
        start = hkkp1
        for i in xrange(K):
            Fu[K,K] = u[i]
            y[start:start+kp1sqr] = -Fu[:] + beta*y[start:start+kp1sqr]
            start += kp1sqr
    if 'T'==trans:
        xab = alpha*x[:hkkp1]
        xKK = alpha*x[hkkp1+kp1sqr-1::kp1sqr]
        start = hkkp1
        xsum = matrix( 0., (kp1sqr, 1))
        for i in xrange(K):
            xsum += x[start:start+kp1sqr]
            start += kp1sqr
        xsum *= alpha
        xsum = matrix( xsum, (K+1, K+1))
        ab = 0
        for a in xrange(K):
            xab[ab] += si2[a,a]*xsum[a,a]
            ab += 1
            for b in xrange(a+1, K):
                xab[ab] += si2[b,a]*(xsum[a,a] + xsum[b,b] - 2*xsum[b,a])
                ab += 1
        y[:hkkp1] = -xab + beta*y[:hkkp1]
        y[hkkp1:] = -xKK + beta*y[hkkp1:]

def Aopt_GhA( si2, nsofar=None, G_as_function=False):
    '''Return the G, h, and A matrix for the cone programming to solve the
    A-optimal problem.

    Args:
    
    si2: symmetric KxK matrix, si2[i,j] = 1/s_{ij}^2
    
    nsofar: symmetric KxK matrix, n_{ij} for existing samples

    Returns:

    G: a K*(K+1)/2 + K*(K+1)^2 by K(K+1)/2 + K matrix or a function
    (if G_as_function==True).

    h: a vector of length K*(K+1)/2 + K*(K+1)^2.

    A: A 1 by K*(K+1)/2 + K matrix.

    '''
    K = si2.size[0]

    if G_as_function:
        def G( x, y, alpha=1., beta=0., trans='N'):
            return Aopt_Gfunc( si2, x, y, alpha, beta, trans)
    else:
        nrows, ncols = K*(K+1)/2 + K*(K+1)*(K+1), K*(K+1)/2 + K
                 
        Gs = []
        # Gs = np.zeros( nrows*ncols)
        # -I_{K*(K+1)/2} identity matrix
        Gs.extend( [ ( i, i, -1. ) for i in xrange(K*(K+1)/2) ])
        # Gs[:K*(K+1)/2*nrows:nrows+1] = -1.  
        
        skip = K*(K+1)/2
        # vec( [ V_{ij}, 0; 0, 0 ])
        col = 0
        for i in xrange(K):
            # Skip the first K*(K+1)/2 rows 
            Gs.extend( [ (skip+i*(K+2)+t*(K+1)*(K+1), col, -si2[i,i])
                         for t in xrange( K) ])
            # offset = col*nrows + K*(K+1)/2  
            # Gs[offset+i*(K+2) : (col+1)*nrows : (K+1)*(K+1)] = si2[i,i]
            col += 1
            for j in xrange(i+1, K):
                Gs.extend( [ (skip+i*(K+2) + t*(K+1)*(K+1), col, -si2[i,j])
                             for t in xrange(K) ])
                Gs.extend( [ (skip+j*(K+2) + t*(K+1)*(K+1), col, -si2[i,j])
                             for t in xrange(K) ])
                Gs.extend( [ (skip+i*(K+1) + j + t*(K+1)*(K+1), col, si2[i,j])
                             for t in xrange(K) ])
                Gs.extend( [ (skip+j*(K+1) + i + t*(K+1)*(K+1), col, si2[i,j])
                             for t in xrange(K) ])
                col += 1
        
        # vec( [ 0, 0; 0, 1 ])
        
        Gs.extend( [ (skip+(i+1)*(K+1)*(K+1)-1, K*(K+1)/2+i, -1.)
                     for i in xrange(K) ])
        I, J, X = [ [ ijx[p] for ijx in Gs ] for p in xrange(3) ]
        G = spmatrix( X, I, J, (nrows, ncols))

    # h vector.
    h = matrix( 0., (K*(K+1)/2 + K*(K+1)*(K+1), 1))
    # F := Fisher matrix
    if nsofar is not None:
        F = Fisher_matrix( si2, nsofar)
    else:
        F = None

    row = K*(K+1)/2
    for i in xrange(K):
        if F is not None:
            for j in xrange(K):
                h[row + j*(K+1) : row + (j+1)*(K+1)-1] = F[:,j]
        h[row + (i+1)*(K+1)-1] = 1.  # e_i^t
        h[row + K*(K+1) + i] = 1.    # e_i
        row += (K+1)*(K+1)
    h = matrix( h, (len(h), 1))

    # A matrix
    A = matrix( np.concatenate( [ np.ones( K*(K+1)/2), np.zeros( K) ]), 
                (1, K*(K+1)/2 + K))
    
    return G, h, A

def A_optimize_fast( sij, N=1., nsofar=None, only_include_measurements=None):
    '''
    Find the A-optimal of the difference network that minimizes the trace of
    the covariance matrix.  This corresponds to minimizing the average error.

    In an iterative optimization of the difference network, the
    optimal allocation is updated with the estimate of s_{ij}, and we
    need to allocate the next iteration of sampling based on what has
    already been sampled for each pair.

    This implementation uses a customized KKT solver.  The time complexity is
    O(K^5), memory complexity is O(K^4).

    Args:

    sij: KxK symmetric matrix, where the measurement variance of the
    difference between i and j is proportional to s[i][j]^2 =
    s[j][i]^2, and the measurement variance of i is proportional to
    s[i][i]^2.

    nadd: float, Nadd gives the additional number of samples to be collected in
    the next iteration.

    nsofar: KxK symmetric matrix, where nsofar[i,j] is the number of samples
    that has already been collected for (i,j) pair.

    only_include_measurements: set of pairs, if not None, indicate which 
    pairs should be considered in the optimal network.  Any pair (i,j) not in 
    the set will be excluded in the allocation (i.e. dn[i,j] = 0).  The pair
    (i,j) in the set must be ordered so that i<=j. 

    Return:

    KxK symmetric matrix of float, the (i,j) element of which gives the
    number of samples to be allocated to the measurement of (i,j) difference
    in the next iteration.
    '''
    si2 = cvxopt.div( 1., sij**2) 
    K = si2.size[0]
    
    if only_include_measurements is not None:
        for i in xrange(K):
            for j in xrange(i, K):
                if not (i,j) in only_include_measurements:
                    # Set the s[i,j] to infinity, thus excluding the pair.
                    si2[i,j] = si2[j,i] = 0.

    Gm, hv, Am = Aopt_GhA( si2, nsofar, G_as_function=True)
    dims = dict( l = K*(K+1)/2,
                 q = [],
                 s = [K+1]*K )
    
    cv = matrix( np.concatenate( [ np.zeros( K*(K+1)/2), np.ones( K) ]),
                 (K*(K+1)/2 + K, 1))
    bv = matrix( float(N), (1, 1))

    def default_kkt_solver( W):
        return misc.kkt_ldl( Gm, dims, Am)(W)

    sol = solvers.conelp( cv, Gm, hv, dims, Am, bv, 
                          options=dict(maxiters=50,
                                       feastol=1e-7),
                           kktsolver=lambda W: Aopt_KKT_solver( si2, W))

    return conelp_solution_to_nij( sol['x'], K)
  
def A_optimize_sdp( sij):
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
    if not isinstance( sij, matrix): sij = matrix( sij)
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
    # Gs = [ matrix( 0., ((K+1)*(K+1), (M+K))) for k in xrange( K) ]
    G0 = []
    hs = [ matrix( 0., (K+1, K+1)) for k in xrange( K) ]
    
    for i in xrange( K):
        # The index of matrix element (i,i) in column-major representation
        # of a (K+1)x(K+1) matrix is i*(K+1 + 1) 
        # Gs[0][i*(K+2), i] = 1./(sij[i,i]*sij[i,i])
        G0.append( (i*(K+2), i, -1./(sij[i,i]*sij[i,i])))
        for j in xrange( i+1, K):
            m = measurement_index( i, j, K)
            # The index of matrix element (i,j) in column-major representation
            # of a (K+1)x(K+1) matrix is j*(K+1) + i
            v2 = 1./(sij[i,j]*sij[i,j])
            # Gs[0][j*(K+1) + i, m] = Gs[0][i*(K+1) + j, m] = -v2
            G0.append( (j*(K+1) + i, m, v2))
            G0.append( (i*(K+1) + j, m, v2))
            # Gs[0][i*(K+2), m] = Gs[0][j*(K+2), m] = v2
            G0.append( (i*(K+2), m, -v2))
            G0.append( (j*(K+2), m, -v2))
            
    # G.(x, u) + h >=0 <=> -G.(x, u) <= h
    # Gs[0] *= -1.
    
    Gs = []
    for k in xrange( K):
        # if (k>0): Gs[k][:,:M] = Gs[0][:,:M]
        # for the term u_k [ [0, 0], [0, 1] ]
        # Gs[k][-1, M+k] = -1.
        I = [ i for i, j, x in G0 ] + [ (K+1)*(K+1) - 1 ]
        J = [ j for i, j, x in G0 ] + [ M + k ]
        X = [ x for i, j, x in G0 ] + [ -1. ]
        Gs.append( spmatrix(X, I, J, ((K+1)*(K+1), M+K)))
        hs[k][k,-1] = hs[k][-1,k] = 1.

    # The constraint n >= 0, as G0.x <= h0
    # G0 = matrix( np.diag(np.concatenate( [ -np.ones( M), np.zeros( K) ])))
    G0 = spmatrix( -np.ones( M), range( M), range( M), (M+K, M+K))
    h0 = matrix( np.zeros( M + K))

    # The constraint \sum_m n_m = 1.
    # A = matrix( [1.]*M + [0.]*K, (1, M + K) )
    A = spmatrix( np.ones( M), np.zeros( M, dtype=int), range( M), (1, M+K))
    b = matrix( 1., (1, 1) )
    
    sol = cvxopt.solvers.sdp( c, G0, h0, Gs, hs, A, b)
    n = solution_to_nij( sol, K)

    return n

def update_A_optimal_sdp( sij, nadd, nsofar, only_include_measurements=None):
    '''
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
    only_include_measurements: set of pairs, if not None, indicate which 
    pairs should be considered in the optimal network.  Any pair (i,j) not in 
    the set will be excluded in the allocation (i.e. dn[i,j] = 0).  The pair
    (i,j) in the set must be ordered so that i<=j. 

    Return:

    KxK symmetric matrix of float, the (i,j) element of which gives the
    number of samples to be allocated to the measurement of (i,j) difference
    in the next iteration.
    '''
    if not isinstance( sij, matrix): sij = matrix( sij)
    assert( sij.size[0] == sij.size[1])
    K = sij.size[0]
    if only_include_measurements is None:
        M = K*(K+1)/2
    else:
        M = len(only_include_measurements)
        measure_indices = dict()
        for mid, (i,j) in enumerate( only_include_measurements):
            measure_indices[(i,j)] = mid

    # x = ( n, u ), where u=(u_1,u_2,...,u_K) is the dual variables.
    # We will minimize \sum_k u_k = c.x
    c = matrix( [0.]*M + [1.]*K )

    # Subject to the following constraints
    # \sum_{m=1}^M (n_m + dn_m) [ [ v_m.v_m^t, 0 ], [0, 0] ]
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
    # where \delta_{i,a} = 1 if i==a else 0 is the Kronecker delta.
    
    # G matrix, of dimension ((K+1)*(K+1), (M+K)).  Each column is a
    # column-major vector representing the KxK matrix of U_m augmented
    # by a length K vector, hence the dimension (K+1)x(K+1).
    # Gs = [ matrix( 0., ((K+1)*(K+1), (M+K))) for k in xrange( K) ]
    G0 = []
    hs = [ matrix( 0., (K+1, K+1)) for k in xrange( K) ]
    
    for i in xrange( K):
        # The index of matrix element (i,i) in column-major representation
        # of a (K+1)x(K+1) matrix is i*(K+1 + 1) 
        v2 = 1./(sij[i,i]*sij[i,i])
        if (only_include_measurements is not None):
            m = measure_indices.get( (i,i), None)
            if m is not None:
                # Gs[0][i*(K+2), m] = v2
                G0.append( (i*(K+2), m, -v2))
        else:
            # Gs[0][i*(K+2), i] = v2
            G0.append( (i*(K+2), i, -v2))
        hs[0][i,i] += nsofar[i,i]*v2
        for j in xrange( i+1, K):
            # The index of matrix element (i,j) in column-major representation
            # of a (K+1)x(K+1) matrix is j*(K+1) + i
            v2 = 1./(sij[i,j]*sij[i,j])
            nv2 = nsofar[i,j]*v2
            hs[0][i,j] = hs[0][j,i] = -nv2
            hs[0][i,i] += nv2
            hs[0][j,j] += nv2
            if (only_include_measurements is not None):
                m = measure_indices.get( (i,j), None)
                if m is None: continue
            else:        
                m = measurement_index( i, j, K)
            # Gs[0][j*(K+1) + i, m] = Gs[0][i*(K+1) + j, m] = -v2
            G0.append( (j*(K+1) + i, m, v2))
            G0.append( (i*(K+1) + j, m, v2))
            # Gs[0][i*(K+2), m] = Gs[0][j*(K+2), m] = v2
            G0.append( (i*(K+2), m, -v2))
            G0.append( (j*(K+2), m, -v2))

    # G.(x, u) + h >=0 <=> -G.(x, u) <= h
    # Gs[0] *= -1.

    Gs = []
    for k in xrange( K):
        if (k>0): 
            # Gs[k][:,:M] = Gs[0][:,:M]
            hs[k][:K,:K] = hs[0][:K,:K]
        # for the term u_k [ [0, 0], [0, 1] ]
        # Gs[k][-1, M+k] = -1.
        I = [ i for i, j, x in G0 ] + [ (K+1)*(K+1) - 1 ]
        J = [ j for i, j, x in G0 ] + [ M + k ]
        X = [ x for i, j, x in G0 ] + [ -1. ]
        Gs.append( spmatrix(X, I, J, ((K+1)*(K+1), M+K)))
        hs[k][k,-1] = hs[k][-1,k] = 1.

    # The constraint dn >= 0, as G0.x <= h0
    # G0 = matrix( np.diag(np.concatenate( [ -np.ones( M), np.zeros( K) ])))
    G0 = spmatrix( -np.ones( M), range(M), range(M), (M+K, M+K))
    h0 = matrix( np.zeros( M + K))

    # The constraint \sum_m dn_m = nadd.
    # A = matrix( [1.]*M + [0.]*K, (1, M + K) )
    A = spmatrix( np.ones( M), np.zeros( M, dtype=int), range( M), (1, M+K))
    b = matrix( float(nadd), (1, 1) )
    
    sol = cvxopt.solvers.sdp( c, G0, h0, Gs, hs, A, b)
    dn = solution_to_nij( sol, K, only_include_measurements and measure_indices)

    return dn

def test_kkt_solver( ntrials=5, tol=1e-6):
    K = 5
    sij = matrix( np.random.rand( K*K), (K, K))
    sij = 0.5*(sij + sij.T)
 
    si2 = cvxopt.div( 1., sij**2)
    G, h, A = Aopt_GhA( si2)
    K = si2.size[0]

    dims = dict( l = K*(K+1)/2,
                 q = [],
                 s = [K+1]*K )
    
    def default_solver( W):
        return misc.kkt_ldl( G, dims, A)(W)

    def my_solver( W):
        return Aopt_KKT_solver( si2, W)

    for t in xrange( ntrials):
        x = matrix( 1*(np.random.rand( K*(K+1)/2+K) - 0.5), (K*(K+1)/2+K, 1))
        y = matrix( np.random.rand( 1), (1,1))
        z = matrix( 0.0, (K*(K+1)/2 + K*(K+1)*(K+1), 1))
        z[:K*(K+1)/2] = 5.*(np.random.rand( K*(K+1)/2) - 0.5)
        offset = K*(K+1)/2
        for i in xrange(K):
            r = 10*(np.random.rand( (K+1)*(K+2)/2) - 0.3)
            p = 0
            for a in xrange(K+1):
                for b in xrange(a, K+1):
                    z[offset + a*(K+1) + b] = r[p]
                    z[offset + b*(K+1) + a] = r[p]
                    p+=1
            offset += (K+1)*(K+1)
        
        ds = matrix( 10*np.random.rand( K*(K+1)/2), (K*(K+1)/2, 1))
        rs = [ matrix(np.random.rand( (K+1)*(K+1)) - 0.3, (K+1, K+1)) 
               for i in xrange(K) ]
        W = dict( d=ds,
                  di=cvxopt.div(1., ds),
                  r=rs,
                  rti=[ matrix( np.linalg.inv( np.array(r)), (K+1,K+1)).T
                        for r in rs ],
                  beta=[],
                  v=[])
        xp = x[:]
        yp = y[:]
        zp = z[:]
        
        default_f = default_solver( W)
        my_f = my_solver( W)
        default_f( x, y, z)
        my_f( xp, yp, zp)

        dx = xp - x
        dy = yp - y
        offset = K*(K+1)/2
        for i in xrange(K):
            symmetrize_matrix( zp, K+1, offset)
            symmetrize_matrix( z, K+1, offset)
            offset += (K+1)*(K+1)
        dz = zp - z
        
        dx, dy, dz = np.max(np.abs(dx)), np.max(np.abs(dy)), np.max(np.abs(dz))
        
        if tol < np.max( [dx, dy, dz]):
            print 'KKT solver FAILS: max(dx=%g, dy=%g, dz=%g) > tol = %g' % \
                (dx, dy, dz, tol)
        print 'KKT solver succeeds: dx=%g, dy=%g, dz=%g' % (dx, dy, dz)

def test_Gfunc( ntrials=10, tol=1e-10):
    K = 5
    sij = matrix( np.random.rand( K*K), (K, K))
    sij = 0.5*(sij.T + sij)
    si2 = cvxopt.div( 1., sij**2)
    alpha = 1.5
    beta = 0.25
    G, h, A = Aopt_GhA( si2)

    for i in xrange( ntrials):
        trans = 'N'
        nx = K*(K+1)/2+K
        ny = K*(K+1)/2+K*(K+1)*(K+1)
        x = matrix( np.random.rand( nx), (nx, 1))
        y = matrix( 1.e6*np.random.rand( ny), (ny, 1))
        
        yp = y[:]
        Aopt_Gfunc( si2, x, y, alpha, beta, trans)
        blas.gemv( matrix(G), x, yp, 'N', alpha, beta)
        
        dy = np.max(np.abs(y - yp))
        if (dy > tol):
            print 'G function fails for trans=N: dy=%g' % dy
        else:
            print 'G function succeeds for trans=N: dy=%g' % dy
            
        trans = 'T'
        nx = K*(K+1)/2 + K*(K+1)*(K+1)
        ny = K*(K+1)/2 + K
        x = matrix( np.random.rand( nx), (nx, 1))
        y = matrix( 1.e6*np.random.rand( ny), (ny, 1))
        
        for i in xrange(K):
            start = K*(K+1)/2 + i*(K+1)*(K+1)
            for a in xrange(K+1):
                for b in xrange(a+1, K+1):
                    x[start+a*(K+1)+b] = x[start+b*(K+1)+a]
    
        yp = y[:]
        Aopt_Gfunc( si2, x, y, alpha, beta, trans)
        blas.gemv( matrix(G), x, yp, 'T', alpha, beta)

        dy = np.max(np.abs(y - yp))
        if (dy > tol):
            print 'G function fails for trans=T: dy=%g' % dy
        else:
            print 'G function succeeds for trans=T: dy=%g' % dy

def test_sumdR2( ntrials=10, tol=1e-9):
    K = 40
    import time

    tnaive = tfast = 0.
    
    for t in xrange(ntrials):
        Ris = [ matrix(np.random.rand(K*K), (K,K)) for i in xrange(K) ]
        for i in xrange(K):
            Ris[i] = 0.5*(Ris[i].T + Ris[i])
        
        tstart = time.time()
        ddR2 = sumdR2( Ris, K)
        tend = time.time()
        tnaive += (tend - tstart)

        tstart = time.time()
        ddR2p = sumdR2_aligned( Ris, K)
        tend = time.time()
        tfast += (tend - tstart)

        delta = np.max(np.abs(ddR2 - ddR2p))
        if (delta > tol):
            print 'sum dR test FAILED: delta=%g > tol=%g' % (delta, tol)
        else:
            print 'sum dR test succeeds: delta=%g' % delta
    print 'Timing for naive sum dR: %f seconds per call.' % (tnaive/ntrials)
    print 'Timing for aligned sum dR: %f seconds per call.' % (tfast/ntrials)
    
if __name__ == '__main__':
    #np.random.seed( 11)
    #test_Gfunc(ntrials=10)
    #test_kkt_solver(ntrials=10)
    
    #test_sumdR2()
    #import sys
    #sys.exit()

    K = 30
    sij = matrix( np.random.rand( K*K), (K, K))
    nsofar = matrix( 0.2*np.random.rand( K*K), (K, K))
    sij = 0.5*(sij + sij.T)
    nsofar = 0.5*(nsofar + nsofar.T)

    if (False):
        connectivity = 5
        only_include_measurements = set()
        for i in xrange( K):
            js = i + np.floor((K-i)*np.random.rand(connectivity)).astype('int')
            for j in js:
                only_include_measurements.add( (i,j))
    else:
        only_include_measurements = None

    N = 1.

    import time
    tstart = time.time()
    nij = A_optimize_fast( sij, N, nsofar, only_include_measurements)
    tend = time.time()
    tlapse = tend - tstart
    print 'Fast A-optimize took %g seconds.' % tlapse
    # print nij

    if (K>=80):
        import sys
        sys.exit()

    tstart = time.time()
    nij0 = update_A_optimal_sdp( sij, N, nsofar, only_include_measurements)
    tend = time.time()
    tlapse = tend - tstart
    print 'SDP A-optimize took %g seconds.' % tlapse
    
    print 'dn=', np.max(np.abs( nij0 - nij))
