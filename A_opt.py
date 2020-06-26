import numpy as np
import cvxopt
from cvxopt import blas, lapack, solvers, matrix, spmatrix, misc
from diffnet import A_optimize, update_A_optimal

def upper_index( i, j, K):
    '''
    Return the position of the tuple (i,j) in the sequence (0,0), (0,1), ... 
    (0,K-1), (1,1), ... (K-1,K-1).  i.e., 0<i<=j<K.
    '''
    return K*i - i*(i+1)/2 + j

def solution_to_nij( x, K):
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

    # TODO: use symmetry!
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
    for i in xrange(n):
        y[:,k] = x[:,i]
        k+=1
        for j in xrange(i+1, n):
            y[:,k] = x[:,i] - x[:,j]
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

from cvxopt.misc_solvers import *
def kkt_ldl(G, dims, A, mnl = 0, kktreg = None):
    """
    Solution of KKT equations by a dense LDL factorization of the 
    3 x 3 system.
    
    Returns a function that (1) computes the LDL factorization of
    
        [ H           A'   GG'*W^{-1} ] 
        [ A           0    0          ],
        [ W^{-T}*GG   0   -I          ] 
    
    given H, Df, W, where GG = [Df; G], and (2) returns a function for 
    solving 
    
        [ H     A'   GG'   ]   [ ux ]   [ bx ]
        [ A     0    0     ] * [ uy ] = [ by ].
        [ GG    0   -W'*W  ]   [ uz ]   [ bz ]
    
    H is n x n,  A is p x n, Df is mnl x n, G is N x n where
    N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ).
    """
    
    p, n = A.size
    ldK = n + p + mnl + dims['l'] + sum(dims['q']) + sum([ int(k*(k+1)/2)
        for k in dims['s'] ])
    K = matrix(0.0, (ldK, ldK))
    ipiv = matrix(0, (ldK, 1))
    u = matrix(0.0, (ldK, 1))
    g = matrix(0.0, (mnl + G.size[0], 1))

    def factor(W, H = None, Df = None):

        blas.scal(0.0, K)
        if H is not None: K[:n, :n] = H
        K[n:n+p, :n] = A
        for k in range(n):
            if mnl: g[:mnl] = Df[:,k]
            g[mnl:] = G[:,k]
            scale(g, W, trans = 'T', inverse = 'I')
            pack(g, K, dims, mnl, offsety = k*ldK + n + p)
        K[(ldK+1)*(p+n) :: ldK+1]  = -1.0
        if kktreg:
            K[0 : (ldK+1)*n : ldK+1]  += kktreg  # Reg. term, 1x1 block (positive)
            K[(ldK+1)*n :: ldK+1]  -= kktreg     # Reg. term, 2x2 block (negative)
        lapack.sytrf(K, ipiv)

        for i in xrange(len(W['r'])):
            print 'r[%d]=' % i
            print W['r'][i]

        def solve(x, y, z):

            # Solve
            #
            #     [ H          A'   GG'*W^{-1} ]   [ ux   ]   [ bx        ]
            #     [ A          0    0          ] * [ uy   [ = [ by        ]
            #     [ W^{-T}*GG  0   -I          ]   [ W*uz ]   [ W^{-T}*bz ]
            #
            # and return ux, uy, W*uz.
            #
            # On entry, x, y, z contain bx, by, bz.  On exit, they contain
            # the solution ux, uy, W*uz.

            print 'x0, y0, z0='
            print x
            print y
            print z

            blas.copy(x, u)
            blas.copy(y, u, offsety = n)
            scale(z, W, trans = 'T', inverse = 'I') 
            pack(z, u, dims, mnl, offsety = n + p)
            lapack.sytrs(K, ipiv, u)
            blas.copy(u, x, n = n)
            blas.copy(u, y, offsetx = n, n = p)
            unpack(u, z, dims, mnl, offsetx = n + p)
    
            print 'x, y, z='
            print x
            print y
            print z

        return solve

    return factor

def Aopt_KKT_solver_naive( si2, W):
    '''
    Naive solution of the KKT equation, for debugging only.
    '''
    K = si2.size[0]
    ds = W['d']
    ris = W['r']

    nz = K*(K+1)/2 + K*(K+1)*(K+1)
    Wm = matrix( 0.0, (nz, nz))
    for i in xrange( K*(K+1)/2):
        Wm[i,i] = ds[i]

    offset = K*(K+1)/2
    for z in xrange(K):
        r = ris[z]
        congruence_matrix( r, Wm, offset)
        offset += (K+1)*(K+1)
    WtW = matrix( 0.0, Wm.size)
    blas.gemm( Wm, Wm, WtW, transA='T')

    for i in xrange(K):
        print 'r[%d] = ' % i
        print ris[i]
    #print 'W='
    #print Wm

    G, h, A = Aopt_GhA( si2)
    nvars = K*(K+1)/2 + K + 1 + K*(K+1)/2 + K*(K+1)*(K+1)
    Bm = matrix( 0.0, (nvars, nvars))
    Bm[:K*(K+1)/2+K, K*(K+1)/2+K] = A.T
    Bm[:K*(K+1)/2+K, K*(K+1)/2+K+1:] = G.T
    Bm[K*(K+1)/2+K,:K*(K+1)/2+K] = A
    Bm[K*(K+1)/2+K+1:,:K*(K+1)/2+K] = G
    Bm[K*(K+1)/2+K+1:,K*(K+1)/2+K+1:] = -WtW

    #print 'G='
    #print G

    #print 'A='
    #print A

    #print 'Bm='
    #print Bm

    Bm0 = matrix( Bm[:], Bm.size)

    Bmp = np.block( [
        [np.zeros( (A.size[1], A.size[1])), np.array(A.T), np.array(matrix(G.T)) ],
        [np.array(A), np.zeros((A.size[0], A.size[0])), np.zeros((A.size[0], G.size[0]))],
        [np.array(matrix(G)), np.zeros((G.size[0], A.size[0])), np.array(-WtW) ] ])
    
    #print 'dBm = ', np.max( np.abs(Bmp - Bm0))

    ipiv = matrix( 0, Bm.size)
    lapack.sytrf( Bm, ipiv)

    def kkt_solver( x, y, z):

        print 'x0, y0, z0='
        print x
        print y
        print z

        lab = z[:K*(K+1)/2]

        lmats = z[K*(K+1)/2:]
        for k in xrange(K):
            zp = matrix( lmats[k*(K+1)*(K+1):(k+1)*(K+1)*(K+1)], (K+1,K+1))
            for i in xrange(K+1):
                for j in xrange(i+1,K+1):
                    zp[i,j] = zp[j,i]
            lmats[k*(K+1)*(K+1):(k+1)*(K+1)*(K+1)] = zp[:]

        print 'z0='
        print lab
        for i in xrange(K):
            zmat = matrix( lmats[i*(K+1)*(K+1):(i+1)*(K+1)*(K+1)], (K+1,K+1))
            print zmat

        # Get the symmetric matrices 
        z[K*(K+1)/2:] = lmats[:]

        Cv = matrix( 0., (nvars, 1))
        Cv[:len(x)] = x
        Cv[len(x):len(x)+len(y)] = y
        Cv[len(x)+len(y):] = z
        lapack.sytrs( Bm, ipiv, Cv)

        xyz = matrix( 0., (nvars, 1))
        blas.gemv( Bm0, Cv, xyz)
        
        print 'dx, dy, dz=' 
        print np.max( np.abs(xyz[:len(x)] - x))
        print np.max( np.abs(xyz[len(x):len(x)+len(y)] - y))
        print np.max( np.abs(xyz[len(x)+len(y):] - z))

        x[:] = Cv[:len(x)]
        y[:] = Cv[len(x):len(x)+len(y)]
        zp = Cv[len(x)+len(y):]
        z2 = matrix( 0., (len(z), 1))
        blas.gemv( Wm, zp, z2)
        
        r = ris[0]
        K1 = r.size[0]
        offset = K*(K+1)/2
        for i in xrange(K):
            r = ris[i]
            print 'r='
            print r
            z3 = matrix( zp[offset:offset+K1*K1], (K1,K1))
            z4 = r.T * z3 * r
            print 'rt.z[%d].r = ' % i
            print z4
            print matrix( z2[offset:offset+K1*K1], (K1,K1))

            print 'rt.z.r - W.z =', np.max( np.abs(z4[:] - z2[offset:offset+K1*K1]))
            offset += K1*K1

        z[:] = z2
        
        print 'x,y,z='
        print x
        print y
        print z
        #import pdb
        #pdb.set_trace()

    return kkt_solver

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
    F = -cvxopt.mul(nij[:], si2[:])
    d = -np.sum( np.array( F).reshape( K,K), axis=0)
    F[::K+1] = d[:]
    return matrix( F, (K,K))

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

    # print 'd='
    # print ds

    # for rti in rtis: print rti
    if (False):
        for i, ri in enumerate(ris): 
            print 'r[%d]=' % i
            print ri

    # R_i = r_i^{-t}r_i^{-1} 
    Ris = [ matrix(0.0, (K+1, K+1)) for i in xrange(K) ]
    for i in xrange(K):
        blas.gemm( rtis[i], rtis[i], Ris[i], transB = 'T')

    # The coefficient for n_{ab} in the row (i,j) is given by
    #    R_{ai}^2 if a=b and i=j
    #    (R_{ai} - R_{bi})^2  if a!=b and i=j
    #    (R_{ai} - R_{aj})^2  if a=b and i!=j
    #    (R_{ai} + R_{bj} - R_{bi} - R_{aj})^2  if a!=b and i!=j
    #  * s_{ab}^{-2} * s_{ij}^{-2}
    ddR2 = matrix( 0., (K*(K+1)/2, K*(K+1)/2))
    for i in xrange(K):
        Ri = Ris[i]
        # dR[:(a,b)] = R[:,a] - R[:,b]
        dR = matrix( 0., (K, K*(K+1)/2))
        pairwise_diff( Ri[:K,:K], dR, K)
        # ddR[:(ap,bp)] = dR'[:,ap] - dR'[:,bp]
        ddR = matrix( 0., (K*(K+1)/2, K*(K+1)/2))
        pairwise_diff( dR.T, ddR, K)
        ddR2 += ddR**2

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
                import pdb
                pdb.set_trace()

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
                # offset = col*nrows + K*(K+1)/2
                # Gs[offset+i*(K+2) : (col+1)*nrows : (K+1)*(K+1)] = -si2[i,j]
                # Gs[offset+j*(K+2) : (col+1)*nrows : (K+1)*(K+1)] = -si2[i,j]
                # Gs[offset+i*(K+1)+j : (col+1)*nrows : (K+1)*(K+1)] = si2[i,j]
                # Gs[offset+j*(K+2)+i : (col+1)*nrows : (K+1)*(K+1)] = si2[i,j]
                col += 1
        
        # vec( [ 0, 0; 0, 1 ])
        
        Gs.extend( [ (skip+(i+1)*(K+1)*(K+1)-1, K*(K+1)/2+i, -1.)
                     for i in xrange(K) ])
        I, J, X = [ [ ijx[p] for ijx in Gs ] for p in xrange(3) ]
        G = spmatrix( X, I, J, (nrows, ncols))

    # h vector.
    h = np.zeros( K*(K+1)/2 + K*(K+1)*(K+1))
    # F := Fisher matrix
    if nsofar is not None:
        f = cvxopt.mul( nsofar, si2)
        F = np.diag( np.sum( f, axis=1))
        for i in xrange(K):
            f[i,i] = 0
        F -= f
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

def A_optimize_fast( sij, N=1., nsofar=None):
    '''
    '''
    # si2[i,j] := 1/s_{ij}^2
    si2 = cvxopt.div( 1., sij**2) 
    K = si2.size[0]

    Gm, hv, Am = Aopt_GhA( si2, nsofar, G_as_function=True)
    # print Gm
    # print hv
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
#                                       abstol=1e-3,
#                                       reltol=1e-3,
                                       feastol=1e-7),
#                          kktsolver=default_kkt_solver)
                           kktsolver=lambda W: Aopt_KKT_solver( si2, W))

    return solution_to_nij( sol['x'], K)
  
def test_congruence():
    r = matrix( [ 1., 2., 3., 4., 5., 6., 7., 8., 9.], (3,3))
    x = matrix( [ 0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.7, 0.8, 0.9 ], (3,3))

    # r = r[:2,:2]
    # x = x[:2,:2]

    K = r.size[0]
    W = matrix( 0.0, (K*K, K*K))
    congruence_matrix( r, W)
    
    y = r.T * x * r
    y2 = x[:]
    cngrnc( r, y2, r.size[0])
    yp = matrix( 0.0, (len(y[:]), 1))
    blas.gemv( W, x[:], yp)

    print yp - y[:]
    print yp - y2[:]

def test_kkt_solver( ntrials=100, tol=1e-6):
    sij = matrix( [[ 1., 0.1, 0.2, 0.5],
                   [ 0.1, 2., 0.3, 0.2],
                   [ 0.2, 0.3, 1.2, 0.1],
                   [ 0.5, 0.2, 0.1, 0.9]])
    K = 2
    sij = matrix( np.random.rand( K*K), (K, K))
    # sij = matrix( np.ones( K*K), (K, K))
    sij = 0.5*(sij + sij.T)
 
   #sij = sij[:2,:2]

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
        #ds[0] = 1e-6
        #ds[1:] = 1e6
        rs = [ matrix(np.random.rand( (K+1)*(K+1)) - 0.3, (K+1, K+1)) 
               for i in xrange(K) ]
        #rs = [ matrix( np.diag( np.random.rand(K+1)), (K+1, K+1))
        #       for i in xrange(K) ]
        #rs = [ matrix( np.diag( np.ones( K+1)), (K+1, K+1))
        #       for i in xrange(K) ]
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
        
        print 'dx, dy, dz=', dx, dy, dz
        # print z
        # print zp

        if tol < np.max( [dx, dy, dz]):
            import pdb
            pdb.set_trace()

def test_Gfunc( ntrials=100, tol=1e-10):
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

if __name__ == '__main__':
    # np.random.seed( 11)
    # test_congruence()
    # test_Gfunc()

    #test_kkt_solver(ntrials=100)

    #import sys
    #sys.exit()
    sij = matrix( [[ 1.5, 0.1, 0.2, 0.5],
                   [ 0.1, 1.1, 0.3, 0.2],
                   [ 0.2, 0.3, 1.2, 0.1],
                   [ 0.5, 0.2, 0.1, 0.9]])
    # sij = sij[:2,:2]
    #sij = matrix( [[1., 1], [1, 1.1]])
    K = 20
    sij = matrix( np.random.rand( K*K), (K, K))
    nsofar = matrix( 2.*np.random.rand( K*K), (K, K))
    sij = 0.5*(sij + sij.T)
    nsofar = 0.5*(nsofar + nsofar.T)

    N = 20.

    nij = A_optimize_fast( sij, N, nsofar)
    nij0 = update_A_optimal( sij, N, nsofar)
    print nij0
    print np.sum( nij0) + np.sum( np.diag( nij0))
    print nij
    print np.sum( nij) + np.sum( np.diag( nij))

    print 'dn=', np.max(np.abs( nij0 - nij))
