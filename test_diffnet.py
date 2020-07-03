import numpy as np
from cvxopt import matrix
from diffnet import *
import netbfe
import A_opt

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

def check_update_A_optimal( sij, delta=5e-1, ntimes=10, tol=1e-5):
    '''
    '''
    K = matrix(sij).size[0]

    ntotal = 100
    fopt = A_optimize( sij)
    nopt = ntotal*fopt
    # remove some random samples from the optimal
    nsofar = nopt - nopt*0.1*np.random.rand( K, K)
    nsofar = matrix( 0.5*(nsofar + nsofar.T))
    nadd = ntotal - sum_upper_triangle( nsofar)
    nnext = A_optimize( sij, nadd, nsofar)
    success1 = True
    if np.abs(sum_upper_triangle( matrix(nnext)) - nadd) > tol:
        print 'Failed to allocate additional samples to preserve the sum!'
        print '|%f - %f| > %f' % (sum_upper_triangle( matrix(nnext)), nadd, tol)
        success1 = False
    # The new samples and the existing samples should together make up the 
    # optimal allocation.
    delta = sum_upper_triangle( abs( nnext + nsofar - nopt))/ntotal
    delta /= (0.5*K*(K+1))
    if delta > tol:
        print 'Failed: Updating allocation does not yield A-optimal!'
        print 'delta = %f > %f' % (delta, tol)
        success1 = False

    sij0 = np.random.rand( K, K)
    sij0 = matrix(0.5*(sij0 + sij0.T))

    nsofar = 100*A_optimize( sij0)

    nadd = 100
    # nnext = update_A_optimal_sdp( sij, nadd, nsofar)
    nnext = A_optimize( sij, nadd, nsofar)
    ntotal = matrix( nsofar + nnext)

    C = covariance( matrix(sij), ntotal/sum_upper_triangle(ntotal))
    trC = np.trace( C)

    dtr = np.zeros( ntimes)
    for t in xrange( ntimes):
        zeta = matrix( 1. + 2*delta*(np.random.rand( K, K) - 0.5))
        nnextp = cvxopt.mul( nnext, zeta)
        nnextp = 0.5*(nnextp + nnextp.trans())
        s = sum_upper_triangle( nnextp)
        nnextp *= (nadd/sum_upper_triangle( nnextp))
        ntotal = matrix( nsofar + nnextp)
        Cp = covariance( matrix(sij), ntotal/sum_upper_triangle(ntotal))
        dtr[t] = np.trace( Cp) - trC

    success2 = np.all( dtr[np.abs(dtr/trC) > tol] >= 0)
    # success2 = np.all( dtr >= 0)
    if not success2:
        print 'Iterative update of A-optimal failed to minimize tr(C)=%f!' % trC
        print dtr
    
    nnext = round_to_integers( nnext)
    if sum_upper_triangle( matrix(nnext)) != nadd:
        print 'Failed to allocate additional samples to preserve the sum!'
        print '%d != %d' % (sum_upper_triangle( matrix(nnext)), nadd)
        success2 = False

    return success1 and success2

def check_sparse_A_optimal( sij, ntimes=10, delta=1e-1, tol=1e-5):
    '''
    '''
    sij = matrix( sij)
    K = sij.size[0]
    nsofar = np.zeros( (K, K))
    nadd = 1.

    nopt = A_optimize( sij)
    nij = sparse_A_optimal_network( sij, nadd, nsofar, 0, K, False)
    
    success = True

    deltan = sum_upper_triangle( abs(nopt - nij))/(0.5*K*(K+1))
    if deltan > tol:
        print 'FAIL: sparse optimization disagree with dense optimzation.'
        print '| n - nopt | = %g > %g' % (deltan, tol)
        success = False
    else:
        print 'SUCCESS: sparse optimization agrees with dense optimization.'
        print '| n - nopt | = %g <= %g' % (deltan, tol)

    n_measures = 8
    connectivity = 2
    nij = sparse_A_optimal_network( sij, nadd, nsofar, n_measures, connectivity,
                                    True)
    print nij
    trC = np.trace( covariance( sij, nij))

    dtr = np.zeros( ntimes)
    for t in xrange( ntimes):
        zeta = matrix( 1. + 2*delta*(np.random.rand(  K, K) - 0.5))
        nijp = cvxopt.mul( nij, zeta)
        nijp = 0.5*(nijp + nijp.trans()) # Symmetrize
        s = sum_upper_triangle( nijp)
        nijp *= nadd/s
        
        trCp = np.trace( covariance( sij, nijp))
        dtr[t] = trCp - trC
    
    success2 = np.all( dtr >= 0)
    if not success2:
        print 'FAIL: sparse optimization fail to minimize.'
        print dtr
    else:
        print 'SUCCESS: sparse optimization minimizes.'

    return success and success2

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
        K = 10
        sij = np.random.rand( K, K)
        sij = matrix( 0.5*(sij + sij.T))
        # nij = A_optimize( sij)
        nij = sparse_A_optimal_network( sij )

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
    
    # Check iteration update
    success = check_update_A_optimal( sij)
    if success:
        print 'Iterative update of A-optimal passed!'
    
    # Check sparse A-optimal
    if (check_sparse_A_optimal( sij)):
        print 'Sparse A-optimal passed!'

if __name__ == '__main__':
    unitTest()
    A_opt.unit_test()
    netbfe.unit_test()
