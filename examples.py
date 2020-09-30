from scipy import linalg
import cvxopt
from cvxopt import matrix
import networkx as nx

import sys
from diffnet import *

from graph import *

def distance_net( K, dim=2, rmin=0):
    '''
    For K randomly placed points in the space of the given dimension, generate 
    the distance matrix between K points, and their distances to the origin.

    Args:
    
    K: int - The number of points
    dim: int - the number of dimensions
    rmin: float - the minimum distance of each point from the origin

    Returns:

    dij: KxK matrix, dij[i][j] is the distance between i and j, dij[i][i] is
    the distance of i to the origin.
    '''

    x = 2.*(np.random.rand( K, dim) - 0.5)
    dij = matrix( 0., (K, K))
    
    for i in xrange( K):
        dij[i,i] = np.sqrt(x[i].dot(x[i]))
        if rmin>0 and dij[i,i]<rmin:
            x[i] *= (rmin/dij[i,i])
            dij[i,i] = rmin

    for i in xrange( K):
        for j in xrange( i+1, K):
            dx = x[i] - x[j]
            dij[i,j] = np.sqrt(dx.dot(dx))
            dij[j,i] = dij[i,j]

    return dij, x

def const_allocation( sij, allocation='std'):
    '''
    Allocate the sampling across the differences so either standard error,
    variance, or fraction of sampling is constant for all edges.

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
    if allocation == 'var':
        nij = cvxopt.mul( sij, sij)
    elif allocation == 'var':
        nij = sij[:,:]
    else:
        nij = matrix( 1., sij.size)
    s = sum_upper_triangle( nij)
    nij /= s
    return nij
    
def benchmark_diffnet( sij_generator, ntimes=100,
                       optimalities = ['D', 'A', 'Etree'],
                       constant_relative_error=False,
                       epsilon=1e-2):
    '''
    For each optimality, compute the reduction of covariance
    in the D-, A-, and E-optimal in reference to the minimum
    spanning tree.

    Args:
    sij_generator: function - sij_generator() generates a symmetric
    matrix of sij.

    Returns:
    ( stats, avg, topo ): tuple - stats['D'|'A'|'E'][o] is a numpy array
    of the covariance ratio ('D': ln(det(C)), 'A': tr(C), 'E': max(eig(C)))
    
    avg['D'|'A'|'E'][o] is the corresponding mean.

    topo[o][0] is the histogram of n_{ii}/s_{ii}.
    topo[o][1] is the histogram of n_{ij}/s_{ij} for j!=i.
    topo[o][2] is the list of connectivities of the measurement networks
    topo[o][3] is the list containing the numbers of edges that need to be 
       added to the measurement networks to make the graphs 2-edge-connected 
       (which ensures a cycle between any two nodes).
 
    o can be 'D', 'A', 'Etree', 'MSTn', 'MSTs', 'MSTv', 'cstn', 'cstv', 'csts'.
    '''
    stats = dict( D=dict(), A=dict(), E=dict())
    for s in stats:
        for o in optimalities + [ 'MSTn', 'MSTs', 'MSTv' ] + \
            [ 'cstn', 'csts', 'cstv' ]:
            stats[s][o] = np.zeros( ntimes)
    emin = -5
    emax = 2
    nbins = 2*(emax + 1 - emin)
    bins = np.concatenate( [ [0] , np.logspace( emin, emax, nbins) ])

    # topo records the topology of the optimal measurement networks
    topo = dict( [ (o, [np.zeros( nbins, dtype=float), 
                        np.zeros( nbins, dtype=float),
                        [], [] ]) for o in optimalities ])
    nfails = 0
    for t in xrange( ntimes):
        if constant_relative_error:
            results = dict()
            si, sij = sij_generator()
            for o in optimalities:
                if o=='A':
                    results[o] = A_optimize_const_relative_error( si)
                elif o=='D':
                    results[o] = D_optimize_const_relative_error( si)
                else:
                    results.update( optimize( sij, [o]))
        else:
            sij = sij_generator()
            results = optimize( sij, optimalities)
        ssum = np.sum( np.triu( sij))
        if None in results.values(): 
            nfails += 1
            continue
        for o in optimalities:
            n = np.array( results[o])
            n[n<0] = 0
            nos = ssum*n/sij
            d = np.diag( nos)
            u = [ nos[i,j] for i in xrange( n.shape[0])
                  for j in xrange( i+1, n.shape[0]) ]
            hd, _ = np.histogram( d, bins, density=False)
            hu, _ = np.histogram( u, bins, density=False)
            topo[o][0] += hd
            topo[o][1] += hu
            nos[nos<epsilon] = 0
            gdn = nx.from_numpy_matrix( nos)
            topo[o][2].append( nx.edge_connectivity( gdn))
            topo[o][3].append( len(sorted(nx.k_edge_augmentation( gdn, 2))))

        results.update( dict(
            MSTn = MST_optimize( sij, 'n'),
            MSTs = MST_optimize( sij, 'std'),
            MSTv = MST_optimize( sij, 'var')))
        results.update( dict(
            cstn = const_allocation( sij, 'n'),
            csts = const_allocation( sij, 'std'),
            cstv = const_allocation( sij, 'var')))
        CMSTn = covariance( cvxopt.div( results['MSTn'], sij**2))
        DMSTn = np.log(linalg.det( CMSTn))
        AMSTn = np.trace( CMSTn)
        EMSTn = np.max(linalg.eig( CMSTn)[0]).real
        for o in results:
            n = results[o]
            C = covariance( cvxopt.div( n, sij**2))
            D = np.log(linalg.det( C))
            A = np.trace( C)
            E = np.max(linalg.eig( C)[0]).real
            stats['D'][o][t-nfails] = D - DMSTn
            stats['A'][o][t-nfails] = A/AMSTn
            stats['E'][o][t-nfails] = E/EMSTn
    
    avg = dict()
    for s in stats:
        avg[s] = dict()
        for o in stats[s]:
            stats[s][o] = stats[s][o][:ntimes-nfails]
            avg[s][o] = np.mean( stats[s][o])

    for o in optimalities:
        topo[o][0] /= (ntimes - nfails)
        topo[o][1] /= (ntimes - nfails)
    return stats, avg, topo

def benchmark_distance_net( K=30, rmin=0.2, dim=2, ntimes=100):
    def sij_generator():
        return distance_net( K, dim, rmin=rmin)[0]

    return benchmark_diffnet( sij_generator, ntimes)

def benchmark_const_rel_net( K=30, ntimes=100):
    def sij_generator():
        si = np.random.rand( K)
        si = np.sort( si)
        sij = constant_relative_error( si)
        return si, sij

    return benchmark_diffnet( sij_generator, ntimes, constant_relative_error=True)

def random_net_sij_generator( K=30, sii_offset=0., sij_min=1., sij_max=5.):
    sij = matrix( (sij_max-sij_min)*np.random.rand(K, K)+sij_min, (K, K))
    sij = 0.5*(sij + sij.trans())
    sij += matrix( sii_offset*np.diag( np.ones( K)), (K, K))
    return sij
    
def benchmark_random_net( K=30, sii_offset=0., sij_min=1., sij_max=5., 
                          ntimes=100):
    def sij_generator():
        return random_net_sij_generator( K, sii_offset, sij_min, sij_max)

    return benchmark_diffnet( sij_generator, ntimes)

def benchmark_sparse_net( K=30, measure_per=3, connectivity=3, 
                          sii_offset=0., sij_min=1., sij_max=5.,
                          ntimes=100):
    nsofar = np.zeros( (K, K))
    nadd = 1
    ratio = np.zeros( ntimes)
    n_measure = int(measure_per*K)
    ncutoff = 1./(10.*n_measure)
    for t in xrange( ntimes):
        sij = random_net_sij_generator( K, sii_offset, sij_min, sij_max)
        nij = A_optimize( sij)
        trC = np.trace( covariance( cvxopt.div(nij, sij**2)))
        nijp = sparse_A_optimal_network( sij, nsofar, nadd, n_measure, connectivity)
        nijp = np.asarray( nijp)
        nijp[nijp < ncutoff] = 0
        nijp = matrix( nijp)
        trCp = np.trace( covariance( cvxopt.div(nijp, sij**2)))
        ratio[t] = trCp/trC
        
    return np.mean(ratio), np.std(ratio)

def analyze_uniform_net( pmax=6, dmax=25., Nd=20):
    
    d = np.linspace( 0., dmax, Nd)

    stats = dict(
        diag = np.zeros( (pmax, len(d)), dtype=float),
        vardiag = np.zeros( (pmax, len(d)), dtype=float),
        offdiag = np.zeros( (pmax, len(d)), dtype=float),
        varoffdiag = np.zeros( (pmax, len(d)), dtype=float))

    def triu( A):
        return [ A[i,j] for i in xrange(A.size[0]) 
                 for j in xrange(i+1, A.size[0]) ]

    ps = np.arange(1, pmax+1)
    for p in ps:
        K = 1<<p
        for j, offset_origin in enumerate( d):
            sij = np.ones( (K,K), dtype=float)
            sij += np.diag( offset_origin*np.ones( K))
            sij = matrix( sij)
            results = optimize( sij, ['A'])
            nij = results['A']
            ndiag = np.diag( nij)
            stats['diag'][p-1,j] = np.mean( ndiag)
            stats['vardiag'][p-1,j] = np.var( ndiag)
            noffd = triu( nij)
            stats['offdiag'][p-1,j] = np.mean( noffd)
            stats['varoffdiag'][p-1,j] = np.var( noffd)

    return dict(K=1<<ps, d=d, stats=stats)

def benchmark_E_tree( K=30, ntimes=100):
    import time
    timings = dict(Etree=np.zeros( ntimes), E=np.zeros( ntimes))
    dn = 0.
    for t in xrange( ntimes):
        sij = random_net_sij_generator(K=K)
        start = time.time()
        nijEt = E_optimal_tree( sij)
        end = time.time()
        timings['Etree'][t] = start - end
        start = time.time()
        nijE = E_optimize( sij)
        end = time.time()
        timings['E'][t] = start - end
        dn += np.sum( np.square(nijE - nijEt))
    dn /= ntimes
    dn = np.sqrt( dn)
    return timings, dn

import argparse

def opts():
    parser = argparse.ArgumentParser(
        description='Benchmark the statistical performance of the optimizers of the difference network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '-K', '--num-points', type=int, default=30,
                         help='Number of points in the network.')
    parser.add_argument( '-T', '--num-times', type=int, default=100,
                         help='Number of times to run the benchmark to collect statistics.')
    parser.add_argument( '--out-distance-net', default=None,
                         help='Name of pickle file to write benchmark results for distance net.')
    parser.add_argument( '--out-const-rel-net', default=None,
                         help='Name of pickle file to write benchmark results for constant relative error net.')
    parser.add_argument( '--out-random-net', default=None,
                         help='Name of pickle file to write benchmark results for random net.')
    parser.add_argument( '--sii-offset', type=float, default=0.,
                         help='The average offset of s_{ii} from s_{ij}.')
    parser.add_argument( '--sij-min', type=float, default=1.,
                         help='Minimum s_{ij}')
    parser.add_argument( '--sij-max', type=float, default=5.,
                         help='Maximum s_{ij}')
    parser.add_argument( '--out-uniform-net', default=None,
                         help='Name of pickle file to write analysis results for uniform net.')
    parser.add_argument( '--out-E-tree-timings', default=None,
                         help='Name of pickle file to write benchmark of E-tree and E-optimal timings.')
    parser.add_argument( '--out-sparse-net', action='store_true', default=False,
                         help='Benchmark of sparse A-optimal efficiency.')
    parser.add_argument( '--connectivity', type=int, default=2,
                        help='Connectivity requirement for sparse network.')
    parser.add_argument( '--measure-per-quantity', type=float, default=3,
                         help='Have at least this many measurements per quantity.')
    return parser

def write_average( avg):
    header = avg.keys()
    rows = avg[header[0]].keys()
    data = np.zeros( (len(avg[avg.keys()[0]]), len(avg)))
    for i, s in enumerate( avg):
        for j, o in enumerate( avg[s]):
            data[j,i] = avg[s][o]
    print '# %s' % (' '.join( [ '%5s' % s for s in header ]))
    for j in xrange(data.shape[0]):
        print '  ' + (' '.join( [ '%5.2f' % r for r in data[j] ])),
        print ' # %s' % rows[j]
    
def main( args):
    import cPickle as pickle
    if args.out_distance_net is not None:
        print 'Benchmarking diffnet for distance nets...'
        stats, avg, topo = benchmark_distance_net( args.num_points, ntimes=args.num_times)
        pickle.dump( dict(stats=stats, topo=topo), 
                     file( args.out_distance_net, 'wb'))
        write_average( avg)
    if args.out_random_net is not None:
        print 'Benchmarking diffnet for random nets...'
        stats, avg, topo = benchmark_random_net( args.num_points, args.sii_offset, args.sij_min, args.sij_max, ntimes=args.num_times)
        pickle.dump( dict(stats=stats, topo=topo), 
                     file( args.out_random_net, 'wb'))
        write_average( avg)
    if args.out_const_rel_net is not None:
        print 'Benchmarking diffnet with constant relative errors...'
        stats, avg, topo = benchmark_const_rel_net( args.num_points, ntimes=args.num_times)
        pickle.dump( dict(stats=stats, topo=topo),
                     file( args.out_const_rel_net, 'wb'))
        write_average( avg)
    if args.out_uniform_net is not None:
        print 'Analyzing diffnet for uniform nets...'
        results = analyze_uniform_net()
        pickle.dump( results, file( args.out_uniform_net, 'wb'))
    if args.out_E_tree_timings is not None:
        print 'Benchmark E-tree timings...'
        timings, dn = benchmark_E_tree( args.num_points, args.num_times)
        print 'timings(E)/timings(E-tree) = %.2f' % np.mean(timings['E']/timings['Etree'])
        print '|dn| = %g' % dn
    if args.out_sparse_net:
        print 'Benchmarking sparse A-optimal network...'
        results = benchmark_sparse_net( args.num_points, args.measure_per_quantity, args.connectivity, args.sii_offset, args.sij_min, args.sij_max, args.num_times)
        print 'mean ratio: %.3f +/- %.3f' % results

if __name__ == '__main__':
    main( opts().parse_args())

    
    
