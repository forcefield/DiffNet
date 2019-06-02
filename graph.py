import numpy as np
import networkx as nx
from cvxopt import matrix

import sys
import diffnet as dn

def diffnet_to_graph( A, origins='O'):
    '''
    Construct a graph of K+1 nodes from the KxK symmetric matrix A,
    where the weight of edge (i,j) is given by A[i][j], and the weight
    of edge (K=origin, i) is given by A[i][i].
    '''
    A = np.array( A)
    g = nx.from_numpy_matrix( A)
    if type(origins)==list:
        originIDs = list(set( origins))
        originIDs.sort()
        for o in originIDs:
            g.add_node( o)
    else:
        if origins is None: origins = 'O'
        g.add_node( origins)
        origins = [ origins ]*A.shape[0]
    for i in xrange( A.shape[0]):
        if A[i][i] != 0:
            g.remove_edge( i, i)
        g.add_edge( origins[i], i, weight=A[i][i])
    return g

def diffnet_connectivity( n, src=None, tgt=None):
    '''
    Compute the connectivity in the difference network of n[i,j].
    If src and tgt are given, compute the local connectivity between
    the src and tgt nodes.
    '''
    g = nx.from_numpy_matrix( n)
    return nx.edge_connectivity( g, src, tgt)

def draw_diffnet_graph( g, pos=None, ax=None,
                        widthscale=None, nodescale=2.5, node_color=None,
                        origins=['O']):
    '''
    Draw a graph representing the difference network.
    
    Args:
    g: nx.Graph - the graph representing the difference network.
    pos: Kx2 numpy array or dict - the coordinates to place the nodes 
    in the graph. If numpy array, pos[i] is the coordinates for node i,
    excluding origin.  If dict, pos[i] is the coordinate of node i, including
    origin. If None, use a spring layout.

    Returns:
    pos: dict - pos[i] gives the positions of the node i.
    '''
    K = g.number_of_nodes() - len(origins)

    if isinstance( pos, np.ndarray):
        mypos = dict( [(i, pos[i]) for i in xrange(K)])
        if (len(pos) == K):
            for d, o in enumerate(origins):
                mypos.update( {o : (-1.0*d, -1.0*d)})
        else:
            for d, o in enumerate(origins):
                mypos.update( {o : pos[K+d]})
    elif type( pos) == dict:
        mypos = pos
    else:
        mypos = nx.spring_layout( g)
    
    node_size = nodescale*K
    if node_color is None:
        node_color = 'red'
    nx.draw_networkx_nodes( g, mypos, nodelist=range(K), 
                            node_size=node_size,
                            node_color=node_color,
                            ax=ax)
    nodeO = nx.draw_networkx_nodes( g, mypos, nodelist=origins,
                                    node_size=node_size*2,
                                    node_color='#FFFFFF',
                                    width=2.,
                                    ax=ax)
    if node_color is None or len(node_color)<=K:
        nodeO.set_edgecolor( 'red')
    else:
        nodeO.set_edgecolor( node_color[K:])

    if widthscale is None:
        widthscale = 5.*K

    weights = np.array( [ w for u, v, w in list(g.edges( data='weight')) ])
    weights[weights<0] = 0 # Set negative numbers to 0.
    width = weights*widthscale
    nx.draw_networkx_edges( g, mypos, 
                            width=width,
                            ax=ax)
    return mypos

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
    G = diffnet_to_graph( sij)
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
    s = dn.sum_upper_triangle( n)
    n *= (1./s)  # So that \sum_{i<=j} n_{ij} = 1
    return n
    
