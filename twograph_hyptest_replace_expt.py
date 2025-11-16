
'''
Code for running an instance of the two-graph hypothesis testing experiment
explored in Figure 2 of Section 3.2 of the (EJS pre-revision) omni paper.
https://arxiv.org/pdf/1705.09355v3

We specify a number of vertices n, generate n latent positions
X_1,X_2,...,X_n iid F = Dir( (1,1,1) )
and generate an adjacency matrix A1 from them.
Then, we generate another adjacency matrix A2 with latent positions
Y_1,Y_2,...,Y_n generated according to
Y_i = Z_i if i \in I, X_i otherwise,
where Z_i are drawn iid from the same distribution as the X_j
and I \subset [n] is a set of size k <= n chosen uniformly at random.

We compare two different methods:
Procrustes-based, following Tang et al (2017) semiparametric testing
Omnibus embedding
'''

import numpy as np
import scipy as sp
import networks as net
import omni
import procrustes as proc
import embeddings

def run_2ght_expt( n, k, alpha=np.ones(3) ):
    '''
    Generate n latent positions drawn from Dirichlet( alpha )
    generate A1 from those latent positions.
    Choose k of those LPs uniformly at random, and redraw them from
    Dirichlet( alpha ), independently of the originals,
    and generate a new network A2.
    So A1 and A2 have the same latent positions save for k vertices.
    Our job is then to detect whether or not this is the case.

    Returns
    hyptest : dictionary that maps method names to 0/1 for acc/reject
    alpha : np.array, parameter to Dirichlet
    '''
    n = int(n)
    if n<1:
        raise ValueError('number of vertices n should be positive')
    k = int(k)
    if k<0:
        raise ValueError('number of flips k should be non-negative')
    if k > n:
        raise ValueError('number of flips cannot exceed n')

    if len(alpha.shape) != 1:
        raise ValueError('alpha should be a vector')
    d = len(alpha)

    # First, we need a function fo generating Dirichlet.
    def F(n):
        # Generate n independent draws from Dir( alpha )
        from scipy.stats import dirichlet
        return dirichlet.rvs( alpha, size= n )

    # generate the first network and its latent positions
    (A1,X) = net.gen_rdpg( n, F )
    # Choose k vertices to change and change those vertices
    Y = np.copy( X )
    if k>0:
        vxs_to_change = np.random.choice( n, k, replace=False )
        Y[ vxs_to_change, : ] = F( k )
    assert( Y.shape == X.shape )
    A2 = net.gen_adj_from_posns(Y)

    # Problem setup is done. Now do hypothesis testing.

    # Run each and record if it accepts/rejects
    hyptest = dict()
    hyptest['proc'] = proc.proc_hyptest( A1, A2, X )
    hyptest['omni'] = omni.omni_hyptest( A1, A2, X )

    return (hyptest, alpha)

if __name__ == '__main__':
    import sys

    if len( sys.argv ) != 3:
        pyscr = sys.argv[0]
        USAGE = '%s takes two arguments: python3 %s n k' % (pyscr, pyscr)
        raise RuntimeError( USAGE )    

    n = int( sys.argv[1] )
    print('n = %d' % n )
    k = int( sys.argv[2] )
    print('k = %d' % k )

    (test,alpha) = run_2ght_expt( n, k ) # Using default alpha=np.ones(3) 
    print('alpha = %s' % str( alpha ) )
    print('=======')
    print('Method\tReject?')

    # Now print the results
    for k,v in test.items():
        print('%s\t%d' % (k,v) )
