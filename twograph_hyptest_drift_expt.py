
'''
Code for running an instance of the two-graph hypothesis testing experiment
explored in Figure 3 of Section 3.2 of the (EJS pre-revision) omni paper.
https://arxiv.org/pdf/1705.09355v3

We consider a 3-dimensional RDPG on n vertices, in which one
latent position, i, is fixed to be equal to xi = (0.8, 0.1, 0.1)T and the
remaining latent positions are drawn i.i.d. from a Dirichlet with parameter
alpha = (1,1,1).
We collect these latent positions in the rows of the matrix X.
To produce the latent positions Y of the second graph, we use the same
latent positions in X, but we alter the i-th position to be
Yi = (1 − lambda)xi + lambda*(0.1, 0.1, 0.8)T , where lambda \in [0, 1]
serves as a “drift” parameter controlling how
much the latent position changes between the two graphs.

We compare two different methods:
-Procrustes-based, following Tang et al (2017) semiparametric testing
-Omnibus embedding
'''

import numpy as np
import scipy as sp
import networks as net
import omni
import procrustes as proc
import embeddings

def run_2ght_expt( n, lam ):
    '''
    1) Generate n latent positions drawn from Dirichlet( 1,1,1 )
    2) Choose one of these latent positions, call it xi,
	 and set it to (0.8,0.1,...,0.1)
    Generate A1 from those.
    Then we replace xi with
	yi = (1 − lambda)xi + lambda*(0.1, ..., 0.1, 0.8)
    and generate A2 from that new set of LPs.

    Returns
    hyptest : dictionary that maps method names to 0/1 for acc/reject
    alpha : np.array, parameter to Dirichlet
    '''
    n = int(n)
    if n<1:
        raise ValueError('number of vertices n should be positive')
    if not isinstance( lam, float ):
        raise TypeError('lam should be a float.')
    if lam < 0 or lam > 1:
        raise ValueError('lam should be between 0 and 1.')

    # Set dimension globally in case we want to change this to being
    # an argument later.
    d = 3
    alpha = np.ones( d )

    # First, we need a function fo generating Dirichlet.
    def F(n):
        # Generate n independent draws from Dir( alpha )
        from scipy.stats import dirichlet
        return dirichlet.rvs( alpha, size= n )

    # generate the first network and its latent positions
    (A1,X) = net.gen_rdpg( n, F )

    # Choose which vertex to change
    i = int( np.random.choice( n, 1 ) )
    # Construct the new latent position
    # Its first entry is 0.8, rest are 0.1
    xi = 0.1*np.ones(d); xi[0] = 0.8
    X[i,:] = xi
   
    # The second network has the same latent positions EXCEPT i 
    Y = np.copy( X )
    # Drift is a linear comb of xi and (0.1,0.1,...0.1,0.8)
    antipode = 0.1*np.ones(d); antipode[-1] = 0.8
    Y[i,:] = (1-lam)*xi + lam*antipode
    A2 = net.gen_adj_from_posns(Y)

    # Problem setup is done. Now do hypothesis testing.

    # Run each and record if it accepts/rejects
    hyptest = dict()
    hyptest['proc'] = proc.proc_hyptest( A1, A2, X )
    hyptest['omni'] = omni.omni_hyptest( A1, A2, X )

    return hyptest

if __name__ == '__main__':
    import sys

    if len( sys.argv ) != 3:
        pyscr = sys.argv[0]
        USAGE = '%s takes two arguments: python3 %s n lam' % (pyscr, pyscr)
        raise RuntimeError( USAGE )    

    n = int( sys.argv[1] )
    print('n = %d' % n )
    lam = float( sys.argv[2] )
    print('lam = %f' % lam )

    test = run_2ght_expt( n, lam ) 
    print('=======')
    print('Method\tReject?')

    # Now print the results
    for k,v in test.items():
        print('%s\t%d' % (k,v) )
