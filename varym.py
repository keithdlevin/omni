
'''
Code for running an experiment varying the number of networks m
and the number of vertices.

We specify a number of vertices n and generate m graphs with the same
set of latent positions X (n-by-d) with rows drawn iid from Dir( [1,1,1] ),
and compare the performance of:

ASE1: we embed only one of the graphs
Abar: we take the mean and embed (this is the gold standard if all you care
	about is estimation)
omni: omnibus embedding yields mn embeddings; we keep the first n
	as our estimate of X.
omnibar: take the omnibus embedding and use the mean of the m different sets
    of latent position estimates
procbar: embed each of the networks separately, align them via Procrustes,
	and average the aligned embeddings.
'''

import numpy as np
import scipy as sp
import networks as net
import omni
import procrustes as proc
import embeddings

def run_varym_expt( n, m, alpha=np.ones(3) ):
    '''
    Generate latent positions and m networks with same LPs
    drawn from Dirichlet( alpha )

    Returns
    MSE : dictionary, keys on method names, values are MSE
    '''
    n = int(n)
    if n<1:
        raise ValueError('number of vertices n should be positive')

    m = int(m)
    if m<1:
        raise ValueError('number of networks m should be positive')

    if len(alpha.shape) != 1:
        raise ValueError('alpha should be a vector')
    d = len(alpha)

    # First, we need a function fo generating Dirichlet.
    def F(n):
        # Generate n independent draws from Dir( alpha )
        from scipy.stats import dirichlet
        return dirichlet.rvs( alpha, size= n )
    # generate m networks with same set of LPs drawn from F
    (A,Xtrue) = net.gen_multirdpg( n, m, F )

    # Obtain our different estimates.
    # Estimate from just one from the networks.
    Xhat_marginal_ASE = np.zeros( (n,d,m) ) # Store the m separate embeddings
    for k in range(m):
        Xhat_marginal_ASE[:,:,k] = embeddings.ase( A[:,:,k], d )
    # Take the first one to get Xhat_ASE1
    Xhat_ASE1 = Xhat_marginal_ASE[:,:,0]
    # Align these to get procbar
    Xhat_procbar = proc.procbar_many_to_one( Xhat_marginal_ASE )
    # Estimate from the mean of the networks
    Xhat_Abar = embeddings.asebar( A, d )
    # Get the omni embedding and build omni and omnibar.
    Zhat = omni.omni_embed( A, d )
    Xhat_omni = Zhat[:n,:]
    Xhat_omnibar = omni.omni_mean( Zhat, m )

    # Compute the estimation error of each of our estimates.
    MSE = dict()
    MSE['ASE1'] = proc.proc_mse( Xhat_ASE1, Xtrue )
    MSE['Abar'] = proc.proc_mse( Xhat_Abar, Xtrue )
    MSE['procbar'] = proc.proc_mse( Xhat_procbar, Xtrue )
    MSE['omni'] = proc.proc_mse( Xhat_omni, Xtrue )
    MSE['omnibar'] = proc.proc_mse( Xhat_omnibar, Xtrue )

    return (MSE, alpha)

if __name__ == '__main__':
    import sys

    if len( sys.argv ) != 3:
        pyscr = sys.argv[0]
        USAGE = '%s takes two arguments: python3 %s n m' % (pyscr, pyscr)
        raise RuntimeError( USAGE )    

    n = int( sys.argv[1] )
    m = int( sys.argv[2] )
    print('n = %d, m = %d\n' % (n,m) )

    (MSE,alpha) = run_varym_expt( n, m ) # Using default alpha=np.ones(3) 
    print('alpha = %s' % str( alpha ) )
    print('=======')
    print('Method\tMSE\tFrobNormSqrd')

    # Now print the results
    for k,v in MSE.items():
        print('%s\t%f\t%f' % (k,v**2,n*(v**2) ) )
