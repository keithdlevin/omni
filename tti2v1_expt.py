
'''
Code for running an instance of the latent position estimation task
explored in Figure 1 of the (EJS pre-revision) omni paper,
but using 2-to-infty error instead of Frobenius.
https://arxiv.org/pdf/1705.09355v3

We specify a number of vertices n, generate two graphs A1 and A2 with the same
set of latent positions X (n-by-d) with rows drawn iid from Dir( [1,1,1] ),
and compare the performance of:

ASE1: we embed only one of two graphs
Abar: we take the mean and embed (this is the gold standard if all you care
	about is estimation)
omni: omnibus embedding yields 2n embeddings; we keep the first n
	as our estimate of X.
omnibar: take the omnibus embedding and use the mean of the first n
	and second n latent positions
procbar: embed A1 and A2 separately, align them via Procrustes,
	and average the aligned embeddings.
'''

import numpy as np
import scipy as sp
import networks as net
import omni
import procrustes as proc
import embeddings

def run_lpest_expt( n, alpha=np.ones(3) ):
    '''
    Generate latent positions and two networks with same LPs
    drawn from Dirichlet( alpha )

    Returns
    MSE : dictionary, keys on method names, values are MSE
    '''
    n = int(n)
    if n<1:
        raise ValueError('number of vertices n should be positive')

    if len(alpha.shape) != 1:
        raise ValueError('alpha should be a vector')
    d = len(alpha)

    # First, we need a function fo generating Dirichlet.
    def F(n):
        # Generate n independent draws from Dir( alpha )
        from scipy.stats import dirichlet
        return dirichlet.rvs( alpha, size= n )
    # generate 2 networks with same set of LPs drawn from F
    (A,Xtrue) = net.gen_multirdpg( n, 2, F )

    # Obtain our different estimates.
    # Estimate from just one from the networks.
    Xhat_ASE1 = embeddings.ase( A[:,:,0], d )
    Xhat_ASE2 = embeddings.ase( A[:,:,1], d )
    # Align these to get procbar
    Xhat_procbar = proc.procbar( Xhat_ASE1, Xhat_ASE2 )
    # Estimate from the mean of the networks
    Xhat_Abar = embeddings.asebar( A, d )
    # Get the omni embedding and build omni and omnibar.
    Zhat = omni.omni_embed( A, d )
    Xhat_omni = Zhat[:n,:]
    Xhat_omnibar = (Zhat[:n,:]+Zhat[n:,:])/2

    # Compute the estimation error of each of our estimates.
    ttierr = dict()
    ttierr['ASE1'] = proc.proc_tti( Xhat_ASE1, Xtrue )
    ttierr['Abar'] = proc.proc_tti( Xhat_Abar, Xtrue )
    ttierr['procbar'] = proc.proc_tti( Xhat_procbar, Xtrue )
    ttierr['omni'] = proc.proc_tti( Xhat_omni, Xtrue )
    ttierr['omnibar'] = proc.proc_tti( Xhat_omnibar, Xtrue )

    return (ttierr, alpha)

if __name__ == '__main__':
    import sys

    if len( sys.argv ) != 2:
        pyscr = sys.argv[0]
        USAGE = '%s takes one argument: python3 %s n' % (pyscr, pyscr)
        raise RuntimeError( USAGE )    

    n = int( sys.argv[1] )
    print('n = %d' % n )

    (ttierr,alpha) = run_lpest_expt( n ) # Using default alpha=np.ones(3) 
    print('alpha = %s' % str( alpha ) )
    print('=======')
    print('Method\tttiErr')

    # Now print the results
    for k,v in ttierr.items():
        print('%s\t%f' % (k,v) )
