
'''
Code for doing permutation testing
'''

import numpy as np
import networks as net

def generate_resamples( Xtrue, Tstat_fn, nMC ):
    '''
    Xtrue : n-by-d latent positions
    Tstat_fn : function that takes a pair of networks and returns
		a test statistic
    nMC : positive int, number of MC iterates

    Return
    samps : nMC-long vector whose elements are iid draws of Tstat_fn(A1,A2)
		where A1,A2 are drawn from same LPs Xtrue.
    '''

    if len( Xtrue.shape ) != 2:
        raise ValueError('X should be a matrix')

    nMC = int(nMC)
    if nMC <= 0:
        raise ValueError('nMC should be a positive integer')

    samps = np.zeros( nMC )
    for i in range(nMC):
        # Draw two networks from the same latent positions and compute
        # the test statistic on that pair.
        A1 = net.gen_adj_from_posns(Xtrue)
        A2 = net.gen_adj_from_posns(Xtrue)
        samps[i] = Tstat_fn( A1, A2 )

    return samps

def do_resample_test( Tobsd, Xtrue, Tstat_fn, nMC, lvl ):
    '''
    Perform a test for Tstat_fn at given level lvl
    based on nMC draws of pairs of networks from Xtrue.

    Tobsd : float, observed value of test stat.
    Xtrue : n-by-d, true latent positions
    Tstat_fn : takes two networks and returns test statistic
    nMC : positive int, number of samples to generate
    lvl : float, level of the test

    Assumes that larger values of the test statistic correspond to more
	extreme observations.
    '''
    if not isinstance( Tobsd, (int,float) ):
        raise TypeError('Tobsd should be numeric')

    if len( Xtrue.shape ) != 2:
        raise ValueError('X should be a matrix')

    nMC = int(nMC)
    if nMC <= 0:
        raise ValueError('nMC should be a positive integer')

    if lvl < 0 or lvl > 1:
        raise ValueError('Level lvl should be between 0 and 1')

    samps = generate_resamples( Xtrue, Tstat_fn, nMC )
    pval = np.mean( samps > Tobsd )
    assert( int( True )==1 ); assert( int(False)==0 )
    return int( pval < lvl ) # return 1 if we reject, 0 otherwise

