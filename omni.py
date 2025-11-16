
import numpy as np
import scipy.linalg as spla

import networks as net
import embeddings
import resample as resamp

def construct_omni_matrix( A ):
    '''
    A : n-by-n-by-m array, A[:,:,i] is the adjm of i-th network.

    Returns
    M : nm-by-nm omnibus matrix
    '''

    if len( A.shape ) != 3:
        raise ValueError('A should be a 3-tensor')

    (_,nvx,nsubj) = A.shape
    if A.shape[0] != nvx:
        raise ValueError('First two dimensions of A should match')

    M = np.zeros( (nsubj*nvx, nsubj*nvx) )

    for s in range(nsubj):
        for t in range(nsubj):
            # Average networks s and t
            B = ( A[:,:,s] + A[:,:,t] )/2;
            # Pick out the corresponding rows and columns of M
            # and populate them accordingly.
            i = nvx*s;
            j = nvx*t;
            M[ i:(i+nvx), j:(j+nvx) ] = B;

    net.check_valid_adjmx( M ) # verify that it's symmetric and all that.

    return M

def omni_embed( A, d ):
    '''
    Construct the d-dimensional omnibus embedding from array A.
    A : n-by-n-by-m array representing m n-by-n matrices.
    d : positive integer representing embedding dimension.
    Returns
    Xhat : nm-by-d matrix, (s*n+i)-th row is embedding of i-th vertex
		as it appears in the s-th network.
    '''

    if not isinstance( d, int ):
        raise TypeError('d should be an integer')
    if d < 1:
        raise ValueError('d should be positive')

    # Construct omnibus matrix from A.
    M = construct_omni_matrix( A )

    # Now do ASE of M
    return embeddings.ase(M,d)

def omni_hyptest( A1, A2, Xtrue, lvl=0.05 ):
    '''
    Run the omnibus-based hypothesis test as discussed in Sec 3.2
    of the omnibus paper.
    For the purposes of this test, we assume that we have access to the true
    latent positions Xtrue,
    albeit solely for the purposes of generating MC draws
    to estimate the null distribution of the test statistic.

    A1,A2 : n-by-n adj mxs
    Xtrue : n-by-d true latent positions that gave rise to A1
    lvl : level at which to do hyp testing

    Returns
    res : 0 or 1, 0 for accept H0:X=Y, 1 for reject.
    '''

    if A1.shape != A2.shape:
        raise ValueError('Input networks should be of the same shape')
    if Xtrue.shape[0] != A1.shape[0]:
        raise ValueError('Latent positions Xtrue mismatch with network size')
    if not isinstance( lvl, float ):
        raise TypeError('Test level lvl should be a float')
    if lvl < 0 or lvl > 1:
        raise ValueError('Test level lvl should be between 0 and 1.')

    (n,d) = Xtrue.shape

    # The function for compute the test statistic Tproc (see Sec 3.2)
    def compute_Tomni( AA1, AA2, dd ):
        nn = AA1.shape[0]
        A = np.zeros( (n,n,2) )
        A[:,:,0] = AA1; A[:,:,1] = AA2
        Zhat = omni_embed( A, dd )
        Xhat1 = Zhat[:n,:]
        Xhat2 = Zhat[n:,:]
        return spla.norm( Xhat1-Xhat2, ord='fro' )**2

    # Compute the test statistic on the observed data
    Tomni = compute_Tomni( A1, A2, d )
    # Now, draw 500 from its null (under true LPs X)
    NMC = 501 # As specified in the paper, plus 1 to avoid comparison with 0.05
    Tomni_short = lambda AA1,AA2 : compute_Tomni( AA1, AA2, d )
    return resamp.do_resample_test( Tomni, Xtrue, Tomni_short, NMC, lvl )

def omni_mean( Zhat, m ):
    '''
    Zhat : mn-by-d array of (presumed estimated) latent positions

    Output
    Zbar : n-by-d array, mean of the omni embeddings.
    '''

    m = int(m)
    if m < 1:
        raise ValueError('m should be positive.')
    if len( Zhat.shape ) != 2:
        raise ValueError('Zhat should be a matrix')
    (mn,d) = Zhat.shape
    n = mn//m
    assert( mn == n*m )
    Zbar = np.zeros( shape=(n,d) ) # store the mean here.
    assert( Zbar.shape==(n,d) )

    # Now average the rest of the blocks of Zhat.
    for k in range(m):
        rowidxs = list(range( (k*n),(n*(k+1)) ))
        Zbar = k*Zbar/(k+1) + Zhat[rowidxs,:]/(k+1)

    return Zbar
