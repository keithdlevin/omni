
import numpy as np
import scipy.linalg as spla

import embeddings
import resample as resamp

def check_proc_inputs( X, Z ):
    if len(X.shape)!=2:
        raise ValueError('X should be a matrix')
    if len(Z.shape)!=2:
        raise ValueError('Z should be a matrix')
    if X.shape != Z.shape:
        raise ValueError('X aand Z should have same shape')

def procrustes_align( X, Z ):
    '''
    Compute the Procrustes alignment of X to Z.
    Return X@R, where X@R is optimally aligned to Z
    '''
    check_proc_inputs( X, Z )

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html#scipy.linalg.orthogonal_procrustes
    (R,_) = spla.orthogonal_procrustes( X, Z )
    return X@R

def procbar( X, Z ):
    '''
    Procrustes-align X and Z and return the mean of the aligned matrices.
    X,Z : n-by-d matrices

    Returns
    Y : n-by-d matrix, Y = (X@R+Z)/2, where R optimally aligned X to Z
    '''
    check_proc_inputs( X, Z )

    XR = procrustes_align(X,Z)
    return (XR + Z)/2

def procbar_many_to_one( Xarray ):
    '''
    Procrustes-align, one layer at a time, multiple sets of latent posns
        to the first embedding.
    Xarray : n-by-d-by-m array. Xarray[:,:,k] is k-th array of latent posns

    Returns
    Xbar : n-by-d matrix, mean of the layers of Xarray, after alignment.
    '''

    (n,d,m) = Xarray.shape
    Xaligned = np.zeros( Xarray.shape )
    # Align the embeddings one at a time.
    # The first one doesn't get aligned to anything.
    for k in range(m):
        Xaligned[:,:,k] = procrustes_align( Xarray[:,:,k], Xarray[:,:,0] )

    # Now, average them.
    Xbar = np.zeros( shape=(n,d) )
    for k in range(m):
        Xbar = k*Xbar/(k+1) + Xaligned[:,:,k]/(k+1)

    return Xbar


def proc_mse( Xhat, X ):
    '''
    Return MSE of Xhat about X after Proc. alignment

    Xhat, X: n-by-d matrices

    Returns
    mse : float, \|\Xhat R - X\|_F^2/n
    '''

    if len(Xhat.shape) != 2:
        raise ValueError('Xhat must be a matrix.')
    if len(X.shape) != 2:
        raise ValueError('X must be a matrix.')

    if Xhat.shape != X.shape:
        raise ValueError('Xhat and X must be same dimension.')

    n = Xhat.shape[0]
    XhatR = procrustes_align( Xhat, X )
    return spla.norm( XhatR-X, ord='fro' )/np.sqrt(n)

def proc_tti( Xhat, X ):
    '''
    Return 2-to-infty error of Xhat about X after Proc. alignment

    Xhat, X: n-by-d matrices

    Returns
    tti : float, \|\Xhat R - X\|_{2,\infty}, where R is the proc. alignment mx
    '''

    if len(Xhat.shape) != 2:
        raise ValueError('Xhat must be a matrix.')
    if len(X.shape) != 2:
        raise ValueError('X must be a matrix.')

    if Xhat.shape != X.shape:
        raise ValueError('Xhat and X must be same dimension.')

    n = Xhat.shape[0]
    XhatR = procrustes_align( Xhat, X )
    # compute norm of each row
    rownorms = spla.norm( XhatR-X, axis=1 ) 
    return float( np.max( rownorms ) )

def proc_hyptest( A1, A2, Xtrue, lvl=0.05 ):
    '''
    Run the Procrustes-based hypothesis test as discussed in Sec 3.2
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
    def compute_Tproc( AA1, AA2, dd ):
        Xhat1 = embeddings.ase( AA1, dd )
        Xhat2 = embeddings.ase( AA2, dd )
        Xhat1R = procrustes_align( Xhat1, Xhat2 )
        return spla.norm( Xhat1R-Xhat2, ord='fro' )**2

    # Compute the test statistic by aligning one to the other.
    Tobsd = compute_Tproc( A1, A2, d )
    NMC = 501 # 501 instead of 500 to avoid rounding issue
    Tproc_short = lambda AA1,AA2 : compute_Tproc( AA1, AA2, d )
    return resamp.do_resample_test( Tobsd, Xtrue, Tproc_short, NMC, lvl )

