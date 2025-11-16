
import numpy as np
import scipy as sp
import scipy.linalg as spla

import networks as net

def ase(A,d):
    '''
    Compute the d-dimensional ASE of A.
    A : adjacency matrix
    d : positive integer, desired embedding dimension

    Returns
    Xhat : n-by-d embedding; Xhat[i,:] is d-dim embedding of vx i.
    '''

    net.check_valid_adjmx( A )

    n = A.shape[0]
    (S,U) = spla.eigh(A,eigvals=(n-d,n-1))
    # Scale the eigenvectors by square roots of eigenvalues.
    # This is the ASE, Xhat = U S^{1/2}
    return np.matrix(U)*np.matrix(np.diag(np.sqrt(S)))

def asebar( A,d ):
    '''
    Compute the d-dimensional ASEbar embedding of array A

    A : n-by-n-by-m array, A[:,:,s] is s-th adj matrix
    d : positive int specifying embedding dimension

    Returns
    Xhat : n-by-d matrix; i-th row is embedding of vx i.
    '''

    Abar = np.mean( A, axis=2 )
    return ase( Abar, d )

