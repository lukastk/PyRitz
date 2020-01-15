import pyritz
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.linalg import toeplitz

def resample(alpha, n1, n2, collocation_scheme=None):
    pass

def finite_difference_gradient(action, alpha, d=1e-10):
    grad = np.zeros(np.size(alpha))
    for i in range(grad.size):
        dalpha = np.zeros(np.size(alpha))
        dalpha[i] = d
        grad[i] = ( action.compute(alpha + dalpha) - action.compute(alpha - dalpha) ) / (2*d)
    return grad

def interpolate(alpha, n, ts, collocation_scheme=None):
    # By default, interpolation over the Chebyshev nodes of the second kind is used
    if collocation_scheme is None:
        collocation_scheme = pyritz.interpolation.collocation.chebyshev2

    cts, _ = collocation_scheme(n, [])
    dim = int(len(alpha) / (n+1))
    alpha_reshaped = alpha.reshape( (dim, n+1) )
    xs = np.zeros( (dim, len(ts)) )

    for i in range(dim):
        xs[i,:] = BarycentricInterpolator(cts, alpha_reshaped[i, :])(ts)

    return xs

def linear_path(x1, x2, n, exclude_x1=True, exclude_x2=True, collocation_scheme=None):
    # By default, interpolation over the Chebyshev nodes of the second kind is used
    if collocation_scheme is None:
        collocation_scheme = pyritz.interpolation.collocation.chebyshev2
    ts, _ = collocation_scheme(n, [])

    x1 = np.array(x1)
    x2 = np.array(x2)
    dim = np.size(x1)

    dx = (x2 - x1)

    d = np.size(x1)
    s = np.zeros((n+1, dim))
    fts = (1+ts)/2
    s = np.array([x1 + dx*fts[i] for i in range(n+1)])

    binmap = np.full(n+1, True)
    if exclude_x1:
        binmap[0] = False
    if exclude_x2:
        binmap[-1] = False
    s = s[binmap]

    xs = s.T.reshape( s.size ).T

    return xs

def chebyshev_nodes(n):
    ts = -np.cos(np.pi * np.arange(n+1) / (n)) # Minus sign here is so that the the points are in increasing order
    return ts

def barycentric_weights(n):
    j = np.arange(0, n+1)
    w = np.power(-1, j)
    o = np.ones(n+1)
    o[0] = 0.5
    o[n] = 0.5
    w = np.flipud(w*o)
    return w

def chebyshev_differentiation_matrices(N, derivatives):
    _, D = __chebdif(N+1, np.max(derivatives))

    # Required because we use a different node ordering
    for i in range(np.max(derivatives)):
        D[i, :, :] = ((-1)**(i+1)) * D[i, :, :]

    Ds = []
    for i in range(len(derivatives)):
        Ds.append(D[derivatives[i]-1, :, :])

    return Ds

def __chebdif(N,M):
    '''
    Credit: https://github.com/ronojoy/pyddx
    '''

    if M >= N:
        raise Exception('numer of nodes must be greater than M')

    if M <= 0:
         raise Exception('derivative order must be at least 1')

    DM = np.zeros((M,N,N))

    n1 = int(np.floor(N/2.)); n2 = int(np.ceil(N/2.))     # indices used for flipping trick
    k = np.arange(N)                    # compute theta vector
    th = k*np.pi/(N-1)

    # Compute the Chebyshev points

    #x = np.cos(np.pi*np.linspace(N-1,0,N)/(N-1))                # obvious way
    x = np.sin(np.pi*((N-1)-2*np.linspace(N-1,0,N))/(2*(N-1)))   # W&R way
    x = x[::-1]

    # Assemble the differentiation matrices
    T = np.tile(th/2,(N,1))
    DX = 2*np.sin(T.T+T)*np.sin(T.T-T)               # trigonometric identity
    DX[n1:,:] = -np.flipud(np.fliplr(DX[0:n2,:]))    # flipping trick
    DX[range(N),range(N)]=1.                         # diagonals of D
    DX=DX.T

    C = toeplitz((-1.)**k)           # matrix with entries c(k)/c(j)
    C[0,:]  *= 2
    C[-1,:] *= 2
    C[:,0] *= 0.5
    C[:,-1] *= 0.5

    Z = 1./DX                        # Z contains entries 1/(x(k)-x(j))
    Z[range(N),range(N)] = 0.        # with zeros on the diagonal.

    D = np.eye(N)                    # D contains differentiation matrices.

    for ell in range(M):
        D = (ell+1)*Z*(C*np.tile(np.diag(D),(N,1)).T - D)      # off-diagonals
        D[range(N),range(N)]= -np.sum(D,axis=1)        # negative sum trick
        DM[ell,:,:] = D                                # store current D in DM

    return x,DM
