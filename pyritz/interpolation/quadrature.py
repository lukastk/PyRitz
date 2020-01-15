import numpy as np

def clenshaw_curtis(nq):
    ## Generate quadrature nodes
    ts = -np.cos(np.pi * np.arange(nq+1) / (nq)) # Minus sign here is so that the the points are in increasing order

    ## Generate quadrature weights
    # From Spectral Methods in Matlab (Trefethen)
    w = np.zeros(nq+1)
    theta = np.pi*np.arange(0, nq+1) / (nq)
    x = np.cos(theta)
    ii = np.arange(1, nq)
    v = np.ones(nq-1)

    if nq % 2 == 0:
        w[0] = 1 / (nq**2 - 1)
        w[nq-1] = w[0]
        for k in range(1, int(nq/2)):
            v = v - 2*np.cos(2 * k * theta[ii]) / (4 * k*k - 1)
        v = v - np.cos( nq * theta[ii]) / ( nq**2 - 1 )
    else:
        w[0] = 1 / (nq**2)
        w[nq-1] = w[0]

        for k in range(1, int((nq-1)/2) + 1):
            v = v - 2*np.cos(2 * k * theta[ii]) / (4 * k*k - 1)

    w[ii] = 2 * v / nq

    return (ts, w)

def gauss(nq):
    ts, w = np.polynomial.legendre.leggauss(nq+1)
    return (ts, w)
