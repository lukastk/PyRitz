import numpy as np

def approxChebyshev(N, f):
    """N: number of Chebyshev polynomials; f: function approximated
    Ans: f(t) = -0.5*c[0] + np.polynomial.chebyshev.chebval(t, c)

    Example
    ------
    c = np.polynomial.chebyshev.approxChebyshev(4, np.exp);
    xRange=np.linspace(-1, 1,32)
    yy = -0.5*c[0]+np.polynomial.chebyshev.chebval(xRange, c)
    plt.plot(xRange, np.exp(xRange), 'r-')
    plt.plot(xRange, yy, 'b--')
    """

    c=np.zeros(N);  fac=2.0/N
    for j in range(N):
        dd=0
        for k in range(N):
            theta=np.pi*(k+0.5)/(N)
            dd += f(np.cos(theta))*np.cos(j*theta)
        c[j]=fac*dd
    return c


def chebDerCoeff(self, a):
    '''
    return coefficients of derivatives of a Chebyshev expansion
    '''
    nn=np.size(a);  b=np.zeros((nn+1));   b[nn]=0;   b[nn-1]=0

    for i in range(nn):
        dd=nn-2-i
        b[dd]= 2*(dd+1)*a[dd+1] + b[dd+2]
    b[0] = b[0]/2
    return b[0:nn-1]


def chebProd(self, a):
    """
    returns coefficients of (1-t*t)(a_n *T_n)
    """
    Nm=np.size(a);  a1=np.zeros(Nm+2)

    if Nm==1:
        a1[0]    = 0.5*a[0]
        a1[2]    = - 0.5*a[0]
    elif Nm==2:
        a1[0]    = 0.5*a[0]
        a1[1]    = 0.5*a[1]    - 0.25*a[1]
        a1[2]    =             - 0.5*a[0]
        a1[3]    =             - 0.25*a[1]
    elif Nm==3:
        a1[0]    = 0.5*a[0]    - 0.25*a[2]
        a1[1]    = 0.5*a[1]    - 0.25*a[1]
        a1[2]    = 0.5*a[2]    - 0.5*a[0]
        a1[3]    =             - 0.25*a[1]
        a1[4]    =             - 0.25*a[2]
    elif Nm==4:
        a1[0]    = 0.5*a[0]    - 0.25*a[2]
        a1[1]    = 0.5*a[1]    - 0.25*a[3] - 0.25*a[1]
        a1[2]    = 0.5*a[2]    - 0.5*a[0]
        a1[3]    = 0.5*a[3]    - 0.25*a[1]
        a1[4]    =             - 0.25*a[2]
        a1[5]    =             - 0.25*a[3]

    else:
        for i in range(Nm+2):
            if i==0:
                a1[0]    = 0.5*a[0]    - 0.25*a[2]
            elif i==1:
                a1[1]    = 0.5*a[1]    - 0.25*a[3] - 0.25*a[1]
            elif i==2:
                a1[2]    = 0.5*a[2]    - 0.25*a[4] - 0.5*a[0]
            elif i==Nm-2:
                a1[Nm-2] = 0.5*a[Nm-2] - 0.25*a[Nm-4]
            elif i==Nm-1:
                a1[Nm-1] = 0.5*a[Nm-1] - 0.25*a[Nm-3]
            elif i==Nm:
                a1[Nm]   =             - 0.25*a[Nm-2]
            elif i==Nm+1:
                a1[Nm+1] =             - 0.25*a[Nm-1]
            else:
                a1[i]    = 0.5*a[i]    - 0.25*(a[i+2]+a[i-2])
    return a1
