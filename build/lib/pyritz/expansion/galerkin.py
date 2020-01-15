import numpy as np
from pyritz.instanton import Instanton

class GalerkinInstanton(Instanton):
    def __init__(self, x1, x2, Lagrangian, Nq=32):
        self.x1  = x1       # start point at t = -1
        self.x2  = x2       # end point at t = +1
        self.Nq  = Nq       # number of quadrature points
        self.lagrangian = Lagrangian

        self.dim  = np.size(x1)  # dimension of the problem


    def position(self, a, t):
        """
        return positions in terms of a (vector of Chebyshev coefficients)
        """
        dim=self.dim;  Nt=np.size(t)
        self.x    = np.zeros((self.dim*Nt))
        Nm = np.int(np.size(a)/dim)  # number of modes

        if dim==1:
            f = np.polynomial.chebyshev.chebval(t, a)
            self.x = 0.5*(1-t)*self.x1 + 0.5*(1+t)*self.x2 + (1-t*t)*f
        else:
            for i in range(dim):
                f = np.polynomial.chebyshev.chebval(t, a[i*Nm:(i+1)*Nm])
                self.x[i*Nt:(i+1)*Nt] = 0.5*(1-t)*self.x1[i] + 0.5*(1+t)*self.x2[i] + (1-t*t)*f
        return self.x


    def velocity(self, a, t):
        '''
        return positions in terms of a (vector of Chebyshev coefficients)
        '''
        dim=self.dim;  Nt=np.size(t)
        self.v    = np.zeros((self.dim*Nt))
        Nm = np.int(np.size(a)/dim)  # number of modes

        if self.dim==1:
            b = self.chebDerCoeff(a)
            f = np.polynomial.chebyshev.chebval(t, a)
            g = np.polynomial.chebyshev.chebval(t, b)
            self.v = -0.5*self.x1      + 0.5*self.x2       -2*t*f + (1-t*t)*g
        else:
            for i in range(self.dim):
                b = self.chebDerCoeff(a[i*Nm:(i+1)*Nm])
                f = np.polynomial.chebyshev.chebval(t, a[i*Nm:(i+1)*Nm])
                g = np.polynomial.chebyshev.chebval(t, b)
                self.v[i*Nt:(i+1)*Nt] = -0.5*self.x1[i] + 0.5*self.x2[i] - 2*t*f + (1-t*t)*g
        return self.v


    def action(self, a):
        '''
        return Action of a given Lagrangian
        '''
        t, w = np.polynomial.legendre.leggauss(self.Nq)
        x = self.position(a, t)
        v = self.velocity(a, t)
        lg = self.lagrangian(x, v, self.Nq)
        action = sum(w*lg)
        return action


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
