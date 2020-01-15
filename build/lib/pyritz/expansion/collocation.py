import numpy as np
from pyritz.instanton import Instanton

class CollocationInstanton(Instanton):
    def __init__(self, x1, x2, Lagrangian, N):
        self.x1  = x1       # start point at t = -1
        self.x2  = x2       # end point at t = +1
        self.N  = N       # number of quadrature points
        self.lagrangian = Lagrangian
        self.dim  = np.size(x1)  # dimension of the problem

        CollocationInstanton.initialize(self.N)

    def position(self, m, t):
        pass

    def velocity(self, m):
        D = CollocationInstanton.get_differentiation_matrix(self.N)
        vs = []

        for i in range(self.dim):
            vs.append(np.dot(D, m[(self.N)*i:(self.N)*(i+1)]))
        v = np.concatenate(vs)
        return v

    def action(self, m):
        w = CollocationInstanton.get_quadrature_weights(self.N)
        D = CollocationInstanton.get_differentiation_matrix(self.N)

        vs = []
        for i in range(self.dim):
            vs.append(np.dot(D, m[(self.N)*i:(self.N)*(i+1)]))

        v = np.concatenate(vs)
        lg = self.lagrangian(m, v, self.N)
        action = np.sum(w*lg)
        return action

    def action_extra(self, m, Nq):
        t = CollocationInstanton.get_chebyshev_grid(Nq)
        w = CollocationInstanton.get_quadrature_weights(Nq)
        D = CollocationInstanton.get_differentiation_matrix(Nq)

        p = self.get_interpolator(m)
        xs = p.compute_position(t)
        vs = []

        if self.dim>1:
            for i in range(self.dim):
                v = np.dot(D, xs[i])
                vs.append(v)
        else:
            vs = np.dot(D, xs)

        if self.dim>1:
            x = np.concatenate(xs)
            v = np.concatenate(vs)
        else:
            x = xs
            v = vs

        lg = self.lagrangian(x, v, Nq)
        action = np.sum(w*lg)
        return action

    def get_interpolator(self, fs):
        return LagrangeInterpolator(fs, self.dim)

    def get_straight_line_path(x1, x2, N):
        x1 = np.array(x1)
        x2 = np.array(x2)

        dx = (x2 - x1)
        cg = (1 + CollocationInstanton.get_chebyshev_grid(N))/2

        d = np.size(x1)
        s = np.zeros((N, d))

        for i in range(N):
            x = x1 + dx*cg[i]
            s[i,:] = x

        #s = s[1:-2] # Remove ends

        xs = np.zeros(N * d)

        for i in range(N):
            for j in range(d):
                xs[i+j*N] = s[i,j]

        return xs

    def get_quadrature_weights(N):
        if not N in CollocationInstanton.weights:
            CollocationInstanton.weights[N] = CollocationInstanton.generate_quadrature_weights(N)

        return CollocationInstanton.weights[N]

    def get_differentiation_matrix(N):
        if not N in CollocationInstanton.differentiaton_matrices:
            CollocationInstanton.differentiaton_matrices[N] = CollocationInstanton.generate_differentiation_matrix(N)

        return CollocationInstanton.differentiaton_matrices[N]

    def get_chebyshev_grid(N):
        x = -np.cos(np.pi * np.arange(N) / (N-1)) # Minus sign here is so that the the points are in increasing order
        return x

    def generate_quadrature_weights(N):
        # From Spectral Methods in Matlab (Trefethen)
        w = np.zeros(N)
        theta = np.pi*np.arange(0, N) / (N-1)
        x = np.cos(theta)
        ii = np.arange(1, N-1)
        v = np.ones(N-2)

        if (N-1) % 2 == 0:
            w[0] = 1 / ((N-1)**2 - 1)
            w[N-1] = w[0]
            for k in range(1, int((N-1)/2)):
                v = v - 2*np.cos(2 * k * theta[ii]) / (4 * k*k - 1)
            v = v - np.cos( (N-1) * theta[ii]) / ( (N-1)**2 - 1 )
        else:
            w[0] = 1 / ((N-1)**2)
            w[N-1] = w[0]

            for k in range(1, int((N-2)/2) + 1):
                v = v - 2*np.cos(2 * k * theta[ii]) / (4 * k*k - 1)

        w[ii] = 2 * v / (N-1)

        return w

    def generate_differentiation_matrix(N):
        x = CollocationInstanton.get_chebyshev_grid(N)
        c = np.ones(N)
        c[0] = 2
        c[N-1] = 2
        c = c * np.power(-1, np.arange(N))
        X = np.tile(x, (N, 1))
        dX = X.T - X
        D = np.outer(c, 1/c) / (dX + np.eye(N))
        D = D - np.diag(np.sum(D.T, axis=0))

        return D

    def initialize(N):
        if not N in CollocationInstanton.weights:
            CollocationInstanton.weights[N] = CollocationInstanton.generate_quadrature_weights(N)
        if not N in CollocationInstanton.differentiaton_matrices:
            CollocationInstanton.differentiaton_matrices[N] = CollocationInstanton.generate_differentiation_matrix(N)

CollocationInstanton.weights = {}
CollocationInstanton.differentiaton_matrices = {}

class LagrangeInterpolator():
    def __init__(self, f, d):
        self.d = d
        self.N = int(np.size(f)/self.d)

        self.fs = []
        for i in range(d):
            self.fs.append(f[i*self.N:(i+1)*self.N])

        self.w = self.__weights(self.N)
        self.chebPoints = CollocationInstanton.get_chebyshev_grid(self.N)

        D = CollocationInstanton.get_differentiation_matrix(self.N)
        vs = []
        for i in range(self.d):
            vs.append(np.dot(D, f[(self.N)*i:(self.N)*(i+1)]))

        self.dfs = vs

    def __weights(self, N):
        j = np.arange(0, N)
        w = np.power(-1, j)
        o = np.ones(N)
        o[0] = 0.5
        o[N-1] = 0.5
        w = np.flipud(w*o)

        return w

    def __barycentric(self, ts, f):
        p = np.zeros(np.size(ts))

        if np.size(ts) == 1:
            ts = [ts]

        s = np.zeros( ( np.size(ts), self.N ) )

        for i in range(np.size(ts)):
            t = ts[i]
            if t in self.chebPoints: # If t is a grid point, then just return the corresponding value in f.
                s[i,np.where(self.chebPoints == t)[0][0]] = 1
            else:
                s[i,:] = self.w / (t - self.chebPoints)

        ss = np.sum(s, axis=1)
        p = np.dot(s, f) / ss

        if np.size(ts) == 1:
            p = p[0]

        return p

    def compute_position(self, t):
        xs = []

        for i in range(self.d):
            xs.append(self.__barycentric(t, self.fs[i]))

        if self.d == 1:
            xs = xs[0]

        return xs

    def compute_velocity(self, t):
        vs = []

        for i in range(self.d):
            vs.append(self.__barycentric(t, self.dfs[i]))

        if self.d == 1:
            vs = vs[0]

        return vs
