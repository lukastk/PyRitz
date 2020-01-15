import pyritz
import numpy as np

class Action():
    def __init__(self, lagrangian, n, nq, x1=None, x2=None, collocation_scheme=None, quadrature_scheme=None, derivatives=1, lagrangian_args=None):
        self.lagrangian = lagrangian
        self.lagrangian_args = lagrangian_args # Stores arguments that should be passed to the larangian

        self.n = n
        self.nq = nq
        self.adj_n = n+1 # The number of actual degrees of freedom
        self.adj_nq = nq+1 # The number of actual nodes calculated in the quadrature

        if np.size(x1) == 1:
            x1 = np.array([x1])
            x2 = np.array([x2])

        self.x1 = np.copy(x1)
        self.x2 = np.copy(x2)
        self.dim = x1.size

        if not self.x1 is None:
            self.adj_n -=1
        if not self.x2 is None:
            self.adj_n -=1

        # The user can specify which derivatives of the path to compute
        if type(derivatives) == int:
            derivatives = list(range(1, derivatives+1))
        self.derivatives = derivatives

        # By default, interpolation over the Chebyshev nodes of the second kind is used
        if collocation_scheme is None:
            collocation_scheme = pyritz.interpolation.collocation.chebyshev2

        # By default, Clenshaw-Curtis quadrature is used
        if quadrature_scheme is None:
            quadrature_scheme = pyritz.interpolation.quadrature.clenshaw_curtis

        self.collocation_ts, self.collocation_w, self.diff_matrices = collocation_scheme(n, derivatives)
        self.quadrature_ts, self.quadrature_w = quadrature_scheme(nq)

        self._lagrangian_arr = np.zeros(self.adj_nq)
        self._dxlagrangian_arr = np.zeros( (self.adj_nq, self.dim) )
        self._dxlagrangian_arr_T = self._dxlagrangian_arr.T
        self._dvlagrangian_arr = np.zeros( (self.adj_nq, self.dim) )
        self._dvlagrangian_arr_T = self._dvlagrangian_arr.T
        self._lxw = np.zeros( (self.adj_nq, self.dim) )
        self._lvw = np.zeros( (self.adj_nq, self.dim) )
        self.__quadrature_weights_stack = np.repeat(  [self.quadrature_w], self.dim, axis=0 ).T # This is an (Nq, dim) array, used for the Hadamard product in the gradient calculation

        self.__initialise_interpolation()
        self.__initialise_interpolation_gradient()

    def compute(self, alpha, grad=None):
        self.__interpolate(alpha)

        if grad is None or grad.size==0:
            self.lagrangian(self._lagrangian_arr, None, None, self.path_T, self.quadrature_ts, self.lagrangian_args)
        else:
            self.lagrangian(self._lagrangian_arr, self._dxlagrangian_arr_T, self._dvlagrangian_arr_T, self.path_T, self.quadrature_ts, self.lagrangian_args)
            np.multiply(self._dxlagrangian_arr, self.__quadrature_weights_stack, out=self._lxw)
            np.multiply(self._dvlagrangian_arr, self.__quadrature_weights_stack, out=self._lvw)
            grad[:] = np.trace(np.tensordot(self._lxw, self.__ffxgrad, axes=([1,2])) + np.tensordot(self._lvw, self.__ffvgrad, axes=([1,2])), axis1=0, axis2=2) # TODO: Vectorize

        np.multiply(self.quadrature_w, self._lagrangian_arr, out=self._lagrangian_arr)
        S = np.sum(self._lagrangian_arr)

        return S

    def compute_lagrangian(self, alpha):
        self.__interpolate(alpha)
        self.lagrangian(self._lagrangian_arr, None, None, self.path_T, self.quadrature_ts, self.lagrangian_args)
        return self._lagrangian_arr

    def compute_gradient(self, alpha):
        grad = np.zeros(np.size(alpha))
        self.__interpolate(alpha)
        self.lagrangian(self._lagrangian_arr, self._dxlagrangian_arr_T, self._dvlagrangian_arr_T, self.path_T, self.quadrature_ts, self.lagrangian_args)
        np.multiply(self._dxlagrangian_arr, self.__quadrature_weights_stack, out=self._lxw)
        np.multiply(self._dvlagrangian_arr, self.__quadrature_weights_stack, out=self._lvw)
        grad[:] = np.trace(np.tensordot(self._lxw, self.__ffxgrad, axes=([1,2])) + np.tensordot(self._lvw, self.__ffvgrad, axes=([1,2])), axis1=0, axis2=2) # TODO: Vectorize
        return grad

    def get_alpha_with_endpoints(self, alpha):
        n_adj = int(len(alpha) / self.dim)
        a = np.array(alpha).reshape( ( self.dim, n_adj ) ).T

        if not self.x1 is None:
            a = np.concatenate([
                [self.x1],
                a
            ])
        if not self.x2 is None:
            a = np.concatenate([
                a,
                [self.x2]
            ])

        return a.T.flatten()


    def __interpolate(self, alpha):
        alpha_reshaped = alpha.reshape( (self.dim, self.adj_n) ).T

        vals = []

        for p, c, B in zip(self.path, self.cs, self.Bs):
            np.dot(B, alpha_reshaped, out=p)
            np.add(c, p, out=p)

    # Initialisation helper functions

    def __initialise_interpolation(self):
        self.Bs = Action.__get_barycentric_upsampling_matrix(self.collocation_ts, self.collocation_w, self.quadrature_ts, self.diff_matrices)
        self.cs = [0 for i in range(len(self.Bs))]
        self.path = [np.zeros( (self.adj_nq, self.dim) ) for i in range(len(self.Bs))]
        self.path_T = [p.T for p in self.path] # Transposed version of the path, which is passed to the lagrangian

        if not self.x1 is None:
            for i in range(len(self.Bs)):
                self.cs[i] += np.array([self.Bs[i][:, 0] * self.x1[j] for j in range(self.dim)]).T
                self.Bs[i] = self.Bs[i][:, 1:]

        if not self.x2 is None:
            for i in range(len(self.Bs)):
                self.cs[i] += np.array([self.Bs[i][:, -1] * self.x2[j] for j in range(self.dim)]).T
                self.Bs[i] = self.Bs[i][:, :-1]

    def __initialise_interpolation_gradient(self):
        Ai = np.zeros( (self.nq+1, self.n+1) )

        for i in range(self.nq+1):
            for j in range(self.n+1):
                if self.quadrature_ts[i] - self.collocation_ts[j] != 0:
                    Ai[i,j] = self.collocation_w[j] / ( self.quadrature_ts[i] - self.collocation_ts[j] )
                    s = 0
                    for k in range(self.n+1):
                        if self.quadrature_ts[i] - self.collocation_ts[k] == 0:
                            s = np.inf
                            break
                        else:
                            s += self.collocation_w[k] / (self.quadrature_ts[i] - self.collocation_ts[k] )
                    a = Ai[i,j]
                    Ai[i, j] = Ai[i, j] / s
                else:
                    Ai[i,j] = 1

        Bi = Ai.dot(self.diff_matrices[0])
        Ai = Ai.T
        Bi = Bi.T

        A = np.zeros( (self.dim*(self.n+1), self.nq+1, self.dim) )
        B = np.zeros( (self.dim*(self.n+1), self.nq+1, self.dim) )

        for i in range(self.dim):
            A[i*(self.n+1):(i+1)*(self.n+1), :, i] = Ai
            B[i*(self.n+1):(i+1)*(self.n+1), :, i] = Bi

        # If end-points are fixed, then we should exclude parts of A and B
        mask = np.full(self.dim*(self.n+1), True)
        if not self.x1 is None:
            for i in range(self.dim):
                mask[i*(self.n+1)] = False
        if not self.x2 is None:
            for i in range(self.dim):
                mask[((self.n+1)-1) + i*(self.n+1)] = False

        self.__ffxgrad = A[mask, :, :]
        self.__ffvgrad = B[mask, :, :]

    def __get_barycentric_upsampling_matrix(cts, cw, qts, diff_matrices):
        """
        Generates the Baryentric upsampling matrix, and the corresponding
        Barycentric Chebyshev differentiation upsampling matrices.
        """

        # Generate barycentric matrix
        B =  np.zeros( (len(qts), len(cts)) )
        for i in range(len(qts)):
            delta_t = qts[i] - cts
            whr = np.where(delta_t == 0)[0]

            if len(whr) == 0:
                pass
                wd = cw/delta_t
                B[i, :] = wd / np.sum(wd)
            else:
                B[i, whr] = 1

        Bs = [B]

        # Derivatives
        for D in diff_matrices:
            Bs.append(B.dot(D))

        return Bs
