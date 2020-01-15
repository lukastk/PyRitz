import pyritz

def chebyshev2(n, derivatives):
    collocation_ts = pyritz.interpolation.utils.chebyshev_nodes(n)
    collocation_w = pyritz.interpolation.utils.barycentric_weights(n)

    if len(derivatives) != 0:
        diff_matrices = pyritz.interpolation.utils.chebyshev_differentiation_matrices(n, derivatives)
        return collocation_ts, collocation_w, diff_matrices
    else:
        return collocation_ts, collocation_w
