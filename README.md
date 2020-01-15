PyRitz
=======

PyRitz is a library for

A python package for direct variational minimisation, specifically suited for finding Freidlin-Wentzell instantons.

You can try out PyRitz

## TODO: Change Path into something else

## TODO: Remove get_action

## TODO: Change dxs to vs

## TODO: Fix catenary

## TODO: Change variables (u,v) on examples

## Usage

```python
import pyritz, nlopt

# Define the Lagrangian
def lagrangian(ls, dxls, dvls, path, ts, args):
    xs, vs = path
    ls[:] = 0.5*np.sqrt((1+4*vs*vs)/(-2*10*xs))
    return lg

# Define the end-point conditions
x1 = np.cosh(-1); x2 = np.cosh(1)

# Set the interpolation and the quadrature order
n = 8
nq = n*10

# Define the initial path for the minimiser
alpha0 = pyritz.interpolation.utils.linear_path(x1, x2, n)

# Setup the path-interpolation and action quadrature of the system using PyRitz
system = pyritz.interpolation.System(lagrangian, n, nq, x1, x2)

# Minimize the action using NLopt
opt = nlopt.opt(nlopt.LD_SLSQP, np.size(alpha0))
opt.set_min_objective(system.action)
opt.set_xtol_rel(1e-16)
alpha = opt.optimize(alpha0)

print("S[alpha0] = %s" % path.action(alpha0))
print("S[alpha]  = %s" % path.action(alpha))
```

Plot the result:

```
```

## Dependencies

- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [NLopt](https://nlopt.readthedocs.io/en/latest/)

To run the examples you need:

- [Jupyter Notebook](https://jupyter.org/) (Easiest way to  install this is via [Anaconda](https://www.anaconda.com/distribution/))
- [Matplotlib](https://matplotlib.org/)

## Installation

### Running the PyRitz examples

It is possible to try out PyRitz *without* installing it to your system (this was only tested on Linux). The script *setup_nlopt_locally.sh* installs NLopt into the repository, so that the examples can be run locally.

```
git clone https://github.com/lukastk/PyRitz.git
cd PyRitz
sh setup_nlopt_locally.sh
cd examples
jupyter notebook
```

### Installing from the repository



```
git clone https://github.com/lukastk/PyRitz.git
cd PyRitz
python setup.py install
```

[dependencies](#dependencies)

### Installing dependencies


## Publications

*Ritz method for transition paths and quasipotentials of rare diffusive events*. L. T. Kikuchi, R. Singh, M. E. Cates, R. Adhikari (To be published)

## Citing PyRitz

If you use PyRitz for academic work, we would request you to cite our papers.

## License

PyRitz is published under the [MIT License](https://opensource.org/licenses/MIT).

This repository includes code from [NLopt](https://nlopt.readthedocs.io/) which is under the [GNU Lesser General Public License](https://en.wikipedia.org/wiki/GNU_Lesser_General_Public_License), developed by [Steven G. Johnson](https://github.com/stevengj).

## Authors

[Lukas Kikuchi](https://github.com/lukastk), [Rajesh Singh](https://github.com/rajeshrinet), Mike Cates, [Ronojoy Adhikari](https://github.com/ronojoy)
