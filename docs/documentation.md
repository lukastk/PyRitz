PyRitz Documentation
===================

## Tutorial

This tutorial will guide you through the process of using PyRitz to find the
Freidlin-Wentzell instanton of Stochasic Differential Equation (SDE). As an
example, we will consider the Maier-Stein system [1]. The completed example
code can be found [here](https://github.com/lukastk/PyRitz/blob/master/examples/Maier-Stein.ipynb).

The Maier-Stein system is of the following form:

$$
dX = a(X)dt + \sqrt{\epsilon} dW.
$$

where

$$
a(x_1, x_2) =
\begin{pmatrix}
x_1 - x_1^3 - \beta x_1 x_2^2 \\
-(1 + x_1^2) x_2
\end{pmatrix}
$$

Walk-through through all the steps required to make a script that can find a
FW instanton.

Note also how to Using PyRitz to find a Freidlin-Wentzell instanton

## References

[1] R. S. Maier and D. L. Stein, Journal of Statistical Physics 83, 291 (1996)
