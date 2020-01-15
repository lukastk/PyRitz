# Gradient handling in Pyritz

The three functions that Pyritz deal with are the following:

```math
\begin{aligned}
S & = \sum_{i=1}^{N_q} w_i L( x(\tau_i), \dot{x}(\tau_i) ) \\
x(t) & = \Phi_N(t ; \{m_i\}_{i=1}^{N_m}) \\
L & = L(x, \dot{x}, t)
\end{aligned}
```

Thus we can see the action as a function of the parameters $`\{m_i\}_{i=1}^{N_q}`$

```math
S = S\left[\{m_i\}_{i=1}^{N_q}\right]
```

Let

```math
\begin{aligned}
& a_k = x(\tau_k) = \Phi_N(\tau_k ; \{m_i\}_{i=1}^{N_q} ) \\
& b_k = \dot{x}(\tau_k) = \dot{\Phi}_N(\tau_k ; \{m_i\}_{i=1}^{N_q} )
\end{aligned}
```

The derivative of $`S`$ is then given as

```math
\begin{aligned}
\frac{\partial S}{\partial m_i} & = \sum_{k=1}^{N_q} w_k \left[ \frac{\partial L}{\partial x}\bigg\vert_{x = a_k, \dot{x} = b_k} \frac{\partial \Phi_N}{\partial m_i}\bigg\vert_{t=\tau_k}
+ \frac{\partial L}{\partial \dot{x}}\bigg\vert_{x = a_k, \dot{x} = b_k} \frac{\partial \dot{\Phi}_N}{\partial m_i}\bigg\vert_{t=\tau_k} \right]
\end{aligned}
```

We define

```math
\begin{aligned}

A_{ij} & \coloneqq \frac{\partial \Phi_N}{\partial m_j}\bigg\vert_{t=\tau_i} \\
B_{ij} & \coloneqq \frac{\partial \dot{\Phi}_N}{\partial m_j}\bigg\vert_{t=\tau_i}
\end{aligned}
```
