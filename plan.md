### Design principles

- Should be agnostic with respect to the quadrature nodes and Ritz functional family.
- Collocation method should be agnostic with respect to the collocation nodes used.

### Structure

- Functional Family.
  - Galerkin
  - Chebsyhev
    - Can specify nodes.
- Action. Can compute the action of a given path.
  - Requires functional family.
  - Has methods for various types of minimisation
    - Standard
    - Iterative.
      - Can specify N sequence as a list.
  - Should support reducing the dimension for linear constraints.
  - Gradients
