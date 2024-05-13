"""
optim.py

Contains functions used for optimiziation with Newton's method
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Callable


def optimize(fun: Callable, x0: ArrayLike, max_iterations: int = 100,
             tol: float = 1e-6) -> ArrayLike:
    """ Search for a local optimum of the given function, starting at x0  """

    x = x0

    x_vals = jnp.array([])
    grads = jnp.array([])

    for _ in range(max_iterations):

        # calculate the first derivative (jacobian) and second derivative (hessian) of the function
        # at x0
        jac = jax.jacobian(fun)(x)
        hess = jax.hessian(fun)(x)
    
        # calculate the step size dx to take (dx = -H⁻¹ / J)
        dx = -jnp.linalg.solve(hess, jac[:, None])[:, 0, 0]
        x_new = x + dx

        x_vals = jnp.append(x_vals, x)
        grads = jnp.append(grads, jac)

        if abs(x_new - x) < tol:
            break

        x = x_new

    return x, x_vals, grads


def poly(x: ArrayLike) -> ArrayLike:
    return x**3 + 2*x**2 - 3*x + 2


def main():

    # optimize f(x) = x², start at 8
    x_opt, x_vals, grads = optimize(poly, jnp.array([3.]))
    print(x_opt, x_vals, grads)


if __name__ == '__main__':
    main()
