"""
optim.py

Contains functions used for optimiziation with Newton's method
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpy.typing import ArrayLike as NpArrayLike
from typing import Callable, Tuple
from vis import plot_1d_interactive


def optimize(fun: Callable, x0: NpArrayLike, max_iterations: int = 100,
             tol: float = 1e-6) -> Tuple[NpArrayLike]:
    """ Search for a local optimum of the given function, starting at x0  """

    x = x0

    x_vals = np.array([])
    grads = np.array([])

    for _ in range(max_iterations):

        # calculate the first derivative (jacobian) and second derivative (hessian) of the function
        # at x0
        jac = jax.jacobian(fun)(x)
        hess = jax.hessian(fun)(x)
    
        # calculate the step size dx to take (dx = -H⁻¹ / J)
        dx = -jnp.linalg.solve(hess, jac[:, None])[:, 0, 0]
        x_new = x + dx

        x_vals = np.append(x_vals, x)
        grads = np.append(grads, jac)

        if abs(x_new - x) < tol:
            break

        x = x_new

    return np.asarray(x), x_vals, grads


def poly(x: ArrayLike) -> ArrayLike:
    return x**3 + 2*x**2 - 3*x + 2


def main():

    # optimize f(x) = x², start at 8
    x_opt, x_vals, grads = optimize(poly, np.array([3.]))
    fig = plot_1d_interactive(poly, np.arange(-3.5, 3.5, 0.05), x_vals, grads)
    fig.show()


if __name__ == '__main__':
    main()
