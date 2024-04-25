"""
optim.py

Contains functions used for optimiziation with Newton's method
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Callable


def optimize(fun: Callable, x0: ArrayLike, max_iterations: int = 100,
        tol: float = 1e-8) -> ArrayLike:
    """ Search for a local optimum of the given function, starting at x0  """

    iter_count = 0
    x_prev = jnp.zeros_like(x0)

    def newton_step(args):
        # do one step in newtons method

        x, _, iter_count = args

        # calculate the first derivative (jacobian) and second derivative (hessian) of the function
        # at x0
        jac = jax.jacobian(fun)(x)
        hess = jax.hessian(fun)(x)
    
        # calculate the step size dx to take (dx = -H⁻¹ / J)
        dx = -jnp.linalg.solve(hess, jac[:, None])[:, 0, 0]
        iter_count += 1

        return x + dx, x, iter_count

    def stop_condition(args):
        # check if the iteration should be stopped
        x, x_prev, iter_count = args

        return jax.lax.bitwise_and(jnp.linalg.norm(x - x_prev) > tol, iter_count < max_iterations)
    
    x_opt, _, _ = jax.lax.while_loop(stop_condition, newton_step, (x0, x_prev, iter_count))
    return x_opt


def square(x: ArrayLike) -> ArrayLike:
    return jnp.square(x)


def main():

    # optimize f(x) = x², start at 8
    optimum = optimize(square, jnp.array([8.]))


if __name__ == '__main__':
    main()
