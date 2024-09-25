"""
poly_optim.py

User interface for optimizing polynomials
"""

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from jax.typing import ArrayLike

from optim import optimize
from vis import plot_1d_interactive, save_1d_vis


def parse_args() -> argparse.Namespace:
    """ Parse CL arguments:
        --coeff (int, int, ...)  -    Polynomial coefficients
        --x-range (float, float) -    [x_min, x_max]
        --x-start (float)        -    Initial guess for Newton's method
        --save-path (str)        -    Save created files at this path
    """

    parser = argparse.ArgumentParser(description="Find the optimum of a polynomial using Newton's method")
    parser.add_argument("--coeff", nargs="+", type=int, help="Coefficients of the polynomial",
                        required=True)
    parser.add_argument("--x-range", nargs=2, type=float, help="x-Range (x_min, x_max)",
                        default=[0, 1])
    parser.add_argument("--x-start", type=float, help="Initial guess for x",
                        required=True)
    parser.add_argument("--save-path", type=str, help="Save created files here",
                        default="img")

    args = parser.parse_args()
    return args


def create_polynomial(*coeffs) -> Callable:
    """ Wrapper functions that creates a polynomial functions out of the coefficients"""

    def polynomial(x: ArrayLike) -> ArrayLike:
        # calculate the polynomial function at x
        result = 0
        degree = len(coeffs) - 1
        for i, coeff in enumerate(coeffs):
            result += coeff * (x ** (degree - i))
        return result

    return polynomial


def main() -> None:
    """ Main function and User interface """

    # parse CL args
    args = parse_args()

    # define polynomial function
    poly_func = create_polynomial(*args.coeff)

    # check if the initial guess lies within the x-range
    # if not, adapt the range
    x_range = args.x_range
    if args.x_start < x_range[0]:
        x_range = [args.x_start, x_range[1]]
    elif args.x_start > x_range[1]:
        x_range = [x_range[0], args.x_start]

    x_range = np.linspace(x_range[0], x_range[1], 100)

    # optimize polynomial
    _, x_vals, grads = optimize(poly_func, np.array([args.x_start]))
    save_1d_vis(poly_func, x_range, x_vals, grads, Path(args.save_path))
    plot_1d_interactive(poly_func, np.arange(-3.5, 3.5, 0.05), x_vals, grads, Path(args.save_path))


if __name__ == "__main__":
    main()
