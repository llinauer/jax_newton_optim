"""
vis.py

Visualize the optimization procedure in 1d & 2d
"""

from typing import Callable
from jax.typing import ArrayLike

def visualize_1d(fun: Callable, x_range: ArrayLike, x0: ArrayLike) -> None:
    """ Visualize the optimization procedure of a function with one variable """

    

