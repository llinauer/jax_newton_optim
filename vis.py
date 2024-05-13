"""
vis.py

Visualize the optimization procedure in 1d & 2d
"""

from typing import Callable
from jax.typing import ArrayLike
import numpy as np
import plotly.graph_objects as go


def get_tangent_line_points(contact_x: ArrayLike, contact_y: ArrayLike, slope: ArrayLike,
                            x_vals: ArrayLike) -> ArrayLike:
    """ Get the points of the tangent line defined by y = slope*x + d """

    # calculate the intersection with the y-axis d
    d = contact_y - slope * contact_x
    # calculate the x-value of the intersection with x-axis
    x_intersect = -d / slope

    # create an array of x and y values corresponding to the tangent points
    # check if the intersection with the x_axis is left or right of contact_x and create the
    # x values of the tangent line accordingly
    if x_intersect > contact_x:
        tangent_x_vals = np.linspace(x_vals.min(), x_intersect, 40)
    else:
        tangent_x_vals = np.linspace(x_intersect, x_vals.max(), 40)

    # create y values of tangent line
    tangent_y_vals = slope * tangent_x_vals + d

    return tangent_x_vals, np.asarray(tangent_y_vals), x_intersect.item()


def visualize_1d(fun: Callable, x_vals: ArrayLike, intermediate_vals: ArrayLike,
                 grads: ArrayLike) -> go.Figure:
    """ Visualize the optimization procedure of a function with 1 variable"""

    # create the frames for visualizing newton's method
    # loop over each intermediate value from the optimization
    frames = []
    for x, grad in zip(intermediate_vals, grads):

        frame_data = []
        # for each x-value, calculate the tangent line
        tx, ty, x_intersect = get_tangent_line_points(x, fun(x), grad, x_vals)

        # get the function value at the intersection of the tangent and the x-axis
        y_intersect = fun(x_intersect)

        # create a frame with the tangent and its intersection with the x-axis
        frame_data.append(go.Scatter(x=[x], y=[fun(x)], mode='markers',
                                     marker=dict(color='red', symbol='x')))
        frame_data.append(go.Scatter(x=tx, y=ty, mode='lines', line=dict(color='red', width=1)))
        frame_data.append(go.Scatter(x=[x_intersect], y=[y_intersect], mode='markers',
                                     marker=dict(color='green', symbol='x')))
        frame_data.append(go.Scatter(x=[x_intersect, x_intersect], y=[0, y_intersect], mode='lines',
                                     line=dict(color='green', dash='dash')))
        frames.append(go.Frame(data=frame_data))

    # calculate the x- & y-range
    x_ranges = [x_vals.min(), x_vals.max()]
    y_range = fun(x_vals).max() - fun(x_vals).min()
    y_ranges = [fun(x_vals).min() - y_range*0.2, fun(x_vals).max() + y_range*0.2]

    # create figure and plot the function
    fig = go.Figure(data=[go.Scatter(x=x_vals, y=fun(x_vals), mode='lines',
                                     line=dict(width=2, color='blue')) for i in range(5)],
                    layout=go.Layout(title_text="Optimization with Newton's method",
                                     xaxis=dict(range=x_ranges, autorange=False, zeroline=False),
                                     yaxis=dict(range=y_ranges, autorange=False, zeroline=False),
                                     updatemenus=[dict(type="buttons",
                                                       buttons=[dict(label="Play",
                                                                     method="animate",
                                                                     args=[None, {
                                                                         "frame": {"duration": 2000,
                                                                                   "redraw": False}, }])])]),
                    frames=frames
                    )

    return fig

