"""
vis.py

Visualize the optimization procedure in 1d & 2d
"""

from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from jax.typing import ArrayLike
from PIL import Image


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


def plot_1d_interactive(fun: Callable, x_vals: ArrayLike, intermediate_vals: ArrayLike,
                        grads: ArrayLike, path: Path) -> None:
    """ Visualize the optimization procedure of a function with 1 variable as an interactive plot"""

    # calculate the x- & y-range
    x_ranges = [x_vals.min(), x_vals.max()]
    y_range = fun(x_vals).max() - fun(x_vals).min()
    y_ranges = [fun(x_vals).min() - y_range*0.2, fun(x_vals).max() + y_range*0.2]

    # calculate first derivative of fun
    first_derivative = jnp.array([jax.grad(fun)(x).item() for x in x_vals])

    # create the frames for visualizing newton's method
    # loop over each intermediate value from the optimization
    frames = []
    slider_steps = []
    for i, (x, grad) in enumerate(zip(intermediate_vals, grads)):

        frame_data = []
        # for each x-value, calculate the tangent line
        tx, ty, x_intersect = get_tangent_line_points(x, jax.grad(fun)(x).item(), grad, x_vals)

        # get the function value at the intersection of the tangent and the x-axis
        y_intersect = jax.grad(fun)(x_intersect)

        # create a frame with the tangent and its intersection with the x-axis
        frame_data.append(go.Scatter(x=[x], y=[jax.grad(fun)(x)], mode="markers",
                                     marker=dict(color="red", symbol="x")))
        frame_data.append(go.Scatter(x=x_vals, y=first_derivative, mode="lines",
                                     line=dict(width=2, color="orange", dash="dot")))
        # in the last step, don't plot the tangent line
        # instead, plot a vertical line
        if i < len(grads) - 1:
            frame_data.append(go.Scatter(x=tx, y=ty, mode="lines", line=dict(color="red", width=1)))
        else:
            vert_x = [x, x]
            vert_y = [y_ranges[0], y_ranges[1]]
            frame_data.append(go.Scatter(x=vert_x, y=vert_y, mode="lines",
                                         line=dict(color="green", width=1)))

        frame_data.append(go.Scatter(x=[x_intersect], y=[y_intersect], mode="markers",
                                     marker=dict(color="green", symbol="x")))
        frame_data.append(go.Scatter(x=[x_intersect, x_intersect], y=[0, y_intersect], mode="lines",
                                     line=dict(color="green", dash="dash")))
        frame_data.append(go.Scatter(x=x_vals, y=fun(x_vals), mode="lines",
                                     line=dict(width=2, color="blue")))

        slider_step = {"args": [
            [i],
            {"frame": {"duration": 300, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 300}}
        ],
            "label": i,
            "method": "animate"}
        slider_steps.append(slider_step)
        frames.append(frame_data)


    # create layout
    layout = go.Layout(
        title="Interactive optimization",
        xaxis=dict(title="x", range=x_ranges),
        yaxis=dict(title="y", range=y_ranges),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 2000, "redraw": False},
                                        "fromcurrent": True,
                                        "transition": {"duration": 300,
                                                       "easing": "quadratic-in-out"}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ]
    )

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Iteration: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": slider_steps
    }

    layout["sliders"] = [sliders_dict]

    # create the figure
    fig = go.Figure(
        data=frames[0],
        layout=layout,
        frames=[go.Frame(data=d, name=str(i)) for i, d in enumerate(frames)]
    )

    # save figure at path
    path.mkdir(parents=True, exist_ok=True)
    fig.write_html(path/"optim_interactive.html")


def make_gif(path: Path, name: str) -> None:
    """ Create a .gif from all .pngs found in the path """

    # get all .pngs in path
    img_list = list(path.glob("*.png"))
    img_list.sort()
    frames = []

    # if optimization.gif already exists, overwrite
    if (path/"optimization.gif").exists():
        (path/"optimization.gif").unlink()

    for png_path in img_list:
        frames.append(Image.open(png_path))
    frame_one = frames[0]
    frame_one.save(f"{path.name}/{name}.gif", format="GIF", append_images=frames, save_all=True,
                   duration=1000, loop=0)

    for png in path.glob("*.png"):
        png.unlink(missing_ok=True)


def save_1d_vis(fun: Callable, x_vals: ArrayLike,
                intermediate_vals: ArrayLike, grads: ArrayLike, path: Path) -> None:
    """ Create an animation of the optimization procedure in 1d and save it as a .gif to the Path"""

    # calculate the x- & y-range
    x_ranges = [x_vals.min(), x_vals.max()]
    y_range = fun(x_vals).max() - fun(x_vals).min()
    y_ranges = [fun(x_vals).min() - y_range * 0.2, fun(x_vals).max() + y_range * 0.2]

    # calculate first derivative of fun
    first_derivative = jnp.array([jax.grad(fun)(x).item() for x in x_vals])

    # create layout
    layout = go.Layout(title_text="Optimization with Newton's method",
                       xaxis=dict(range=x_ranges, autorange=False, zeroline=False),
                       yaxis=dict(range=y_ranges, autorange=False, zeroline=False),
                       showlegend=False)

    # for each intermediate optimization step, create a proper figure
    for i, (x, grad) in enumerate(zip(intermediate_vals, grads)):

        fig = go.Figure(layout=layout)
        # plot the function
        fig.add_trace(
            go.Scatter(x=x_vals, y=fun(x_vals), mode="lines", line=dict(width=2, color="blue"))
        )

        # plot the first derivative
        fig.add_trace(
            go.Scatter(x=x_vals, y=first_derivative, mode="lines",
                       line=dict(width=2, color="orange", dash="dot"))
        )

        # for each x-value, calculate the tangent line
        tx, ty, x_intersect = get_tangent_line_points(x, jax.grad(fun)(x).item(), grad, x_vals)

        # get the function value at the intersection of the tangent and the x-axis
        y_intersect = fun(x_intersect)

        # create a frame with the tangent and its intersection with the x-axis
        fig.add_trace(go.Scatter(x=[x], y=[jax.grad(fun)(x).item()], mode="markers",
                                 marker=dict(color="red", symbol="x")))

        # in the last step, don't plot the tangent line
        # instead, plot a vertical line
        if i < len(grads) - 1:
            fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines", line=dict(color="red", width=1)))
        else:
            vert_x = [x, x]
            vert_y = [y_ranges[0], y_ranges[1]]
            fig.add_trace(go.Scatter(x=vert_x, y=vert_y, mode="lines", line=dict(color="green", width=1)))

        fig.add_trace(go.Scatter(x=[x_intersect], y=[y_intersect], mode="markers",
                                 marker=dict(color="green", symbol="x")))
        fig.add_trace(go.Scatter(x=[x_intersect, x_intersect], y=[0, y_intersect], mode="lines",
                                 line=dict(color="green", dash="dash")))

        # if the path does not exist, create it
        path.mkdir(parents=True, exist_ok=True)
        fig.write_image(path/f"{i}.png")

    # create a .gif out of all .pngs
    make_gif(path, "optimization")
