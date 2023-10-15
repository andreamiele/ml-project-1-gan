# -*- coding: utf-8 -*-
"""function for plot."""
import matplotlib.pyplot as plt
import numpy as np


def prediction(w0, w1, mean_x, std_x):
    """Get the regression line from the model."""
    x = np.arange(1.2, 2, 0.01)
    x_normalized = (x - mean_x) / std_x
    return x, w0 + w1 * x_normalized


def gradient_descent_visualization(
    gradient_losses,
    gradient_ws,
    grid_losses,
    grid_w0,
    grid_w1,
    mean_x,
    std_x,
    height,
    weight,
    n_iter=None,
):
    """Visualize how the loss value changes until n_iter."""
    fig = base_visualization(
        grid_losses, grid_w0, grid_w1, mean_x, std_x, height, weight
    )

    ws_to_be_plotted = np.stack(gradient_ws)
    if n_iter is not None:
        ws_to_be_plotted = ws_to_be_plotted[:n_iter]

    ax1, ax2 = fig.get_axes()[0], fig.get_axes()[2]
    ax1.plot(
        ws_to_be_plotted[:, 0],
        ws_to_be_plotted[:, 1],
        marker="o",
        color="w",
        markersize=10,
    )
    pred_x, pred_y = prediction(
        ws_to_be_plotted[-1, 0], ws_to_be_plotted[-1, 1], mean_x, std_x
    )
    ax2.plot(pred_x, pred_y, "r")

    return fig
