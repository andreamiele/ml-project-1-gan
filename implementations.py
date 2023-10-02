import numpy as np
import matplotlib.pyplot as plt
import datetime
from helpers import *
from plots import gradient_descent_visualization
from ipywidgets import IntSlider, interact


def generic_regression(y, tx, initial_w, max_iters, gamma, batch_size, grad, loss):
  ws = [initial_w]
  losses = []
  w = initial_w
  for n_iter in range(max_iters):
    g = grad(y, tx, w, batch_size)
    l = loss(y, tx, w)

    wold = w
    w = w - gamma * g
    if np.linalg.norm(w - wold) == 0:
      break

    ws.append(w)
    losses.append(l)
    # print(
    #    "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
    #        bi=n_iter, ti=max_iters - 1, l=l, w0=w[0], w1=w[1]
    #    )
    # )
  
  return losses, ws


def mse_loss(y, tx, w):
  return (np.linalg.norm(y - np.dot(tx, w), ord=2) ** 2) / (2 * y.shape[0])


def mse_gradient(y, tx, w, batch_size):
  return -np.dot(np.transpose(tx), y - np.dot(tx, w)) / y.shape[0]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
  return generic_regression(
    y, tx, initial_w, max_iters, gamma, -1, mse_gradient, mse_loss
  )


def mse_stoch_gradient(y, tx, w, batch_size):
  stochastic_grad = np.zeros((2,))
  for mini_batch_y, mini_batch_x in batch_iter(y, tx, batch_size):
    tmp = mini_batch_y - np.dot(mini_batch_x, w)
    stochastic_grad += np.array([tmp[0], tmp[0] * mini_batch_x[0][1]])
  return -stochastic_grad / batch_size

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
  return generic_regression(
      y, tx, initial_w, max_iters, gamma, 1, mse_stoch_gradient, mse_loss
    )