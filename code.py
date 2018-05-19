#!/usr/bin/env python3

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def o_f(W):
    """ The objective function -log(p(Y|W))
    The equation for this can be found in README.md with the full explanation
    in equations.pdf
    """

    W = np.reshape(W, (D, 2))
    WWT = np.dot(W, np.transpose(W))

    Id = np.eye(D)

    A = np.trace(np.dot(np.dot(Y, np.linalg.inv(WWT + Id)), np.transpose(Y)))
    B = np.log(np.linalg.det(WWT + Id))
    C = D * np.log(2*np.pi)

    return 0.5 * N * (A + B + C)


def o_dfx(W):
    """ Gradient of the objective function dL/dW
    The equation for this can be found in README.md with the full explanation
    in equations.pdf
    """

    W = np.reshape(W, (D, 2))
    WWT = np.dot(W, np.transpose(W))

    Id = np.eye(D)

    # Inverse of the variance
    inv_var = np.linalg.inv(WWT + Id)

    gradient = np.empty(W.shape)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):

            # Construct the single entry matrix
            J = np.zeros(np.shape(W))
            J[i, j] = 1

            # Calculate the partial derivative of WW^T as JWWJ
            JWWJ = np.dot(J, np.transpose(W)) + np.dot(W, np.transpose(J))

            # Calculate the partial derivative of the inverse of the variance
            d_inv_var = np.dot(np.dot(-inv_var, JWWJ), inv_var)

            A = np.trace(np.dot(np.dot(np.transpose(Y), Y), d_inv_var))
            B = np.trace(np.dot(inv_var, JWWJ))

            gradient[i, j] = N * 0.5 * (A + B)

    gradient = np.reshape(gradient, (20,))

    return gradient


# Generate our data

def f(xlist):
    return np.array([[x*np.sin(x), x*np.cos(x)] for x in xlist])


D = 10
W = np.random.randn(20)
W = W.reshape((D, 2))

N = 100
xsmall = np.linspace(0, 4*np.pi, N)
x = f(xsmall)

Y = np.dot(x, np.transpose(W))

# Reset W for gradient descent

W = np.random.randn(20)
W = np.reshape(W, (20,))

# Representation learning of X

Wstar = opt.fmin_cg(o_f, W, fprime=o_dfx)   # Optimise using gradient descent
Wprime = np.reshape(Wstar, (10, 2))

WTW = np.dot(np.transpose(Wprime), Wprime)
learned = np.dot(Y, np.dot(Wprime, np.linalg.inv(WTW)))

# Graphical the results

plt.figure(1)
plt.scatter(x[:, 0], x[:, 1])
plt.xlabel("$x_i \sin(x_i)$")
plt.ylabel("$x_i \cos(x_i)$")
plt.title("Original parameters X")

plt.figure(2)
plt.scatter(learned[:, 0], learned[:, 1])
plt.xlabel("$x_i \sin(x_i)$")
plt.ylabel("$x_i \cos(x_i)$")
plt.title("Learned X$^\prime$")

plt.show()
