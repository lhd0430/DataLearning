#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    h1 = data.dot(W1) + b1 # (M, H)
    a1 = sigmoid(h1)

    h2 = a1.dot(W2) + b2 # (M, Dy)
    scores = softmax(h2)
    cost = -np.sum( np.log(scores) * labels )
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    gradh2 = scores - labels

    gradW2 = a1.T.dot(gradh2)
    gradb2 = np.sum(gradh2, axis=0)

    grada1 = gradh2.dot(W2.T)

    gradh1 = grada1 * sigmoid_grad(a1)

    gradW1 = data.T.dot(gradh1)
    gradb1 = np.sum(gradh1, axis=0)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad





if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
