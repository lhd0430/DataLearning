import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        maxi = np.max(x, axis=1)
        x = np.exp(x - maxi[:, np.newaxis])
        x = x/np.sum(x, keepdims=True, axis=1)
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        maxi = np.max(x, axis=0)
        x = np.exp(x - maxi)
        x = x/np.sum(x, axis=0)
        ### END YOUR CODE

    assert x.shape == orig_shape
    return x





if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
