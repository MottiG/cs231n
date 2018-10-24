import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        f = X[i].dot(W)  # (1,C)
        f -= np.max(f)  # stability normalization
        sum_i = np.sum(np.exp(f))
        p = np.exp(f[y[i]]) / sum_i
        loss += -np.log(p)
        for j in range(num_classes):
            p = np.exp(f[j]) / sum_i
            if j == y[i]:
                dW[:, j] += (p - 1)*X[i]
            else:
                dW[:, j] += p*X[i]


    loss /= num_train
    loss += reg * np.sum(W*W)

    dW /= num_train
    dW += 2*reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    rows = np.arange(num_train)

    f = X.dot(W)  # (N,C)  X=(N,D), W=(D,C)
    f -= np.amax(f, axis=1)[:, np.newaxis]  # stability normalization
    f = np.exp(f)
    p = f / np.sum(f, axis=1)[:, np.newaxis]
    loss = np.sum(-np.log(p[rows, y]))

    p[rows, y] -= 1  # for j==y[i] we have dW[:,j = (p-1)*X[i]
    dW += X.T.dot(p)

    loss /= num_train
    loss += reg * np.sum(W*W)
    dW /= num_train
    dW += 2*reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
