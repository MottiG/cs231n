import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)  # (1,C)
        correct_class_score = scores[y[i]]  # choose the actual correct class of this row (image)
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i, :]  # the gradient of W_j where j is not the correct class
                dW[:, y[i]] -= X[i, :]  # the gradient of W_i where i is the correct class


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # Add the regularization derivative to dW
    dW += 2*reg*W


    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """

    num_train = X.shape[0]
    rows = np.arange(num_train)
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores = X.dot(W)  # (N,C)
    correct_classes_scores = scores[rows, y]  # from each image take only the score of the correct class
    margins = np.maximum(0, scores - correct_classes_scores[:, np.newaxis] + 1)  # hinge
    margins[rows, y] = 0  # since we want to sum only when j!=i, we need to remove the added deltas (1)
    loss = np.sum(margins) / num_train
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # for each class, we're adding the images that the margin > 0 and subtracting the class itself X times of margin>0
    num_to_add = margins.astype(bool).astype(int)
    num_to_add[rows, y] -= np.sum(num_to_add, axis=1)
    dW += X.T.dot(num_to_add)
    dW /= num_train
    dW += 2*reg*W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
