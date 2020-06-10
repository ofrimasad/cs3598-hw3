import numpy as np
from random import shuffle
from random import randrange

def softmax_loss_vectorized(W, X, y, reg=0):
  """
  Softmax loss function, vectorized version. Same as HW1.
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
  N = X.shape[0]
  C = W.shape[1]

  f = X.dot(W)
  f -= np.matrix(np.max(f, axis=1)).T

  term1 = -f[np.arange(N), y]
  sum_j = np.sum(np.exp(f), axis=1)
  term2 = np.log(sum_j)
  loss = term1 + term2
  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  loss = np.squeeze(loss)

  coef = np.exp(f) / np.matrix(sum_j).T
  coef[np.arange(N), y] -= 1
  coef = np.squeeze(coef)

  dW = X.T.dot(coef)
  dW /= N
  dW += reg * W
  dW = np.squeeze(dW)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW

def grad_check(f, x, analytic_grad, num_checks=10, h=1e-5):
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evaluate f(x + h)
        x[ix] = oldval - h # increment by h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print ('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))


