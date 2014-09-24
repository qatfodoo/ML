import numpy as np

def basic_gradient_descent(theta, gradient, objective, step=0.001, threshold=0.000001):
  """
  A basic gradient descent
  theta specify the parameter to optimize
  grandient and objective are functions that compute the gradient and objective score for a given theta
  step stands for the step size of the grafient descent
  If the score improvement is smaller than the threshold, we stop the algorithm
  """
  score, improvement, count = objective(theta), threshold + 1, 0
  while improvement >= threshold or improvement < 0:
    if improvement < 0: step = step /2.0 # This to avoid infinite loops that never reach the threshold
    count += 1
    grad = gradient(theta)
    norm = sum([x**2 for x in grad[0]])**0.5
    if norm == 0: return theta, count, score
    theta = theta - step * grad / norm
    obj = objective(theta)
    improvement, score = score - obj, obj
  return theta, count, score


def approximate_gradient_descent(theta, objective, step=0.001, threshold=0.000001):
  """ A basic gradient descent without gradient function"""
  if len(theta.shape) == 1: theta = np.array([theta])

  """ Approximate the gradient with the objective function"""
  def approximate(x):
    score = objective(x)
    dx = np.zeros(x.shape)
    grad = np.zeros(x.shape)
    l = step / 2.0
    for i in range(x.shape[0]):
      dx[i][0] = l
      grad[i][0] = (objective(x + dx) - score) / l
      dx[i][0] = 0
    return grad

  return basic_gradient_descent(theta, approximate, objective, step=step, threshold=threshold)
