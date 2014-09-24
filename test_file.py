from gradDescent import *
from homework1 import *
import numpy as np


"""Compute the maximum likelyhood estimators, directly and approximately"""

X, Y = bishopCurveData()
regressionPlot(X, Y, 0, gradientFit)
regressionPlot(X, Y, 1, gradientFit)
regressionPlot(X, Y, 4, gradientFit)
regressionPlot(X, Y, 9, gradientFit)
#pl.show()

regressionPlot(X, Y, 0, regressionFit)
regressionPlot(X, Y, 1, regressionFit)
regressionPlot(X, Y, 4, regressionFit)
regressionPlot(X, Y, 9, regressionFit)



"""
Some convex functions to test the gradient descent: the quadratic bowl and an inverted gaussian
"""

mu = np.array([[4, 5, 6, 1, 3]]).transpose()
theta_0 = np.array([np.random.random(len(mu))]).transpose()

def quadratic_bowl(mu, x):
  return sum([(x[0] - x[1])**2 for x in zip(mu[:, 0], x[:, 0])])

def quadratic_bowl_gradient(mu, x):
  return np.array([[2 * (x[0] - x[1]) for x in zip(x[:, 0], mu[:, 0])]]).transpose()

theta, count, score = basic_gradient_descent(
  theta_0,
  lambda x: quadratic_bowl_gradient(mu, x),
  lambda x: quadratic_bowl(mu, x),
)
print 'quadratic bowl: theta', theta, '#iterations', count, 'score', score
# theta is close to mu





def gaussian_distribution(mu, sigma_inv, x):
  """
  return exp(0.5 (x-mu)T sigma_inv (x-mu)), which is the gaussian distribution multiplied by a constant
  """
  return np.exp( - np.dot( np.dot((x - mu).transpose(), sigma_inv), x - mu) / 2.0)[0][0]

def gaussian_gradient(mu, sigma_inv, x):
  """
  return sigma_inv (x-mu)
  """
  return np.dot(sigma_inv, x - mu)

sigma = 10 * np.eye(len(mu))
sigma_inv = np.linalg.inv(sigma)

theta, count, score = basic_gradient_descent(
  theta_0,
  lambda x: gaussian_gradient(mu, sigma_inv, x),
  lambda x: - gaussian_distribution(mu, sigma_inv, x),
)
print 'gaussian: theta', theta, '#iterations', count, 'score', score

theta, count, score = approximate_gradient_descent(
  theta_0,
  lambda x: - gaussian_distribution(mu, sigma_inv, x),
)
print 'approximate gaussian: theta', theta, '#iterations', count, 'score', score
# With both methods, theta is close to mu.



"""A non convex function: x -> x^4 + 3x^2 + x """
theta, count, score = basic_gradient_descent(
  np.array([[-10]]),
  lambda x: np.array([[4 * x[0][0] ** 3 - 6 * x[0][0] + 1]]),
  lambda x: x[0][0] **4 - 3 * x[0][0] ** 2 + x[0][0],
)
print 'non convex function 1: theta', theta, '#iterations', count, 'score', score

theta, count, score = approximate_gradient_descent(
  np.array([-10]),
  lambda x: sum([a **4 - 3 * a ** 2 + a for a in x[0]]),
)
print 'approximate non convex function 1: theta', theta, '#iterations', count, 'score', score
# Again, both methods give the same result

theta, count, score = basic_gradient_descent(
  np.array([[10]]),
  lambda x: np.array([[4 * x[0][0] ** 3 - 6 * x[0][0] + 1]]),
  lambda x: x[0][0] **4 - 3 * x[0][0] ** 2 + x[0][0],
)
print 'non convex function 2: theta', theta, '#iterations', count, 'score', score
theta, count, score = basic_gradient_descent(
  np.array([[10]]),
  lambda x: np.array([[4 * x[0][0] ** 3 - 6 * x[0][0] + 1]]),
  lambda x: x[0][0] **4 - 3 * x[0][0] ** 2 + x[0][0],
  step = 2,
)
print 'non convex function 2 with large step: theta', theta, '#iterations', count, 'score', score
# With a big enough step, it is possible to avoid being trapped in a minimum non global

from scipy import optimize
optimize.fmin_bfgs(lambda x: x**4 - 3*x**2 + x, np.array([10]))
#find the result in fewer steps
