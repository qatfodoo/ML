import pdb
import random
import pylab as pl
from scipy.optimize import fmin_bfgs
import numpy as np
from gradDescent import basic_gradient_descent, approximate_gradient_descent

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def regressionPlot(X, Y, order, regression_method):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function

    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = designMatrix(X, order)
    # compute the weight vector
    w = regression_method(X, Y, phi)

    # produce a plot of the values of the function 
    pts = np.array([[p] for p in pl.linspace(min(X), max(X), 100)])
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
    pl.plot(pts, Yp.tolist()[0])
    return w

def designMatrix(X, order):
    return np.array([X[:, 0] **i for i in range(order + 1)]).transpose()

def regressionFit(X, Y, phi):
    tmp = np.linalg.inv(np.dot(phi.transpose(), phi))
    tmp = np.dot(tmp, phi.transpose())
    return np.dot(tmp, Y)

def gradientFit(X, Y, phi):
    weights, count, score = approximate_gradient_descent(
      np.array([[i] for i in phi[0]]),
      lambda w: SSE(phi, Y, w),
    )
    return weights


def SSE(phi, Y, weights):
    err = 0
    for i in range(len(Y[:, 0])):
        appr = sum([weights[j][0] * phi[i][j] for j in range(len(phi[i]))])
        err += (Y[:, 0][i] - appr)**2
    return err

def SSE_prime(phi, Y, weights):
    score = SSE(phi, Y, weights)
    dw = np.zeros(weights.shape)
    grad = np.zeros(weights.shape)
    l = 0.001
    for i in range(weights.shape[0]):
        dw[i] = l
        grad[i] = (SSE(phi, Y, weights + dw) - score) / l
        dw[i] = 0
    return grad

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def bishopCurveData():
    # y = sin(2 pi x) + N(0,0.3),
    return getData('curvefitting.txt')

def regressAData():
    return getData('regressA_train.txt')

def regressBData():
    return getData('regressB_train.txt')

def validateData():
    return getData('regress_validate.txt')
