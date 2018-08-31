# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 01
# Tamara Campos - RA 157324
# João Vítor B. Silva - RA 155951

import sys
import numpy as np
import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

# gradient descent
def gradientDescent(x, y, alpha, n, m, it):
    xTran = x.transpose()
    theta = np.ones(n, dtype=float)
    J = np.zeros(it)

    for i in range(it):
        hyp = np.dot(x, theta)
        loss = hyp - y['price'].values
        J[i] = (np.sum(loss**2)/(2*m))
        gradient = np.squeeze(np.dot(xTran, loss))/m
        theta = np.squeeze(theta - alpha * gradient)

    return J, theta

def normalEquation(x, y):
    inverse = np.linalg.inv(np.dot(x.T, x))
    thetas = np.dot(np.dot(inverse, x.T), y['price'].values)
    return thetas

def calcMAE(x, y, theta, m):

    hypothesis = np.dot(x, theta)
    MAE_validation = np.sum(abs(hypothesis - y)) / m

    return MAE_validation

## MAIN

# disable SettingWithCopyWarning warnings
pd.options.mode.chained_assignment = None  # default='warn'

# Read database
filename = sys.argv[1]
data = pd.read_csv(filename)

# Map values of non-numerical features
cutValue = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
colorValue = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarityValue = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

data['color'] = data['color'].map(colorValue)
data['cut'] = data['cut'].map(cutValue)
data['clarity'] = data['clarity'].map(clarityValue)

# separate target from data sets
target = data.drop(data.columns.values[:-1], axis='columns')
data = data.drop('price', axis='columns')

# split dataset into train and validation
trainData, trainTarget = data.iloc[8091:], target.iloc[8091:]
validData, validTarget = data.iloc[:8091], target.iloc[:8091]

# normalize features
for c in trainData.columns.values:
    max = float(trainData[c].max())
    min = float(trainData[c].min())

    if (max == min):
        max = min + 1

    diff = float(max - min)

    trainData[c] -= min
    trainData[c]  /= diff

    validData[c] -= min
    validData[c] /= diff

# TODO: save min and diff of each column for posterior use with test data(?)

# calculate cost
m, n = trainData.shape
it = 10000 # itMax = 100000
alpha = 0.01 # alphaMax = 0.00000001

# apply gradient descent
J, thetas = gradientDescent(trainData, trainTarget, alpha, n, m, it)

plt.plot(J)
plt.show()

clf = linear_model.SGDRegressor(max_iter=it, eta0=alpha, learning_rate='constant')
clf.fit(trainData, trainTarget['price'].values)

thetasNE = normalEquation(trainData, trainTarget)

y = trainTarget['price'].values.ravel()
print('MSE')
print('Our Model: ' + str(mean_squared_error(np.dot(trainData, thetas), y)))
print('SKLearn: ' + str(mean_squared_error(np.dot(trainData, clf.coef_), y)))
print('Normal Equation: ' + str(mean_squared_error(np.dot(trainData, thetasNE), y)))

print('MAE')
print('Our Model: ' + str(calcMAE(trainData, y, thetas, m)))
print('SKLearn: ' + str(calcMAE(trainData, y, clf.coef_, m)))
print('Normal Equation: ' + str(calcMAE(trainData, y, thetasNE, m)))
