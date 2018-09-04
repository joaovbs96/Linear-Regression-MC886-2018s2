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
import sklearn.metrics as metrics

# calculates gradient descent
def gradientDescent(x, y, alpha, n, m, it):
    xTran = x.transpose()
    thetas = np.ones(n, dtype=float)
    J = np.zeros(it)

    for i in range(it):
        hypothesis = np.dot(x, thetas)
        diff = hypothesis - y['price'].values
        J[i] = ((np.sum(diff**2)/(2*m)))
        gradient = np.squeeze(np.dot(xTran, diff))/m
        thetas = np.squeeze(thetas - alpha * gradient)

    return J, thetas

# calculates gradient descent with regularization
def gradientDescentReg(x, y, alpha, n, m, it, reg):
    xTran = x.transpose()
    thetas = np.ones(n, dtype=float)
    J = np.zeros(it)

    for i in range(it):
        hypothesis = np.dot(x, thetas)
        diff = hypothesis - y['price'].values
        J[i] = (np.sum(diff**2) + reg*np.sum(thetas**2))/(2*m)
        gradient = np.squeeze(np.dot(xTran, diff))/m
        thetas = np.squeeze(thetas - alpha * gradient)

        thetas = thetas * (1 - alpha * (reg / m)) - alpha * gradient

    return J, thetas

# function to calculate normal equation
def normalEquation(x, y):
    inverse = np.linalg.inv(np.dot(x.T, x))
    thetas = np.dot(np.dot(inverse, x.T), y['price'].values)

    return thetas

# function to calculate normal equation with regularization
def normalEquationReg(x, y, reg):
    identity = np.identity(x.shape[1])
    identity[0][0] = 0
    inverse = np.linalg.inv(np.dot(x.T, x) + reg*identity)
    thetas = np.dot(np.dot(inverse, x.T), y['price'].values)

    return thetas

# function to calculate mean absolute error
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

# insert bias
m, _ = data.shape
data.insert(0, 'bias', np.array(m*[1.0]))

# insert volume feature
volume = np.multiply(data['x'].values, data['y'].values)
volume = np.multiply(volume, data['z'].values)
data.insert(5, 'volume',  volume)
data = data.drop(['x', 'y', 'z'], axis='columns')

#print(data)
#sys.exit(0)

# separate target from data sets
target = data.drop(data.columns.values[:-1], axis='columns')
data = data.drop('price', axis='columns')

# split dataset into train and validation
trainData, trainTarget = data.iloc[8091:], target.iloc[8091:]
validData, validTarget = data.iloc[:8091], target.iloc[:8091]

trainData['cut'] = trainData['cut']**2

# normalize features
for c in trainData.columns.values:
    if(c != 'bias'):
        max = float(trainData[c].max())
        min = float(trainData[c].min())

        if (max == min):
            max = min + 1

        diff = float(max - min)

        trainData[c] -= min
        trainData[c]  /= diff

        validData[c] -= min
        validData[c] /= diff


# calculate cost
m, n = trainData.shape

y = validTarget['price'].values.ravel()


it = 100000
alpha = 0.1
clf = linear_model.SGDRegressor(max_iter = it, eta0=alpha, learning_rate = 'constant')
clf.fit(trainData, trainTarget['price'].values)
targetPrediction = clf.predict(validData)
print('SKLearn:')
print("MAE: " + str(metrics.mean_absolute_error(y, targetPrediction)))
print('MAE2: ' + str(calcMAE(validData, y, clf.coef_, m)))
