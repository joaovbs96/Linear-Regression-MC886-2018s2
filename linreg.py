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

# execução: linreg.py [train_data] [test_data]

# disable SettingWithCopyWarning warnings
pd.options.mode.chained_assignment = None  # default='warn'

# Read train database
filename = sys.argv[1]
data = pd.read_csv(filename)

# read test database
filename = sys.argv[2]
testData = pd.read_csv(filename)

# Map values of non-numerical features
cutValue = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
colorValue = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarityValue = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

data['color'] = data['color'].map(colorValue)
data['cut'] = data['cut'].map(cutValue)
data['clarity'] = data['clarity'].map(clarityValue)

testData['color'] = testData['color'].map(colorValue)
testData['cut'] = testData['cut'].map(cutValue)
testData['clarity'] = testData['clarity'].map(clarityValue)

# insert bias
m, _ = data.shape
data.insert(0, 'bias', np.array(m*[1.0]))

m, _ = testData.shape
testData.insert(0, 'bias', np.array(m*[1.0]))

# insert volume feature
volume = np.multiply(data['x'].values, data['y'].values)
volume = np.multiply(volume, data['z'].values)
data.insert(5, 'volume',  volume)
data = data.drop(['x', 'y', 'z'], axis='columns')

testVolume = np.multiply(testData['x'].values, testData['y'].values)
testVolume = np.multiply(testVolume, testData['z'].values)
testData.insert(5, 'volume',  testVolume)
testData = testData.drop(['x', 'y', 'z'], axis='columns')

# separate target from data sets
target = data.drop(data.columns.values[:-1], axis='columns')
data = data.drop('price', axis='columns')

targetTest = testData.drop(testData.columns.values[:-1], axis='columns')
testData = testData.drop('price', axis='columns')

# split dataset into train and validation
trainData, trainTarget = data.iloc[8091:], target.iloc[8091:]
validData, validTarget = data.iloc[:8091], target.iloc[:8091]

# cut^2
trainData['cut'] = trainData['cut']**2
validData['cut'] = validData['cut']**2
testData['cut'] = testData['cut']**2

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

        testData[c] -= min
        testData[c] /= diff


# calculate cost
m, n = trainData.shape
it = 100000
r = 10
alpha = 0.1
yV = validTarget['price'].values.ravel()
yTr = trainTarget['price'].values.ravel()
yTe = targetTest['price'].values.ravel()

# execute GD with regularization
print('GD:')
J, thetas = gradientDescentReg(trainData, trainTarget, alpha, n, m, it, r)
print('Train: ' + str(calcMAE(trainData, yTr, thetas, m)))
print('Validation: ' + str(calcMAE(validData, yV, thetas, m)))
print('Test: ' + str(calcMAE(testData, yTe, thetas, m)))
print()

# plot graph for GD with regularization
plt.plot(J, 'blue')
plt.ylabel('Função de custo J')
plt.xlabel('Número de iterações')
plt.title('DG para alpha 0.1 e regularização 10')
plt.savefig('GDModel.png')
plt.gcf().clear()

# execute normal equation
print('NE:')
thetasNE = normalEquationReg(trainData, trainTarget, r)
print('Train: ' + str(calcMAE(trainData, yTr, thetasNE, m)))
thetasNE = normalEquationReg(validData, validTarget, r)
print('Validation: ' + str(calcMAE(validData, yV, thetasNE, m)))
thetasNE = normalEquationReg(testData, targetTest, r)
print('Test: ' + str(calcMAE(testData, yTe, thetasNE, m)))
print()

# execute Sklearn method
clf = linear_model.SGDRegressor(max_iter = it, eta0=alpha, learning_rate = 'constant')
clf.fit(trainData, trainTarget['price'].values)
print('SKLearn:')
print("Train: " + str(metrics.mean_absolute_error(trainTarget, clf.predict(trainData))))
print("Validation: " + str(metrics.mean_absolute_error(validTarget, clf.predict(validData))))
print("Test: " + str(metrics.mean_absolute_error(targetTest, clf.predict(testData))))
