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

# gradient descent with regularization
def gradientDescentWithRegularization(x, y, alpha, n, m, it, reg):
    xTran = x.transpose()
    theta = np.ones(n, dtype=float)
    J = np.zeros(it)

    for i in range(it):
        hyp = np.dot(x, theta)
        loss = hyp - y['price'].values
        J[i] = (np.sum(loss**2) + reg*np.sum(theta**2))/(2*m)
        gradient = np.squeeze(np.dot(xTran, loss))/m
        theta = np.squeeze(theta - alpha * gradient)

        theta = theta * (1 - alpha * (reg / m)) - alpha * gradient

    return J, theta

def normalEquation(x, y):
    inverse = np.linalg.inv(np.dot(x.T, x))
    thetas = np.dot(np.dot(inverse, x.T), y['price'].values)
    return thetas

def normalEquationWithRegularization(x, y, reg):
    identity = np.identity(x.shape[1])
    identity[0][0] = 0
    inverse = np.linalg.inv(np.dot(x.T, x) + reg*identity)
    thetas = np.dot(np.dot(inverse, x.T), y['price'].values)
    return thetas

def calcMAE(x, y, theta, m):

    hypothesis = np.dot(x, theta)
    MAE_validation = np.sum(abs(hypothesis - y)) / m

    return MAE_validation

def runGD(trainData, trainTarget, n, m, it, validData, validTarget):
    alphas = [0.1, 0.01, 0.001, 0.0001]
    alphas = np.array(alphas)
    colors = ['cyan', 'green', 'magenta', 'blue']

    for i,alphaGD in enumerate(alphas):

        # apply gradient descent
        J, thetas = gradientDescent(trainData, trainTarget, alphaGD, n, m, it)

        printErros("GD alpha=" + str(alphaGD), validData, validTarget, thetas)

        plt.plot(J, colors[i], label='alpha: ' + str(alphaGD))
        plt.legend()
        plt.ylabel('Função de custo J')
        plt.xlabel('Número de itarações')
        plt.title('GD para diferentes taxas de aprendizado')
    plt.show()

def runSklearn(trainData, trainTarget, it, alpha, validData, validTarget):
    clf = linear_model.SGDRegressor(max_iter=it, eta0=alpha, learning_rate='constant')
    clf.fit(trainData, trainTarget['price'].values)

    printErros("SKLearn", validData, validTarget, clf.coef_)


def runNE(trainData, trainTarget, validData, validTarget):
    thetasNE = normalEquation(trainData, trainTarget)

    printErros("NE", validData, validTarget, thetasNE)

def runGDWithRegularization(trainData, trainTarget, n, m, it, validData, validTarget):
    alphaGDR = 0.1
    regularizations = [10, 100, 1000, 10000]
    regularizations = np.array(regularizations)
    colors = ['blue', 'green', 'red', 'purple']

    for i,reg in enumerate(regularizations):
        J, thetasGDR = gradientDescentWithRegularization(trainData, trainTarget, alphaGDR, n, m, it, reg)
        
        printErros("GD alpha=0.1 + Reg=" + str(reg), validData, validTarget, thetasGDR)

        plt.plot(J, colors[i], label='lambda reg: ' + str(reg))
        plt.legend()
        plt.ylabel('Função de custo J')
        plt.xlabel('Número de itarações')
        plt.title('GD com regularização - alpha: 0.1')
    plt.show()


def runNEWithRegularization(trainData, trainTarget, validData, validTarget):
    regularizations = [10, 100, 1000, 10000]
    regularizations = np.array(regularizations)

    for _,reg in enumerate(regularizations):
        thetasNER = normalEquationWithRegularization(trainData, trainTarget, reg)

        printErros("NE + Reg=" + str(reg), validData, validTarget, thetasNER)


def printErros(method, validData, validTarget, thetas):
    y = validTarget['price'].values.ravel()
    numberOfSamples, _ = validData.shape

    print('MSE para ' + method + ': ' + str(mean_squared_error(np.dot(validData, thetas), y)))
    print('MAE para ' + method + ': ' + str(calcMAE(validData, y, thetas, numberOfSamples)))


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

runGD(trainData, trainTarget, n, m, it, validData, validTarget)
runSklearn(trainData, trainTarget, it, alpha, validData, validTarget)
runNE(trainData, trainTarget, validData, validTarget)

runGDWithRegularization(trainData, trainTarget, n, m, it, validData, validTarget)
runNEWithRegularization(trainData, trainTarget, validData, validTarget)
