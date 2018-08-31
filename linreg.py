# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 01
# Tamara Campos - RA 157324
# João Vítor B. Silva - RA 155951

import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

# function to normalize data
def normalize(data, n):
    for col in range(1, n):
        max = float(data[:,col].max())
        min = float(data[:,col].min())
        if (max == min):
            max = min + 1
        
        diff = float(max - min)
        data[:,col] = (data[:,col] - min) / diff
        
    return data, min, diff

#function to normalize data
def normalizeValidation(data, n, min, diff):
    for col in range(1, n):
        data[:,col] = (data[:,col] - min) / diff
        
    return data

# gradient descent
def gradientDescent(x, y, theta, alpha, m, n, it):
    xTran = x.transpose()

    J = np.zeros(it)

    for i in range(it):
        hyp = np.dot(x, theta)
        loss = np.squeeze(hyp - y)
        J[i] = (np.sum(loss**2)/(2*m))
        gradient = np.squeeze(np.dot(xTran, loss))/m
        theta = np.squeeze(theta - alpha * gradient)

    return J, theta

def normalEquation(x, y):
    inverse = np.linalg.inv(np.dot(x.T, x))
    thetas = np.dot(np.dot(inverse, x.T), y)
    return thetas


def calcMAE(x, y, theta, m):

    hypothesis = np.dot(x, theta)
    MAE_validation = np.sum(abs(hypothesis - y)) / m
    
    return MAE_validation

## MAIN

# Read database
filename = sys.argv[1]
file = open(filename, "r")
line = file.readline()

# Number of features
n = len(line.split(",")[1:]) + 1

# Value dictionaries for non-numerical features
cutValue = {}
cutValue["Fair"] = 1
cutValue["Good"] = 2
cutValue["Very Good"] = 3
cutValue["Premium"] = 4
cutValue["Ideal"] = 5
colorValue = dict(J = 1, I = 2, H = 3, G = 4, F = 5, E = 6, D = 7)
clarityValue = dict(I1 = 1, SI2 = 2, SI1 = 3, VS2 = 4, VS1 = 5, VVS2 = 6, VVS1 = 7, IF = 8)

data, price = [], []
for line in file:
    line = line.split(",")

    temp = []
    temp.append(1)
    temp.append(float(line[0]))
    temp.append(float(cutValue[line[1].replace(chr(34), "")]))
    temp.append(float(colorValue[line[2].replace(chr(34), "")]))
    temp.append(float(clarityValue[line[3].replace(chr(34), "")]))
    temp.append(float(line[4]))
    temp.append(float(line[5]))
    temp.append(float(line[6]))
    temp.append(float(line[7]))
    temp.append(float(line[8]))
    data.append(temp)

    price.append(float(line[9]))

# split training/test
validationData, trainingData = [], []

validationData = data[-8091:]
trainingData = data[:-8091]
validationPrice = price[-8091:] 
trainingPrice = price[:-8091]

validationData, min, diff = normalize(np.array(validationData), n)
trainingData = normalizeValidation(np.array(trainingData), n, min, diff)

validationPrice = np.array(validationPrice)
trainingPrice = np.array(trainingPrice)

m = len(trainingPrice)

# cost
# alphaMax = 0.00000001
alpha = 0.00000001
# itMax = 100000
it = 1000
thetasGD = np.ones(n, dtype=float)
J, thetGD = gradientDescent(trainingData, trainingPrice, thetasGD, alpha, m, n, it)

samplesNumber, _ = np.shape(validationData)
#T = [0.95526007, 0.99452856, 0.96676548, 0.97391538, 0.98008266, 0.97620609, 0.99565883, 0.98043281, 0.97619554, 0.98760811]
MAE_test_reg_GD = calcMAE(validationData, validationPrice, thetasGD, samplesNumber)
print(MAE_test_reg_GD)

#plt.plot(J)
#plt.show()

clf = linear_model.SGDRegressor(max_iter=it, eta0=alpha, learning_rate='constant')
clf.fit(trainingData, trainingPrice)

MAE_test_reg_GD = calcMAE(validationData, validationPrice, clf.coef_, samplesNumber)
print(MAE_test_reg_GD)

thetasNE = normalEquation(trainingData, trainingPrice)
T = [ 0.06855487, 2.81424015, 0.02465473, 0.10490848, 0.19005936, -0.53677227, 0.15405773, -0.00671723, -0.15854809, -0.08902239]
MAE_test_NE = calcMAE(validationData, validationPrice, thetasNE, samplesNumber)
print(MAE_test_NE)
