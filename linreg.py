# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 01
# Tamara Campos - RA 157324
# João Vítor B. Silva - RA 155951

import sys
import numpy as np
from matplotlib import pyplot as plt

# function to normalize data
def normalize(data, n):
    for col in range(n):
        max = float(data[:][col].max())
        min = float(data[:][col].min())
        if (max == min):
            max = min + 1
        
        diff = float(max - min)
        data[:,col] = (data[:,col] - min) / diff

# gradient descent
def gradientDescent(x, y, theta, alpha, m, n, it):
    xTran = x.transpose()

    J = np.zeros(it)

    for i in range(it):
        hyp = np.dot(x, theta)
        loss = np.squeeze(hyp - y)
        J[i] = (np.sum(loss**2)/(2*m))
        gradient = np.squeeze(np.dot(xTrans, loss))/m
        theta = np.squeeze(theta - alpha * gradient)

    return J, theta

## MAIN

# Read database
filename = sys.argv[1]
file = open(filename, "r")
line = file.readline()

# Number of features
n = len(line.split(",")[1:])

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
    temp.append(float(line[1]))
    temp.append(float(cutValue[line[2].replace(chr(34), "")]))
    temp.append(float(colorValue[line[3].replace(chr(34), "")]))
    temp.append(float(clarityValue[line[4].replace(chr(34), "")]))
    temp.append(float(line[5]))
    temp.append(float(line[6]))
    temp.append(float(line[8]))
    temp.append(float(line[9]))
    temp.append(float(line[10]))
    data.append(temp)

    price.append(float(line[7]))


# normalize dat
data = normalize(data, n)

# split training/test
testData, trainingData = [], []
for f in data:
    testData.append(f[:8091])
    trainingData.append(f[8091:])
testPrice, trainingPrice = price[:8091], price[8091:]

testData = np.array(testData)
trainingData = np.array(trainingData)

testPrice = np.array(testPrice)
trainingPrice = np.array(trainingPrice)

m = len(trainingPrice)

# cost
alpha = 0.1
it = 100
thetas = np.ones(n, dtype=float)
J, thetas = gradientDescent(trainingData, trainingPrice, thetas, alpha, m, n, it)

plt.plot(J)
plt.show()
