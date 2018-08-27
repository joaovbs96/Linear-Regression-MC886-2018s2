# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 01
# Tamara Campos - RA 157324
# João Vítor B. Silva - RA 155951

import sys
import numpy as np

# Read database
filename = sys.argv[1]
file = open(filename, "r")
line = file.readline()

# Number of features
n = len(line.split(",")[1:]) - 1

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
    temp.append(float(line[1]))
    temp.append(cutValue[line[2].replace(chr(34), "")])
    temp.append(colorValue[line[3].replace(chr(34), "")])
    temp.append(clarityValue[line[4].replace(chr(34), "")])
    temp.append(float(line[5]))
    temp.append(float(line[6]))
    temp.append(float(line[8]))
    temp.append(float(line[9]))
    temp.append(float(line[10]))
    data.append(temp)

    price.append(float(line[7]))

# convert lists to numpy arrays
print(data)
data = np.array(data)
print(data)
