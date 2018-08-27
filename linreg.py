# coding: utf-8

# MC886/MO444 - 2018s2 - Assignment 01
# Tamara Campos - RA XXXXXX
# João Vítor B. Silva - RA 155951

import sys
import numpy as np

# Read database
filename = sys.argv[1]
file = open(filename, "r")
line = file.readline()

# Number of features
n = len(line.split(",")[1:])

# Feature lists
carat = []
cut = []
color = []
clarity = []
depth = []
table = []
price = []
x = []
y = []
z = []

# Value dictionaries for non-numerical features
cutValue = {}
cutValue["Fair"] = 1
cutValue["Good"] = 2
cutValue["Very Good"] = 3
cutValue["Premium"] = 4
cutValue["Ideal"] = 5
colorValue = dict(J = 1, I = 2, H = 3, G = 4, F = 5, E = 6, D = 7)
clarityValue = dict(I1 = 1, SI2 = 2, SI1 = 3, VS2 = 4, VS1 = 5, VVS2 = 6, VVS1 = 7, IF = 8)

# Append values to lists
for line in file:
    line = line.split(",")

    carat.append(line[1])
    cut.append(cutValue[line[2].replace(chr(34), "")])
    color.append(colorValue[line[3].replace(chr(34), "")])
    clarity.append(clarityValue[line[4].replace(chr(34), "")])
    depth.append(float(line[5]))
    table.append(float(line[6]))
    price.append(float(line[7]))
    x.append(float(line[8]))
    y.append(float(line[9]))
    z.append(float(line[10]))

# convert lists to numpy arrays
carat = np.array(carat)
cut = np.array(cut)
color = np.array(color)
clarity = np.array(clarity)
depth = np.array(depth)
table = np.array(table)
price = np.array(price)
x = np.array(x)
y = np.array(y)
z = np.array(z)
