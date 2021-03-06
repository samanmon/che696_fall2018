# -*- coding: utf-8 -*-

"""
Author: Saman Moniri (U. Michigan, November 2018)
Course project for ChE 696: On-ramp to Data Science

Project description:
Train a linear regression model using random numbers.
Use trained model to predict the total payment for all the claims in thousands of Swedish Kronor (y) given the total number of claims (x).
The “Auto Insurance in Sweden” dataset is available at:
https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/

"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.metrics import *
from sklearn.datasets import fetch_mldata
from csv import reader

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Use random numbers for training =
xtrain = np.linspace(0,150, 150)
ytrain = 5*(xtrain+3*np.random.random(xtrain.shape[0]))

# reshape input to 2d arrays (code expects it this way)
xtrain = xtrain.reshape(xtrain.shape[0], -1)
ytrain = ytrain.reshape(ytrain.shape[0], -1)

# create the machine learning model object
regr = linear_model.LinearRegression()

# fit the machine learning model object
regr.fit(xtrain, ytrain)

# Use the "Auto Insurance in Sweden" dataset (link below) to predict the total payment for all the claims in thousands of Swedish Kronor (y) given the total number of claims (x)
#https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
filename = 'insurance.csv'
dataset = load_csv(filename)
dataset = dataset[1:] # exclude header
xtest = []
ytest = []

for i in range(len(dataset)):
    xtest.append(dataset[i][0]) # x
    ytest.append(dataset[i][1]) # y

xtest = np.array(xtest) # convert to np array
ytest = np.array(ytest) # convert to np array
xtest = xtest.reshape(xtest.shape[0], -1) # reshape to 2d array
ytest = ytest.reshape(ytest.shape[0], -1) # reshape to 2d array
xtest = xtest.astype(float) # change type to float
ytest = ytest.astype(float) # change type to float

# predict values using the fitted model
ypred = regr.predict(xtest)

# the coefficients
print('Coefficients: \n', regr.coef_)
# the mean squared error
print("Mean squared error: %.2f" % mean_squared_error(ytest, ypred))
# variance score
print('Variance score: %.2f' % r2_score(ytest, ypred))

plt.figure()
plt.scatter(xtest[:,0], ytest,  color='black', label="Testing Data")
plt.scatter(xtrain[:,0], ytrain, facecolors="none", edgecolors="blue", label="Training Data")
plt.plot(xtest[:,0], ypred, color='red', linewidth=3, label="Prediction")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.show()
