# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:23:08 2021

@author: haohua

First Part - Regression 
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate

# loads the .npy files that cointains the data set provided for the problem
xtrain=np.load('Xtrain_Regression_Part1.npy')
ytrain=np.load('Ytrain_Regression_Part1.npy')

'''Validation of the predictor'''

# simple split into train and test sets (80% and 20%)
x_train, x_test, y_train, y_test=train_test_split(xtrain, ytrain, test_size=0.2, random_state=0)

# regression models: linear and polynomial
reg=LinearRegression()
poli=make_pipeline(PolynomialFeatures(2),LinearRegression(fit_intercept=False)) # pipeline so we can see values simultaneously

# cross-validation on the training set by applying k-fold method, cv=k
scoring="neg_mean_squared_error" # mean squared error
linscore=cross_validate(reg, xtrain, ytrain, scoring=scoring, return_estimator=True, cv=15)
poliscore=cross_validate(poli, xtrain, ytrain, scoring=scoring, return_estimator=True, cv=15)

# evaluate linear and polynomial model. test_score has the mean_squared_error in each run (k=10)
mse_reg=abs(np.average(linscore["test_score"]))
mse_poli=abs(np.average(poliscore["test_score"]))
print('mean MSE Linear:%.4f' % mse_reg)
print('mean MSE Polynomial:%.4f' % mse_poli)
print('maximum MSE from Linear:%.4f' % abs(np.min(linscore["test_score"])))
print('maximum MSE from Polynomial:%.4f' % abs(np.min(poliscore["test_score"])))

# fits the linear model Y^=B0+B1*x with the 70% training set
reg.fit(x_train, y_train)

y_predicted_1=reg.predict(x_test) # evaluation of the predictor using the test set splitted from the data set
mse=mean_squared_error(y_test, y_predicted_1) # focuses on larger errors
R=reg.score(x_train,y_train) # coefficient of determination 
B0=reg.intercept_ # slope of the linear model
B1=reg.coef_ # vector with the B1 corresponding to each feature of x

print('coefficient of determination R^2:', R)
print('coefficient of determination R^2 (estimation):', reg.score(x_test,y_test))
print('Bo:', B0)
print('B1:', B1)
print('Test set MSE:%.4f' % mse)

'''Load Xtest, fit the model using the whole training set, predict y outcomes and save to a .npy file'''

xtest=np.load('Xtest_Regression_Part1.npy') # loads the independent test set
reg.fit(xtrain,ytrain) # trains the linear model with the given training set
y_predicted=reg.predict(xtest) # evaluation of the predictor using the independent test set (corresponding outcomes for professor)
np.save('Ypredict_Regression_Part1.npy',y_predicted) # saves the predicted y^ values into a .npy file


'''Plotting figures '''

# trainy as a function of the 1st feature of our data input x, with its estimated linear regression
plt.scatter(xtrain[:,1],ytrain, color='blue', linewidth=1)
plt.xlabel('Xtrain feature[1]')
plt.ylabel('Ytrain')
plt.grid()
x=np.linspace(min(xtrain[:,1])-1, 1+max(xtrain[:,1]), 100)
y=B0*x+B1[:,1]
plt.plot(x, y, '-r', label='y=B0+B1*x', linewidth=3)
plt.legend(loc='lower left')
plt.figure()

# predicted y^ as a function of the input index i.
plt.scatter(range(len(xtest)), y_predicted, color='green', linewidth=0.1)
plt.xlabel('Index Position of Xtest')
plt.ylabel('Y^ (estimation)')
plt.grid()
plt.figure()

# 1st feature of input x in function of its respective index
plt.scatter(range(len(xtrain)), xtrain[:,1], color='red', linewidth=1)
plt.xlabel('Index Position of Xtrain')
plt.ylabel('Xtrain feature[1]')
plt.grid()
plt.figure()

