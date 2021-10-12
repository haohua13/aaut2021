# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:23:20 2021

@author: haohua
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split   
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import cross_validate

# loads the .npy files that cointains the data set provided for the problem
xtrain=np.load('Xtrain_Regression_Part2.npy')
ytrain=np.load('Ytrain_Regression_Part2.npy')

'''Isolation Forest: tree-based outlier detection algorithm'''
'''
forest = IsolationForest(random_state=1, contamination=0.09) # number of outliers is maximum 10% of the data set
outlier = forest.fit_predict(xtrain) # returns -1 for outliers and 1 for inliers
remove = outlier != -1 # only selects inliers
xtrain, ytrain = xtrain[remove,:], ytrain[remove]
print(xtrain.shape,ytrain.shape) # data set shape with outliers removed
'''

'''EllipticEnvelope: outlier detection for gaussian distributed dataset'''
'''
ee = EllipticEnvelope(random_state=1)
outlier = ee.fit_predict(xtrain) # fits the model to the data set x and returns the labels 1 for inliers, -1 outliers
remove = outlier != -1
xtrain, ytrain = xtrain[remove,:], ytrain[remove]
print(xtrain.shape,ytrain.shape) # data set shape with outliers removed
'''

'''Unsupervised outlier detection using Local Outlier Factor (LOF)'''
'''
lof = LocalOutlierFactor(contamination=0.1)
outlier = lof.fit_predict(xtrain)
remove = outlier != -1
xtrain, ytrain = xtrain[remove,:], ytrain[remove]
print(xtrain.shape,ytrain.shape) # data set shape with outliers removed
'''


'''One-Class SVM'''
'''
sv = OneClassSVM(nu=0.11)
outlier = sv.fit_predict(xtrain)
remove = outlier != -1
xtrain, ytrain = xtrain[remove,:], ytrain[remove]
print(xtrain.shape,ytrain.shape) # data set shape with outliers removed
'''


'''Validation of the predictor'''

# simple split into train and test sets (80% and 20%)
x_train, x_test, y_train, y_test=train_test_split(xtrain, ytrain, test_size=0.2, random_state=0)

# regression models: linear and polynomial
reg=LinearRegression()
poli=make_pipeline(PolynomialFeatures(2),LinearRegression(fit_intercept=False)) # pipeline so we can see values simultaneously

# cross-validation on the training set by applying k-fold method, cv=k
scoring="neg_mean_squared_error" # mean squared error
linscore=cross_validate(reg, x_train, y_train, scoring=scoring, return_estimator=True, cv=10)
poliscore=cross_validate(poli, x_train, y_train, scoring=scoring, return_estimator=True, cv=10)

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