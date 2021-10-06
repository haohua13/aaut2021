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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

xtrain = np.load('Xtrain_Regression_Part2.npy') # loads the data set x
ytrain = np.load('Ytrain_Regression_Part2.npy') # loads the data set y

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
# split train/test
x_train, x_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.3, random_state=0 )
'''Different Regression Models'''
reg=LinearRegression()
model1=reg.fit(x_train,y_train)
model2=GradientBoostingRegressor(random_state=0).fit(x_train, np.ravel(y_train))
model3=RandomForestRegressor(random_state=0).fit(x_train,np.ravel(y_train))

R=model1.score(x_train,y_train)
R2=model2.score(x_train,y_train)
R3=model3.score(x_train,y_train)
B0=reg.intercept_
B1=reg.coef_

print('coefficient of determination R^2:', R, R2, R3)
print('coefficient of determination R^2(estimation):', model1.score(x_test,y_test), model2.score(x_test,y_test), model3.score(x_test,y_test))
print('Bo:', B0)
print('B1:', B1)

y_predicted1=reg.predict(x_test)
mse=mean_squared_error(y_test,y_predicted1)
print('MSE (linear regression): %.3f' % mse)

xtest=np.load('Xtest_Regression_Part2.npy') # loads the independent test set
y_predicted=reg.predict(xtest) # evaluates the predictor
np.save('Ypredict_Regression_Part2.npy',y_predicted) # saves the y^ values into a .npy file

'''
Plotting with scatter
first plot uses the output y as a function of the 1st feature of our data input x.
second plot uses the output y as a function of the input index, i.
third plot is to check outliers, by plotting specific features of x.
'''

'''
plt.scatter(xtrain[:,1],ytrain, color='blue', linewidth=1)
plt.figure()
plt.scatter(range(len(xtest)), y_predicted, color='green', linewidth=1)
plt.figure()
plt.scatter(xtrain[:,1], xtrain[:,2], color='red', linewidth=1)
plt.show()'''
