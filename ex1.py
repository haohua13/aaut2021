# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:23:08 2021

@author: haohua
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
reg=LinearRegression()

xtrain=np.load('Xtrain_Regression_Part1.npy')
ytrain=np.load('Ytrain_Regression_Part1.npy')

# split into train and test sets (70% and 30%)
x_train, x_test, y_train, y_test=train_test_split(xtrain, ytrain, test_size=0.3, random_state=1)

reg.fit(xtrain,ytrain) # fits the model with the given training set

xtest=np.load('Xtest_Regression_Part1.npy') # loads the independent test set
y_predicted=reg.predict(xtest) # evaluation of the predictor using the test set 
y_predicted_1=reg.predict(x_test) # evaluation of the predictor using the test set splitted from the data set


mae=mean_absolute_error(y_test, y_predicted_1) # average error
mse=mean_squared_error(y_test, y_predicted_1) # focuses on larger errors
R=reg.score(xtrain,ytrain)
B0=reg.intercept_
B1=reg.coef_

print('coefficient of determination R^2:', R)
print('coefficient of determination R^2 (estimation):', reg.score(x_test,y_test))
print('Bo:', B0)
print('B1:', B1)
print('MAE:%.3f' % mae)
print('MSE:%.3f' % mse)


np.save('Ypredict_Regression_Part1.npy',y_predicted) # saves the predicted y^ values into a .npy file

'''Plotting with scatter. 
first plot uses the output y as a function of the 1st feature of our data input x.
third plot uses the output y as a function of the input index, i. training
third plot uses the output y as a function of the input index, i. prediction
'''

plt.scatter(xtrain[:,1],ytrain, color='blue', linewidth=1)
plt.xlabel('Xtrain feature[1]')
plt.ylabel('Ytrain')
plt.figure()
plt.scatter(range(len(xtrain)),ytrain, color='red', linewidth=1)
plt.figure()
plt.xlabel('Index Position')
plt.ylabel('Xtest')
plt.scatter(range(len(xtest)), y_predicted, color='green', linewidth=1)
plt.xlabel('Index Position')
plt.ylabel('Y^ (estimation)')
plt.show()
