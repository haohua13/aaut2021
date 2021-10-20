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
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# loads the .npy files that cointain the data set provided for the problem
xtrain=np.load('Xtrain_Regression_Part1.npy')
ytrain=np.load('Ytrain_Regression_Part1.npy')

'''Validation of the predictor'''

# simple split into train and test sets (80% and 20%)
x_train, x_test, y_train, y_test=train_test_split(xtrain, ytrain, test_size=0.2, random_state=0)

# For loop to find the best alpha for LASSO Regression
error=5
best_alpha=0
for i in range(1000, 1, -1):
    las=Lasso(alpha=i/100000)
    lasscore=cross_validate(las, xtrain, ytrain, scoring="neg_mean_squared_error", return_estimator=True, cv=10)
    mse_las=abs(np.average(lasscore["test_score"]))
    if mse_las<error:
        error=mse_las
        best_alpha=i/100000
        
# For loop to find the best alpha for Ridge Regression       
error1=5
best_alpha1=0
for i in range(100, 1, -1):
    ridge=Ridge(alpha=i/10000)
    ridgescore=cross_validate(ridge, xtrain, ytrain, scoring="neg_mean_squared_error", return_estimator=True, cv=10)
    mse_ridge=abs(np.average(ridgescore["test_score"]))
    if mse_las<error1:
        error1=mse_ridge
        best_alpha1=i/10000
        
# regression models: linear, polynomial, LASSO and Ridge
reg=LinearRegression()
poli=make_pipeline(PolynomialFeatures(2),LinearRegression(fit_intercept=False)) # pipeline so we can see values simultaneously
las=Lasso(alpha=best_alpha)
ridge=Ridge(alpha=best_alpha1)

# cross-validation on the training set by applying k-fold method, cv=k
scoring="neg_mean_squared_error" # mean squared error
linscore=cross_validate(reg, xtrain, ytrain, scoring=scoring, return_estimator=True, cv=10)
poliscore=cross_validate(poli, xtrain, ytrain, scoring=scoring, return_estimator=True, cv=10)
lasscore=cross_validate(las, xtrain, ytrain, scoring=scoring, return_estimator=True, cv=10)
ridgescore=cross_validate(ridge, xtrain, ytrain, scoring=scoring, return_estimator=True, cv=10)

# evaluate the models. test_score has the mean_squared_error in each run (k=10)
mse_reg=abs(np.average(linscore["test_score"]))
mse_poli=abs(np.average(poliscore["test_score"]))
mse_las=abs(np.average(lasscore["test_score"]))
mse_ridge=abs(np.average(ridgescore["test_score"]))
print('mean MSE Linear:%.4f' % mse_reg)
print('mean MSE Polynomial:%.4f' % mse_poli)
print('mean MSE LASSO:%.4f, with alpha=%.4f' % (mse_las, best_alpha))
print('mean MSE Ridge:%.4f, with alpha=%.4f' % (mse_ridge, best_alpha1))
print('maximum MSE from Linear:%.4f' % abs(np.min(linscore["test_score"])))
print('maximum MSE from Polynomial:%.4f' % abs(np.min(poliscore["test_score"])))
print('maximum MSE from LASSO:%.4f' % abs(np.min(lasscore["test_score"])))
print('maximum MSE from Ridge:%.4f' % abs(np.min(ridgescore["test_score"])))

# fits the linear model (Linear Regression) Y^=B0+B1*x with the 80% training set
reg.fit(x_train, y_train)

y_predicted_1=las.predict(x_test) # evaluation of the predictor using the test set splitted from the data set
mse=mean_squared_error(y_test, y_predicted_1) # focuses on larger errors
R=reg.score(x_train,y_train) # coefficient of determination 
B0=reg.intercept_ # slope of the linear model
B1=reg.coef_ # vector with the B1 corresponding to each feature of x

'''Load Xtest, fit the LASSO Regression model using the whole training set, predict y outcomes and save to a .npy file'''

xtest=np.load('Xtest_Regression_Part1.npy') # loads the independent test set
las.fit(xtrain,ytrain) # trains the linear model with the given training set
y_predicted=reg.predict(xtest) # evaluation of the predictor using the independent test set (corresponding outcomes for professor)
np.save('Ypredict_Regression_Part1.npy',y_predicted) # saves the predicted y^ values into a .npy file


'''Plotting figures '''

# trainy as a function of the 1st feature of our data input x, with its estimated linear regression
plt.scatter(xtrain[:,1],ytrain, color='tab:red')
plt.xlabel('Xtrain feature[1]')
plt.ylabel('Ytrain')
plt.grid()
plt.title('Ytrain in function of Xtrain[1]')
x=np.linspace(min(xtrain[:,1])-1, 1+max(xtrain[:,1]), 100)
y=B0*x+B1[:,1]
plt.plot(x, y, 'tab:blue', label='y=B0+B1*x', linewidth=3)
plt.legend(loc='lower left')
plt.figure()

# predicted y^ as a function of the input index i.
plt.scatter(range(len(xtest)), y_predicted, color='tab:blue',s=10)
plt.xlabel('Index Position of Xtest')
plt.title('Ytest in function of the index of Xtest')
plt.ylabel('Y^ (estimation)')
plt.grid()
plt.figure()

# 1st feature of input x in function of its respective index
plt.scatter(range(len(xtrain)), xtrain[:,1], color='tab:green')
plt.xlabel('Index Position of Xtrain')
plt.ylabel('Xtrain feature[1]')
plt.title('xtrain[1] in function of its index position')
plt.grid()
plt.figure()

