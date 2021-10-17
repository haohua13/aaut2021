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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline 
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats

# loads the .npy files that cointains the data set provided for the problem
xtrain=np.load('Xtrain_Regression_Part2.npy')
ytrain=np.load('Ytrain_Regression_Part2.npy')

x_train, x_test, y_train, y_test=train_test_split(xtrain, ytrain, test_size=0.25, random_state=1)


# fits the linear model (Linear Regression) Y^=B0+B1*x with the % training set
reg=LinearRegression()
reg.fit(x_train, y_train)

y_predicted_1=reg.predict(x_test) # evaluation of the predictor using the test set splitted from the data set
mse=mean_squared_error(y_test, y_predicted_1) # focuses on larger errors
print('Test set MSE:%.4f' % mse)

'''Outlier Detection and Removal'''

'''Isolation Forest: tree-based outlier detection algorithm'''

forest = IsolationForest(random_state=1) # number of outliers is maximum 10% of the data set
outlier = forest.fit_predict(xtrain) # returns -1 for outliers and 1 for inliers
remove = outlier != -1 # only selects inliers
xtrain, ytrain = xtrain[remove,:], ytrain[remove]
print(xtrain.shape,ytrain.shape) # data set shape with outliers removed


'''EllipticEnvelope: outlier detection for gaussian distributed dataset'''

# ee = EllipticEnvelope(random_state=1, contamination=0.06)
# outlier = ee.fit_predict(xtrain) # fits the model to the data set x and returns the labels 1 for inliers, -1 outliers
# remove = outlier != -1
# xtrain, ytrain = xtrain[remove,:], ytrain[remove]
# print(xtrain.shape,ytrain.shape) # data set shape with outliers removed


'''Unsupervised outlier detection using Local Outlier Factor (LOF)'''

# lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
# outlier = lof.fit_predict(xtrain)
# remove = outlier != -1
# xtrain, ytrain = xtrain[remove,:], ytrain[remove]
# print(xtrain.shape,ytrain.shape) # data set shape with outliers removed

reg.fit(xtrain, ytrain)
y_predicted_1=reg.predict(x_test)
mse=mean_squared_error(y_test, y_predicted_1) # focuses on larger errors
print('MSE after Outlier Removal: %.4f' % mse)

'''Validation of the predictor'''

# For loop to find the best alpha for LASSO Regression
error=5
best_alpha=0
for i in range(1000, 1, -5):
    las=Lasso(alpha=i/10000)
    lasscore=cross_validate(las, xtrain, ytrain, scoring="neg_mean_squared_error", return_estimator=True, cv=10)
    mse_las=abs(np.average(lasscore["test_score"]))
    if mse_las<error:
        error=mse_las
        best_alpha=i/10000
        
# For loop to find the best alpha for Ridge Regression       
error1=5
best_alpha1=0
for i in range(100, 1, -5):
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


'''Load Xtest, fit the Linear Regression model using the whole training set, predict y outcomes and save to a .npy file'''

xtest=np.load('Xtest_Regression_Part2.npy') # loads the independent test set
reg.fit(xtrain,ytrain) # trains the linear model with the given training set
y_predicted=reg.predict(xtest) # evaluation of the predictor using the independent test set (corresponding outcomes for professor)
np.save('Ypredict_Regression_Part2.npy',y_predicted) # saves the predicted y^ values into a .npy file


'''Plotting figures '''

# predicted y^ as a function of the input index i.
plt.scatter(range(len(xtest)), y_predicted, color='green', linewidth=0.1)
plt.xlabel('Index Position of Xtest')
plt.title('Ytest in function of the index of Xtest')
plt.ylabel('Y^ (estimation)')
plt.grid()
plt.figure()

# 1st feature of input x in function of its respective index
plt.scatter(range(len(xtrain)), xtrain[:,1], color='red', linewidth=1)
plt.xlabel('Index Position of Xtrain')
plt.ylabel('Xtrain feature[1]')
plt.title('xtrain[1] in function of its index position')
plt.grid()
plt.figure()