# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 23:24:35 2021

@author: haohu
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import HuberRegressor, TheilSenRegressor, RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from scipy import stats


# loads the .npy files that cointains the data set provided for the problem
xtrain = np.load('./Xtrain_Regression_Part2.npy')
ytrain = np.load('./Ytrain_Regression_Part2.npy')
'''Unsupervised outlier detection using Local Outlier Factor (LOF)'''
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.09)
outlier = lof.fit_predict(ytrain)
remove = outlier != -1
xtrain, ytrain = xtrain[remove,:], ytrain[remove]
print(xtrain.shape,ytrain.shape) # data set shape with outliers removed

# ytrain with outlier removal in function of its respective index
plt.scatter(range(len(ytrain)), ytrain)
plt.title('Ytrain after applying LOF algorithm for Outlier Removal')
plt.xlabel('Index of ytrain')
plt.ylabel('Ytrain after removing 9 outliers')
plt.grid()


'''Feature Selection algorithm on our training set-eliminates insignificant features using Linear Regression'''

temp=np.empty((91,1)) # saves the current feature to test
index_saver=np.empty((20,1)) # saves the indexes of the feature that we removed
reg=LinearRegression() # MSE value from Linear Regression without Feature Selection
k = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
best_mse=abs(np.average(cross_val_score(reg, xtrain, np.ravel(ytrain), scoring="neg_mean_squared_error", cv=k)))
print(best_mse)
n=0
b=0

# Indexes are found on ex2.py
xtrain_3=np.delete(xtrain, 3 , axis=1)
mse1=abs(np.average(cross_val_score(reg, xtrain_3, np.ravel(ytrain), scoring="neg_mean_squared_error", cv=k)))
print(mse1)   
xtrain_7=np.delete(xtrain_3, 7-1 , axis=1)
mse2=abs(np.average(cross_val_score(reg, xtrain_7, np.ravel(ytrain), scoring="neg_mean_squared_error", cv=k)))
print(mse2)
xtrain_9=np.delete(xtrain_7, 9-2 , axis=1)
mse3=abs(np.average(cross_val_score(reg, xtrain_9, np.ravel(ytrain), scoring="neg_mean_squared_error", cv=k)))
print(mse3)
xtrain_10=np.delete(xtrain_9, 10-3 , axis=1)
mse4=abs(np.average(cross_val_score(reg, xtrain_10, np.ravel(ytrain), scoring="neg_mean_squared_error", cv=k)))
print(mse4)
xtrain_12=np.delete(xtrain_10, 12-4 , axis=1)
mse4=abs(np.average(cross_val_score(reg, xtrain_12, np.ravel(ytrain), scoring="neg_mean_squared_error", cv=k)))
print(mse4)

# just to see if it improves more... ( if it removes more features)
for b in range(len(xtrain_12[0])):
    temp=xtrain_12[:,b]
    xtrain_temp=np.delete(xtrain_12, b , axis=1)
    mse=abs(np.average(cross_val_score(reg, xtrain_temp, np.ravel(ytrain), scoring="neg_mean_squared_error", cv=k)))
    if mse<mse4:
        xtrain_12=xtrain_temp
        best_mse=mse
        index_saver[n]=n+b
        n=n+1
    else:
        xtrain_12=np.insert(xtrain_temp, b, temp, axis=1)






