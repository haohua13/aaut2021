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
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import HuberRegressor, TheilSenRegressor, RANSACRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


# loads the .npy files that cointains the data set provided for the problem
xtrain=np.load('Xtrain_Regression_Part2.npy')
ytrain=np.load('Ytrain_Regression_Part2.npy')
'''Outlier Detection and Removal'''

'''Isolation Forest: tree-based outlier detection algorithm'''

# forest = IsolationForest(random_state=1) # number of outliers is maximum 10% of the data set
# outlier = forest.fit_predict(xtrain) # returns -1 for outliers and 1 for inliers
# remove = outlier != -1 # only selects inliers
# xtrain, ytrain = xtrain[remove,:], ytrain[remove]
# print(xtrain.shape,ytrain.shape) # data set shape with outliers removed


'''EllipticEnvelope: outlier detection for gaussian distributed dataset'''

# ee = EllipticEnvelope(random_state=1, contamination=0.08)
# outlier = ee.fit_predict(xtrain) # fits the model to the data set x and returns the labels 1 for inliers, -1 outliers
# remove = outlier != -1
# xtrain, ytrain = xtrain[remove,:], ytrain[remove]
# print(xtrain.shape,ytrain.shape) # data set shape with outliers removed


'''Unsupervised outlier detection using Local Outlier Factor (LOF)'''

# lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# outlier = lof.fit_predict(xtrain)
# remove = outlier != -1
# xtrain, ytrain = xtrain[remove,:], ytrain[remove]
# print(xtrain.shape,ytrain.shape) # data set shape with outliers removed


'''Validation of the predictor'''
# For loop to find the best alpha for LASSO Regression
error=5
best_alpha=0
for i in range(100, 1, -1):
    las=Lasso(alpha=i/1000)
    lasscore=cross_val_score(las, xtrain, np.ravel(ytrain), scoring="neg_mean_squared_error", cv=10)
    mse_las=abs(np.average(lasscore))
    if mse_las<error:
        error=mse_las
        best_alpha=i/1000
        
# For loop to find the best alpha for Ridge Regression       
error1=5
best_alpha1=0
for i in range(100, 1, -1):
    ridge=Ridge(alpha=i/10000)
    ridgescore=cross_val_score(ridge, xtrain, np.ravel(ytrain), scoring="neg_mean_squared_error", cv=10)
    mse_ridge=abs(np.average(ridgescore))
    if mse_las<error1:
        error1=mse_ridge
        best_alpha1=i/10000
        
# Regression models
reg=LinearRegression()
poli=make_pipeline(PolynomialFeatures(2),LinearRegression(fit_intercept=False)) # pipeline so we can see values simultaneously
las=Lasso(alpha=best_alpha)
ridge=Ridge(alpha=best_alpha1)

huber=HuberRegressor()
ransac=RANSACRegressor(random_state=1)
theilsen=TheilSenRegressor(random_state=1)
sgd=SGDRegressor(random_state=1)
gb=GradientBoostingRegressor(random_state=1)
quantile=GradientBoostingRegressor(loss='quantile', alpha=0.4)
random=RandomForestRegressor(random_state=0)



# cross-validation on the training set by applying repeated k-fold method
k = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scoring="neg_mean_squared_error" # mean squared error
linscore=cross_val_score(reg, xtrain, np.ravel(ytrain), scoring=scoring, cv=k)
poliscore=cross_val_score(poli, xtrain, np.ravel(ytrain), scoring=scoring, cv=k)
lasscore=cross_val_score(las, xtrain, np.ravel(ytrain), scoring=scoring, cv=k)
ridgescore=cross_val_score(ridge, xtrain, np.ravel(ytrain), scoring=scoring, cv=k)
huberscore=cross_val_score(huber, xtrain, np.ravel(ytrain), scoring=scoring, cv=k)
ransacscore=cross_val_score(ransac, xtrain, np.ravel(ytrain), scoring=scoring, cv=k)
theilsenscore=cross_val_score(theilsen, xtrain, np.ravel(ytrain), scoring=scoring, cv=k)
quantilescore=cross_val_score(quantile, xtrain, np.ravel(ytrain), scoring=scoring, cv=k)
sgdscore=cross_val_score(sgd, xtrain, np.ravel(ytrain), scoring=scoring, cv=k)
gbscore=cross_val_score(gb, xtrain, np.ravel(ytrain), scoring=scoring, cv=k)
randomscore=cross_val_score(random, xtrain, np.ravel(ytrain), scoring=scoring, cv=k)

# evaluate the models. test_score has the mean_squared_error in each run (k=10)
mse_reg=abs(np.average(linscore))
mse_poli=abs(np.average(poliscore))
mse_las=abs(np.average(lasscore))
mse_ridge=abs(np.average(ridgescore))
mse_huber=abs(np.average(huberscore))
mse_ransac=abs(np.average(ransacscore))
mse_theilsen=abs(np.average(theilsenscore))
mse_sgd=abs(np.average(sgdscore))
mse_gb=abs(np.average(gbscore))
mse_quantile=abs(np.average(quantilescore))   
mse_random=abs(np.average(randomscore))

print('mean MSE Linear:%.4f' % mse_reg)
print('mean MSE Polynomial:%.4f' % mse_poli)
print('mean MSE LASSO:%.4f, with alpha=%.4f' % (mse_las, best_alpha))
print('mean MSE Ridge:%.4f, with alpha=%.4f' % (mse_ridge, best_alpha1))
print('mean MSE Huber:%.4f' % mse_huber)
print('mean MSE RANSAC:%.4f' % mse_ransac)
print('mean MSE Theilsen:%.4f' % mse_theilsen)
print('mean MSE Random Forest:%.4f' % mse_random)
print('mean MSE SGD:%.4f' % mse_sgd)
print('mean MSE GB:%.4f' % mse_gb)
print('mean MSE Quantile:%.4f' % mse_quantile)

'''Load Xtest, fit the model using the whole training set, predict y outcomes and save to a .npy file'''

xtest=np.load('Xtest_Regression_Part2.npy') # loads the independent test set
theilsen.fit(xtrain,np.ravel(ytrain)) # trains the linear model with the given training set
y_predicted=theilsen.predict(xtest) # evaluation of the predictor using the independent test set (corresponding outcomes for professor)
np.save('Ypredict_Regression_Part2.npy',y_predicted) # saves the predicted y^ values into a .npy file

def get_models():
 	models = list()
 	models.append(LinearRegression())
 	models.append(HuberRegressor())
 	models.append(RANSACRegressor())
 	models.append(TheilSenRegressor())
 	return models
 
def plot_fit(x, y, xaxis, model):
	# fit the model with our training set
	model.fit(x, y)
	# calculate outputs
	yaxis = model.predict(xaxis.reshape((len(xaxis), 1)))
	# plot the line of best fit of the model
	plt.plot(xaxis, yaxis, label=type(model).__name__)
    
'''Plotting figures '''
# feature g of input x in function of ytrain. with lines of every robust model evaluated
g=1
a=xtrain[:,g].reshape((-1,1))
b=np.ravel(ytrain)
xaxis = np.arange(a.min(), a.max(), 0.01)
for model in get_models():
 	# plot the line of best fit
 	plot_fit(a, b, xaxis, model)
# plot the dataset
plt.scatter(a, b)
# show the plot
plt.title('Robust Regression Models')
plt.legend()
plt.grid()  
plt.xlabel('specified feature of Xtrain')
plt.ylabel('Ytrain')
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
    