#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 14:45:59 2018

@author: logancross
"""

print(__doc__)

import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# this first import is only required to run the example a part of the test suite
from mvpa2 import cfg
from mvpa2.clfs.skl.base import SKLLearnerAdapter
from mvpa2.datasets import dataset_wizard
ds_train=dataset_wizard(samples=X, targets=y)

clf_1 = SKLLearnerAdapter(DecisionTreeRegressor(max_depth=2))
clf_2 = SKLLearnerAdapter(DecisionTreeRegressor(max_depth=5))

clf_1.train(ds_train)
clf_2.train(ds_train)

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = clf_1.predict(X_test)
y_2 = clf_2.predict(X_test)

# plot the results
# which clearly show the overfitting for the second depth setting
import pylab as pl

pl.figure()
pl.scatter(X, y, c="k", label="data")
pl.plot(X_test, y_1, c="g", label="max_depth=2", linewidth=2)
#pl.plot(X_test, y_2, c="r", label="max_depth=5", linewidth=2)
pl.xlabel("data")
pl.ylabel("target")
pl.title("Decision Tree Regression")
pl.legend()

from sklearn.metrics import r2_score

y_all = clf_1.predict(X)
print r2_score(y_all, y)




from sklearn.datasets import make_regression
from matplotlib import pyplot
# generate regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
# plot regression dataset
pyplot.scatter(X,y)
pyplot.show()

from sklearn.datasets import make_regression

n_samples = 1000
n_outliers = 200


X, y, coef = make_regression(n_samples=n_samples, n_features=1,
                                      n_informative=1, noise=10,
                                      coef=True, random_state=0)

# Add outlier data
np.random.seed(0)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

X, y = make_regression(n_samples=100, n_features=50,  n_informative=2, noise=0.9)
groups = np.hstack([[ii] * 10 for ii in range(100)])
#groups = np.arange(100)

ds_train=dataset_wizard(samples=X, targets=y, chunks=groups)

ridge = RidgeReg()
#ridge.train(ds_train)
#preds = ridge.predict(ds_train)
#r2_score(preds, y)

cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks'), errorfx=correlation)
cv_results = cv(ds_train)
print np.mean(cv_results)


from sklearn.model_selection import cross_val_score
from sklearn import linear_model

skridge = linear_model.Ridge(alpha=0.1)
np.mean(cross_val_score(skridge,X,y,cv=10))
