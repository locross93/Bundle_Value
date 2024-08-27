#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 13:14:54 2018

@author: logancross
"""

num_voxs = evds.shape[1]
rand_inds = np.random.randint(0, num_voxs, size=200)

sweep_params = np.arange(0,5,1)
sweep_params = np.arange(4,9,1)
scores = []
for c in sweep_params:
    lm = math.pow(10,c)
    ridge = RidgeReg(lm=100)
    cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks'), errorfx=correlation)
    sl = sphere_searchlight(cv, radius=c, space='voxel_indices', center_ids=rand_inds,
                             postproc=mean_sample())
    rand_s1 = sl(evds)
    scores.append(np.mean(rand_s1))
    
#for sweeping
plt.plot(sweep_params,scores,'ro-', label="Crossval Fit")
plt.xlabel('Regularization Alpha')
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

X = evds.samples
y = evds.targets
cv_groups = evds.chunks

from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from sklearn import linear_model

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

alp = math.pow(10,6)
sk_ridge = linear_model.Ridge(alpha=alp)
sk_lasso = linear_model.Lasso(alpha=alp)


def get_correlation(y, y_pred):
    correlation = pearsonr(y, y_pred)[0]
    
    return correlation

r_scorer = make_scorer(get_correlation)
np.mean(cross_val_score(sk_ridge,X_pca,y,groups=cv_groups,scoring=r_scorer,cv=11))

np.mean(cross_val_score(sk_lasso,X_pca,y,groups=cv_groups,scoring=r_scorer,cv=11))


X_fds = fds_mask.samples
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_fds)
