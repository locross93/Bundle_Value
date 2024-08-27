#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 13:59:50 2019

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
from os import listdir

subj = 101

onsets_folder = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model001/onsets/'

dir_onsets = listdir(onsets_folder)
if dir_onsets[0] == '.DS_Store':
    dir_onsets.remove('.DS_Store')

value_list = []
chunks_list = []
run_num = 0
for run in dir_onsets:
    temp_folder = onsets_folder+run
    cond001_onsets = np.genfromtxt(temp_folder+'/cond001.txt')
    cond002_onsets = np.genfromtxt(temp_folder+'/cond002.txt')
    timing = np.concatenate((cond001_onsets[:,0], cond002_onsets[:,0]))
    sort_time_inds = np.argsort(timing)
    value = np.concatenate((cond001_onsets[:,2], cond002_onsets[:,2]))
    value = value[sort_time_inds]
    value_list.append(value)
    chunks = run_num*np.ones([len(value)])
    chunks_list.append(chunks)
    run_num+=1
    
value_allruns = np.asarray([item for sublist in value_list for item in sublist]).astype(int)
chunks_allruns = np.asarray([item for sublist in chunks_list for item in sublist]).astype(int)  
    

glm_ds_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial/test_4D_zip.nii.gz'
mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'

fds = fmri_dataset(samples=glm_ds_file, targets=value_allruns, chunks=chunks_allruns, mask=mask_name)

fds_sub = fds[:,:10000]

ridge = RidgeReg()
cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks'), errorfx=correlation)
cv_results = cv(fds_sub)

print np.mean(cv_results)


X = fds.samples
y = fds.targets
cv_groups = fds.chunks

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer

def get_correlation(y, y_pred):
    correlation = pearsonr(y, y_pred)[0]
    
    return correlation

selectk = SelectKBest(f_regression, k=500)
X_fs = selectk.fit_transform(X, y) 
best_feats = selectk.get_support(indices=True)

alp=5
skridge = linear_model.Ridge(alpha=10**alp)
#skridge = linear_model.Ridge(alpha=25)
r_scorer = make_scorer(get_correlation)
np.mean(cross_val_score(skridge,X_fs,y,groups=cv_groups, scoring=r_scorer, cv=15))

from sklearn.pipeline import Pipeline

fs_ridge = Pipeline([('select_k', selectk), ('ridge', skridge)])

np.mean(cross_val_score(fs_ridge,X,y,groups=cv_groups, scoring=r_scorer, cv=15))


fs_fds = fds[:,best_feats]

ridge = RidgeReg()
cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks'), errorfx=correlation)
cv_results = cv(fs_fds)

print np.mean(cv_results)


#searchlight
# enable debug output for searchlight call
if __debug__:
    debug.active += ["SLC"]
    
sweep_params = np.arange(3,8,1)
scores = []
for c in sweep_params:
    alp=c
    ridge = RidgeReg(lm=10**alp)
    cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks'), errorfx=correlation)
    
    ridge_sl = sphere_searchlight(cv, radius=4, space='voxel_indices',
                                 postproc=mean_sample())
    
    start_time = time.time()
    print 'starting searchlight',time.time() - start_time
    res_sl = ridge_sl(fs_fds)
    print 'finished searchlight',time.time() - start_time
    
    scores.append(np.mean(res_sl))
    
#for sweeping
plt.plot(sweep_params,scores,'ro-', label="Crossval Fit")
plt.xlabel('Regularization Alpha')
plt.ylabel('Acc')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#alpha = 3 seems to be good
ridge = RidgeReg(lm=10**3)
cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks'), errorfx=correlation)
ridge_sl = sphere_searchlight(cv, radius=4, space='voxel_indices',
                                 postproc=mean_sample())
print 'starting searchlight',time.time() - start_time
res_sl = ridge_sl(fds)
print 'finished searchlight',time.time() - start_time


#Lasso PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=500)
X_pca = pca.fit_transform(X)

alp=7
sk_lasso = linear_model.Ridge(alpha=10**alp)

#skridge = linear_model.Ridge(alpha=25)
r_scorer = make_scorer(get_correlation)
np.mean(cross_val_score(sk_lasso,X_pca,y,groups=cv_groups, scoring=r_scorer, cv=15))

ridge_cv = linear_model.RidgeCV(alphas=[1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9], cv=15)
model = ridge_cv.fit(X_pca,y)

#enet_cv = linear_model.ElasticNetCV(l1_ratio=[.1, .2, .3, .5, .7, .9, 1], alphas=[1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9], cv=15)
enet_cv = linear_model.ElasticNetCV(l1_ratio=0.01, alphas=[1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9], cv=15)
model = enet_cv.fit(X_pca,y)

enet = linear_model.ElasticNet(alpha=1e2, l1_ratio=0.2)
np.mean(cross_val_score(enet,X_pca,y,groups=cv_groups, scoring=r_scorer, cv=15))
model = enet.fit(X_pca,y)

ridge = linear_model.Ridge(alpha=10**7)
np.mean(cross_val_score(ridge,X_pca,y,groups=cv_groups, scoring=r_scorer, cv=15))
model = ridge.fit(X_pca,y)

#Grid search 
from sklearn.model_selection import GridSearchCV

tuned_parameters = [{'alpha': [1, 25, 50, 75, 100],
                     'l1_ratio': [0.15, 0.2, 0.25]}]
    

enet_grid = GridSearchCV(linear_model.ElasticNet(), tuned_parameters, cv=15,
                       scoring=r_scorer)
enet_grid.fit(X_pca, y)


#project back to voxel space
coefs = model.coef_
coefs2voxs = pca.inverse_transform(coefs)
abs_coefs2voxs = np.abs(coefs2voxs)

# reverse map scores back into nifti format
nimg = map2nifti(fds, coefs2voxs)
nii_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/test_pca_ridge.nii.gz'
nimg.to_filename(nii_file)

nimg = map2nifti(fds, abs_coefs2voxs)
nii_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/test_pca_ridge_abs.nii.gz'
nimg.to_filename(nii_file)
