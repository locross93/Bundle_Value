#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:19:43 2018

@author: logancross
"""
from os import listdir

onsets_folder = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub102/model/model001/onsets/'

dir_onsets = listdir(onsets_folder)
dir_onsets.remove('.DS_Store')

value_list = []
for run in dir_onsets:
    temp_folder = onsets_folder+run
    cond001_onsets = np.genfromtxt(temp_folder+'/cond001.txt')
    cond002_onsets = np.genfromtxt(temp_folder+'/cond002.txt')
    timing = np.concatenate((cond001_onsets[:,0], cond002_onsets[:,0]))
    sort_time_inds = np.argsort(timing)
    value = np.concatenate((cond001_onsets[:,2], cond002_onsets[:,2]))
    value = value[sort_time_inds]
    value_list.append(value)
    
value_allruns = np.asarray([item for sublist in value_list for item in sublist])

evds_reg = evds[:,:1000]
evds_reg.targets = value_allruns

ridge = RidgeReg()
cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks'), errorfx=correlation)
cv_results = cv(evds_reg)
    
fsel = SensitivityBasedFeatureSelection(
           CorrCoef(),
           FixedNElementTailSelector(500, mode='select', tail='upper'))
fsel.train(evds)
fs_evds = fsel(evds)

cv_results = cv(fs_evds)
print np.mean(cv_results)


print 'starting svr',time.time() - start_time
cv_svr_results = cv_svr(fs_evds)
print 'finished svr',time.time() - start_time

#searchlight
# enable debug output for searchlight call
if __debug__:
    debug.active += ["SLC"]

sl = sphere_searchlight(cv, radius=4, space='voxel_indices',
                             postproc=mean_sample())

start_time = time.time()
print 'starting searchlight',time.time() - start_time
sl_map = sl(evds)
print 'finished searchlight',time.time() - start_time



X = evds.samples
y = evds.targets
cv_groups = evds.chunks

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

alp=3
skridge = linear_model.Ridge(alpha=10**alp)
#skridge = linear_model.Ridge(alpha=25)
r_scorer = make_scorer(get_correlation)
np.mean(cross_val_score(skridge,X_fs,y,groups=cv_groups, scoring=r_scorer, cv=15))

fs_evds = evds[:,best_feats]
cv_results = cv(fs_evds)
print np.mean(cv_results)

#initial for loop for sweeping only
scores = np.zeros(5)
alps = np.arange(-10,-5,1)
alp_count = -1
for alp in alps:
    alp = math.pow(10,alp)
    alp_count+=1
    skridge = linear_model.Ridge(alpha=alp)
    scores[alp_count] = np.mean(cross_val_score(skridge,X_fs,y,groups=cv_groups, scoring=r_scorer, cv=15))
    
#for sweeping
plt.plot(alps,scores,'ro-', label="Crossval Fit")
plt.xlabel('Regularization Alpha')
plt.ylabel('R Squared')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



evds = h5load('/Users/logancross/Documents/Bundle_Value/mvpa/datasets/sub'+str(subj)+'/glm_ds_pfc.hdf5')

sl = sphere_searchlight(cv, radius=4, space='voxel_indices', center_ids=best_voxs,
                             postproc=mean_sample())


svr = SVM(svm_impl='EPSILON_SVR')
cv_svr = CrossValidation(svr, NFoldPartitioner(attr='chunks'), errorfx=correlation)
svr_sl = sphere_searchlight(cv_svr, radius=4, space='voxel_indices', center_ids=best_voxs,
                             postproc=mean_sample())