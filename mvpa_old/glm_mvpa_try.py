#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:48:54 2018

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt

start_time = time.time()

subj = 102
task = 1
run = 1
model = 1

fds_file = '/Users/logancross/Documents/Bundle_Value/mvpa/datasets/sub'+str(subj)+'/raw_voxels_pfc.hdf5'

print 'loading now',time.time() - start_time

fds = h5load(fds_file)

print 'loading done',time.time() - start_time

events = find_events(targets=fds.sa.targets, chunks=fds.sa.chunks)
events = [ev for ev in events if ev['targets'] in ['item', 'bundle']]

# temporal distance between samples/volume is the volume repetition time
TR = np.median(np.diff(fds.sa.time_coords))
for ev in events:
    ev['onset'] = ev['onset'] * TR
    ev['duration'] = ev['duration'] * TR
    
mask_name='/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/ofc_mask.nii.gz'

#load mask and create a mapper for it
brain_mask ='/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
masked = fmri_dataset(mask_name, mask=brain_mask)
reshape_masked=masked.samples.reshape(fds.shape[1])
reshape_masked=reshape_masked.astype(bool)
mask_map = mask_mapper(mask=reshape_masked)

#map the dataset with the mask
print 'masking now',time.time() - start_time

fds_mask = mask_map(fds)

print 'masking done',time.time() - start_time
    
fds_subset = fds_mask

print 'fitting model',time.time() - start_time
evds = fit_event_hrf_model(fds_subset, events, time_attr='time_coords', condition_attr=('targets', 'chunks'))
print 'model finished',time.time() - start_time

zscore(evds, chunks_attr=None)

clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
cv = CrossValidation(clf, NFoldPartitioner(attr='chunks'))
cv_glm = cv(evds)
print '%.2f' % np.mean(cv_glm)

ridge_log = PLR()
cv = CrossValidation(ridge_log, NFoldPartitioner(attr='chunks'))
cv_glm = cv(evds)
print '%.2f' % np.mean(cv_glm)

X = evds.samples
y = evds.targets
num_features = X.shape[1]
num_runs = 15

#remove features with no variance
var_feats = np.var(X, axis=0)
yes_var = np.where(var_feats != 0)[0]

X = X[:,yes_var]

from sklearn.feature_selection import SelectKBest, f_classif
fs = SelectKBest(f_classif, k=200)
fs.fit(X, y)
best_inds = fs.get_support(indices=True)
X = X[:,best_inds]

#define groups/chunks for cross_validation
cv_groups = evds.chunks

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sweep_params = np.arange(0,5,1)
scores = []
for c in sweep_params:
    c = math.pow(10,c)
    ridge_log = LogisticRegression(penalty='l2', C=c*num_features)
    temp = np.mean(cross_val_score(ridge_log,X,y,groups=cv_groups,cv=num_runs))
    print temp
    scores.append(temp)

#for sweeping
plt.plot(np.arange(1,6,1),scores,'ro-', label="Crossval Fit")
plt.xlabel('Regularization Alpha')
plt.ylabel('Class Acc')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


evds_fs = evds[:,best_inds]
ridge_log = PLR()
cv = CrossValidation(ridge_log, NFoldPartitioner(attr='chunks'))
cv_glm = cv(evds_fs)
print '%.2f' % (1 - np.mean(cv_glm))

clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
cv = CrossValidation(clf, NFoldPartitioner(attr='chunks'))
cv_glm = cv(evds_fs)
print '%.2f' % (1 - np.mean(cv_glm))