#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:41:15 2018

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

#glm over every trial
events = find_events(targets=fds.sa.targets, chunks=fds.sa.chunks)
events = [ev for ev in events if ev['targets'] in ['item', 'bundle']]

# temporal distance between samples/volume is the volume repetition time
TR = np.median(np.diff(fds.sa.time_coords))
# convert onsets and durations into timestamps
trial_num = 0
for ev in events:
    ev['onset'] = ev['onset'] * TR
    ev['duration'] = ev['duration'] * TR
    ev['trial_num'] = trial_num
    trial_num+=1
    
fds_mask.sa['trial_num'] = np.zeros(len(fds))
trial_num = 0
in_trial = False
for i in range(len(fds)):
    if fds_mask.targets[i] == 'item' or fds_mask.targets[i] == 'bundle':
        in_trial = True
    elif in_trial:
        #you were in trial in last i but now out
        trial_num+=1
        in_trial = False
        
    fds_mask.sa.trial_num[i] = trial_num
    
print 'fitting model',time.time() - start_time
evds = fit_event_hrf_model(fds_mask, events, time_attr='time_coords', condition_attr=('targets', 'trial_num'))
print 'model finished',time.time() - start_time

evds.sa['chunks'] = np.zeros(len(evds)).astype(int)
for ev in events:
    trial_num = ev['trial_num']
    trial_ind = np.where(evds.sa.trial_num == trial_num)[0]
    evds.sa.chunks[trial_ind] = ev['chunks']
    
start_time = time.time() 
print 'training model',time.time() - start_time

clf = LinearCSVMC()
clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')

balancer = ChainNode([NFoldPartitioner(attr='chunks'),Balancer(attr='targets',count=1,limit='partitions',apply_selection=True)],space='partitions')

cvte = CrossValidation(clf, balancer)
                       #enable_ca=['stats'])

cv_results = cvte(evds)

print 'model finished',time.time() - start_time

print np.mean(cv_results)

#searchlight

balancer = ChainNode([NFoldPartitioner(attr='chunks'),Balancer(attr='targets',count=1,limit='partitions',apply_selection=True)],space='partitions')

#cvte = CrossValidation(clf, balancer)

#sl = sphere_searchlight(cvte, radius=3, postproc=mean_sample())

sl = GNBSearchlight(GNB(), balancer, IndexQueryEngine(voxel_indices=Sphere(radius=3)), postproc=mean_sample())

res = sl(evds)

res_acc = res
res_acc.samples = 1 - res.samples

# reverse mapping does not work with mask, so create a mapper through slicing
mask_set = set(tuple(x) for x in fds_mask.fa.voxel_indices)
masked_ind = [c for c in range(len(fds.fa.voxel_indices)) if tuple(fds.fa.voxel_indices[c,:]) in mask_set]
fds_mask_mapped = fds[:,masked_ind]

scores_per_voxel = np.zeros(1,fds.shape[1])
scores_per_voxel[masked_ind] = res_acc.samples

# reverse map scores back into nifti format
nimg = map2nifti(fds_mask_mapped, res_acc)
nii_file = '/Users/logancross/Documents/Bundle_Value/mvpa/gnb_test.nii.gz'
nimg.to_filename(nii_file)
