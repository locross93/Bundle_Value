#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:41:26 2021

@author: logancross
"""

#prevent multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
sys.path.insert(0, '/state/partition1/home/lcross/Bundle_Value/mvpa')
os.chdir('/state/partition1/home/lcross/Bundle_Value/mvpa')

#from mvpa2.suite import *
from mvpa2.base.hdf5 import h5load, h5save
from mvpa2.misc.neighborhood import Sphere
import mvpa_utils_lab
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from sklearn import linear_model
from sklearn.model_selection import GroupKFold
import random
import time
import numpy as np
import scipy

def get_correlation(y, y_pred):
    correlation = pearsonr(y, y_pred)[0]
    
    return correlation

def get_voxel_sphere(center_coords, voxel_indices):
    radius = 4
    sphere = Sphere(radius)
    all_coords = sphere(center_coords)
    inds2use = []
    for coords in all_coords:
        coords = np.array(coords)
        temp_ind = np.where((voxel_indices == coords).all(axis=1))[0]
        if len(temp_ind) > 0:
            assert len(temp_ind) == 1
            inds2use.append(temp_ind[0])
            
    return inds2use

###SCRIPT ARGUMENTS
start_time = time.time()

bundle_path = '/state/partition1/home/lcross/Bundle_Value/'

#Break up script by parts or no?
split = True
part=0
if split:
    part = int(sys.argv[1])
    num_parts = int(sys.argv[2])

subj = str(sys.argv[3])

analysis_name = 'decode_choice'

relative_value = True

save = True

subsample = True

#None or a number 
save_incomplete = 2000

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle'] 

#which ds to use and which mask to use
glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii'
#mask_name = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
mask_name = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'

#make targets with mvpa utils    
fds = mvpa_utils_lab.make_targets_choice(subj, glm_ds_file, mask_name, conditions, system='labrador')
    
#balance dataset, subsample from bigger class
choice = fds.targets
ref_choices = np.where(choice == 0)[0]
item_choices = np.where(choice == 1)[0]
if ref_choices.shape[0] > item_choices.shape[0]:
    num_examples = item_choices.shape[0]
    ref_choices_sub = np.sort(np.random.choice(ref_choices, num_examples, replace=False))
    inds2use = np.sort(np.concatenate((ref_choices_sub, item_choices)))
    fds = fds[inds2use,:]
elif item_choices.shape[0] > ref_choices.shape[0]:
    num_examples = ref_choices.shape[0]
    item_choices_sub = np.sort(np.random.choice(item_choices, num_examples, replace=False))
    inds2use = np.sort(np.concatenate((ref_choices, item_choices_sub)))
    fds = fds[inds2use,:]
    
#Splice brain into parts depending on argument
if split:
    split_size = int(fds.shape[1]/num_parts)
    remainder = fds.shape[1]%num_parts
    splice = np.arange(0,fds.shape[1],split_size) 
    splice[-1] = splice[-1] + (remainder)
    if remainder == 0 and part == num_parts:
        splice = np.arange(0,fds.shape[1]+1,split_size) 
        splice[-1] = splice[-1] + (remainder)
    fds = fds[:,splice[part-1]:splice[part]]

#define model
num_vox = fds.shape[1]

y = fds.targets
cv_groups = fds.chunks
voxel_indices = fds.fa.voxel_indices

scores_per_voxel = np.zeros([num_vox])

voxs2run = np.arange(num_vox)
prev_time = time.time()
for vox in voxs2run:
    vox_coord = voxel_indices[vox,:]
    sphere_inds = get_voxel_sphere(vox_coord, voxel_indices)
    fds_sphere = fds[:,sphere_inds]
    
    scores_per_voxel[vox] = mvpa_utils_lab.roiClass_1Ss(fds_sphere, mask_name)
    
    vox_ind = np.where(vox == voxs2run)[0][0]
    
    current_time = time.time()
    time_dif_s = current_time - start_time
    time_dif_hrs = round((current_time - start_time)/3600,2)
    
     #how long do we have left
    if vox_ind%100==0:
        log_flag=True
        print('Time elapsed: ',time_dif_hrs,' hrs')
        
        #estimate analysis rate per voxel every 100 voxs
        time10vox = current_time - prev_time
        time_per_vox = time10vox/50
        remaining_vox = voxs2run.shape[0] - vox_ind
        time_left = round((remaining_vox*time_per_vox)/3600,2)
        print('Estimated time left: ',time_left,' hrs')
        prev_time = current_time
    
    #save array every X voxels, delete temp folder contents when done
    if save_incomplete and vox_ind > 0 and vox_ind%save_incomplete==0:
        temp_folder = bundle_path+'mvpa/analyses/sub'+str(subj)+'/temp/'
        if not os.path.isdir(temp_folder):
            os.makedirs(temp_folder)
        vector_file = temp_folder+analysis_name+'_part'+str(part)+'_vox'+str(vox_ind)
        np.save(vector_file,scores_per_voxel)
        
sl_time = time.time() - start_time
print('finished searchlight',sl_time)

comp_speed = sl_time/fds.shape[1]
print('Analyzed at a speed of ',comp_speed,'  per voxel')

#save
vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_part'+str(part)
h5save(vector_file,scores_per_voxel)