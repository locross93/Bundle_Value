# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:54:36 2021

@author: locro
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
from mvpa2.datasets.mri import map2nifti
import mvpa_utils_lab
import numpy as np

bundle_path = '/state/partition1/home/lcross/Bundle_Value/'

subj = '110'

analysis_list = [['105','decode_choice'],['107','decode_choice'],['108','decode_choice']
                 ,['109','decode_choice'],['110','decode_choice']]

#Save?
save = True
save_nifti=True

delete_parts=False

parts = 5

for analysis in analysis_list:
    subj = analysis[0]
    analysis_prefix = analysis[1]

    analysis_path = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'
    
    #which ds to use and which mask to use
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii'
    mask_name = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle'] 
    fds = mvpa_utils_lab.make_targets_choice(subj, glm_ds_file, mask_name, conditions, system='labrador')

    scores_per_voxel = np.array([])
    for i in range(parts):
        i = i + 1
        temp_array = h5load(analysis_path+analysis_prefix+'_part'+str(i))
        scores_per_voxel = np.append(scores_per_voxel, temp_array)
    #get rid of NaNs
    nan_inds = np.where(np.isnan(scores_per_voxel))[0]
    scores_per_voxel[nan_inds] = 0
    
    assert scores_per_voxel.shape[0] == fds.shape[1]
        
    if save:
        vector_file = analysis_path+analysis_prefix
        h5save(vector_file,scores_per_voxel)
        if save_nifti:
            nimg = map2nifti(fds, scores_per_voxel)
            nii_file = vector_file+'.nii.gz'
            nimg.to_filename(nii_file)
            
    if delete_parts:
        for i in range(parts):
            i = i + 1
            os.remove(analysis_path+analysis_prefix+'_part'+str(i))