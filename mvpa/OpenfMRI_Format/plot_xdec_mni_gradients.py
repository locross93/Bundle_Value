#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:46:03 2021

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/Users/logancross/Documents/Bundle_Value/mvpa/")
import os
os.chdir("/Users/logancross/Documents/Bundle_Value/mvpa/")
import mvpa_utils 
import seaborn as sns

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj_list = ['104','105','107','108','109','110','111','113','114']
subj_list = ['104','107','108','109','110','111','113','114']

plot = False

for subj in subj_list:
    analysis_name = 'cross_decoding_rel_value'

    s2s_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_s2s'
    s2b_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_s2b'
    b2b_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_b2b'
    b2s_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_b2s'
    
    scores_s2s = h5load(s2s_file)
    scores_s2b = h5load(s2b_file)
    scores_b2b = h5load(b2b_file)
    scores_b2s = h5load(b2s_file)
    
    #make fds to get shape
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle'] 
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=False)
    
    voxel_inds = fds.fa.voxel_indices
    
    #get pfc mask inds
    mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    masked = fmri_dataset(mask_name, mask=brain_mask)
    reshape_masked=masked.samples.reshape(fds.shape[1])
    reshape_masked=reshape_masked.astype(bool)
    mask_map = mask_mapper(mask=reshape_masked)
    mask_slice = mask_map[1].slicearg
    mask_inds = np.where(mask_slice == 1)[0]
    
    print subj
    
    #difference between bundle score and single item score
    #bundle_pref = scores_b2b[mask_inds] - scores_b2s[mask_inds]
    bundle_pref = scores_s2s[mask_inds] - scores_s2b[mask_inds]
    
    #take z inds for dorsal to ventral gradient
    z_inds = voxel_inds[mask_inds,2]
    corr_z = pearsonr(bundle_pref, z_inds)[0]
    print 'ventral/dorsal:',round(corr_z,3)
    
    if plot:
        ax = plt.axes()
        plt.scatter(z_inds, bundle_pref)
        plt.text(0.8,0.15,'Correlation: '+'%.3f'%corr_z, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes)
        plt.xlabel('Z Index')
        plt.ylabel('Bundle Score - Single Item Score')
        plt.title('Dorsal/Ventral Gradient Sub'+subj, fontsize=18)
        plt.show()
    
    #take y inds too for posterior - anterior
    y_inds = voxel_inds[mask_inds,1]
    corr_y = pearsonr(bundle_pref, y_inds)[0]
    print 'posterior/anterior:',round(corr_y,3)
    
    if plot:
        ax = plt.axes()
        plt.scatter(y_inds, bundle_pref)
        plt.text(0.8,0.15,'Correlation: '+'%.3f'%corr_y, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes)
        plt.xlabel('Y Index')
        plt.ylabel('Bundle Score - Single Item Score')
        plt.title('Posterior/Anterior Gradient Sub'+subj, fontsize=18)
        plt.show()
    
    #take x inds and center at x=36 for medial - lateral
    x_inds = voxel_inds[mask_inds,0]
    laterality = np.absolute(x_inds - 36)
    corr_lat = pearsonr(bundle_pref, laterality)[0]
    print 'medial/lateral:',round(corr_lat,3)
    
    if plot:
        ax = plt.axes()
        plt.scatter(y_inds, bundle_pref)
        plt.text(0.8,0.15,'Correlation: '+'%.3f'%corr_lat, horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes)
        plt.xlabel('Laterality')
        plt.ylabel('Bundle Score - Single Item Score')
        plt.title('Medial/Lateral Gradient Sub'+subj, fontsize=18)
        plt.show()
        
    print '\n'