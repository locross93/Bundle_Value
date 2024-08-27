#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:11:07 2021

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

analysis_name = 'wbrain_rsa_pixels_ind_item_trials'

subj_list = ['104','105','107','108','109','110','111','113','114']

#mask_loop = ['ACC_pre','ACC_sub','ACC_sup','Amygdala','Caudate','Cingulate_Mid','Cingulate_Post','Cuneus',
#    	'Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial','Fusiform',
#    	'Hippocampus','Insula','N_Acc','OFCant','OFClat','OFCmed','OFCpost','Paracentral_Lobule','Precentral','Precuneus','Putamen','Supp_Motor_Area']

#PFC rois
mask_loop = ['ACC_pre','ACC_sub','ACC_sup','Caudate',
    	'Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial',
    	'Insula','N_Acc','OFCant','OFClat','OFCmed','OFCpost','Paracentral_Lobule','Precentral','Putamen','Supp_Motor_Area']

mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

mask_loop = ['Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial',
    'Insula','OFCant','OFClat','OFCmed','OFCpost',
    'Paracentral_Lobule','Precentral','Supp_Motor_Area','ACC_pre','ACC_sub','ACC_sup', #motor, frontal
    'Calcarine', 'Lingual','Occipital_Inf','Occipital_Mid','Occipital_Sup','Fusiform','Temporal_Inf','Temporal_Mid','Temporal_Pole_Mid','Temporal_Pole_Sup','Temporal_Sup', #visual areas
    'Amygdala','Caudate','Putamen','N_Acc','Hippocampus','ParaHippocampal', #subcortical areas
    'Cingulate_Mid','Cingulate_Post','Cuneus','Precuneus','Parietal_Inf','Parietal_Sup','Postcentral','SupraMarginal','Angular'] #parietal, cingulate
             
mask_names = ['Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial',
    'Insula','OFCant','OFClat','OFCmed','OFCpost',
    'Paracentral_Lobule','Precentral','Supp_Motor_Area','ACC_pre','ACC_sub','ACC_sup', #motor, frontal
    'Calcarine', 'Lingual','Occipital_Inf','Occipital_Mid','Occipital_Sup','Fusiform','Temporal_Inf','Temporal_Mid','Temporal_Pole_Mid','Temporal_Pole_Sup','Temporal_Sup', #visual areas
    'Amygdala','Caudate','Putamen','N_Acc','Hippocampus','ParaHippocampal', #subcortical areas
    'Cingulate_Mid','Cingulate_Post','Cuneus','Precuneus','Parietal_Inf','Parietal_Sup','Postcentral','SupraMarginal','Angular'] #parietal, cingulate

#scores_by_roi = np.zeros([len(subj_list), len(mask_loop)])
scores_by_roi = {}
for mask in mask_loop:
    scores_by_roi[mask] = []

subj_count = 0
for subj in subj_list:
    vector_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name
    #vector_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/rsa_value_btwn_day'
    scores_per_voxel = h5load(vector_file)
    scores_per_voxel = scores_per_voxel.reshape(-1)
    
    #make fds to get shape
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle'] 
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=False)
    
    score_list_temp = []
    sem_list_temp = []
    
    for mask in mask_loop: 
        mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
        brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
        #brain_mask = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
        masked = fmri_dataset(mask_name, mask=brain_mask)
        reshape_masked=masked.samples.reshape(fds.shape[1])
        reshape_masked=reshape_masked.astype(bool)
        mask_map = mask_mapper(mask=reshape_masked)
        
        mask_slice = mask_map[1].slicearg
        mask_inds = np.where(mask_slice == 1)[0]
        
        mask_scores = scores_per_voxel[mask_inds]
        
        score_list_temp.append(np.mean(mask_scores))
        sem_list_temp.append(scipy.stats.sem(mask_scores))
        
        scores_by_roi[mask].append(mask_scores)
    plt.bar(np.arange(len(mask_loop)), score_list_temp, yerr=sem_list_temp, edgecolor='black')
    plt.xticks(np.arange(len(mask_loop)), mask_names, rotation=90)
    plt.ylabel('Prediction Accuracy (r)', fontsize=16)
    plt.title('MVPA SL Scores Sub'+subj, fontsize=18)
    plt.show()
    
    #scores_by_roi[subj_count,:] = score_list_temp
    
    subj_count+=1
 
#plot the average
avg_mvpa_scores = []
sem_mvpa_scores = []
for mask in mask_loop:
    mask_scores = np.concatenate((scores_by_roi[mask]))
    avg_mvpa_scores.append(np.mean(mask_scores, axis=0))
    sem_mvpa_scores = scipy.stats.sem(mask_scores)

#plt.bar(np.arange(len(mask_loop)), avg_mvpa_scores, yerr=sem_mvpa_scores, edgecolor='black')
#plt.xticks(np.arange(len(mask_loop)), mask_names, rotation=90)
#plt.ylabel('Prediction Accuracy (r)', fontsize=16)
#plt.title('MVPA SL Scores All Subjects', fontsize=18)
#plt.show()

plt.figure(figsize=(10, 5)) 
plt.bar(np.arange(len(mask_loop)), avg_mvpa_scores, yerr=sem_mvpa_scores, edgecolor='black')
plt.xticks(np.arange(len(mask_loop)), mask_names, rotation=90, fontsize=10)
plt.ylabel('Average Voxel Prediction Accuracy (r)', fontsize=12)
plt.title('MVPA SL Scores All Subjects', fontsize=18)
plt.show()

    
    