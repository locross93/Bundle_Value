#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:45:06 2021

@author: logancross
"""

from mvpa2.suite import *
from pymvpaw import *
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/Users/logancross/Documents/Bundle_Value/mvpa/")
import os
os.chdir("/Users/logancross/Documents/Bundle_Value/mvpa/")
import mvpa_utils 

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj_list = ['104','105','107','108','109','110','111','113','114']
subj_list = ['105','107','108','109','110','111','113','114']

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']

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
             
#mask_loop = ['Frontal_Med_Orb','Frontal_Mid_2',
#    'OFClat','OFCmed',
#    'Supp_Motor_Area','ACC_pre', #motor, frontal
#    'Calcarine', 'Fusiform', #visual areas
#    'Caudate','Hippocampus', #subcortical areas
#    'Precuneus','Parietal_Sup'] #parietal, cingulate
             
save = True

###SCRIPT ARGUMENTS END

for subj in subj_list:
    print subj
    #which ds to use and which mask to use
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    #brain_mask = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    fds = mvpa_utils.make_targets_choice(subj, glm_ds_file, brain_mask, conditions)
    
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
        
    clf = LinearCSVMC(C=-1)
    
    score_dict = {}
    mask_scores = []
    for mask in mask_loop: 
        print mask
        mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
        temp_score = mvpa_utils.roiClass_1Ss(fds, mask_name)
        print temp_score
        mask_scores.append(temp_score)
        score_dict[mask] = temp_score
        print '\n'
        
        
        
    
#    plt.bar(np.arange(len(mask_names)), mask_scores, edgecolor='black')
#    plt.xticks(np.arange(len(mask_names)), mask_names, rotation=90)
#    plt.ylabel('Prediction Accuracy (%) Sub'+subj, fontsize=16)
#    plt.show()
    
    plt.bar(np.arange(len(mask_loop)), mask_scores, edgecolor='black')
    plt.xticks(np.arange(len(mask_loop)), mask_loop, rotation=90)
    plt.ylabel('Prediction Accuracy (%) Sub'+subj, fontsize=16)
    plt.show()
    
    #save
    if save:
        save_path = bundle_path+'mvpa/analyses/sub'+str(subj)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        np.save(save_path+'/decode_choice_rois.npy', score_dict)
  
#if __debug__:
#    debug.active += ["SLC"]
#    
#radius = 4    
#fds_small = fds[:,:100]
#sl_map = slClass_1Ss(fds_small, radius)

#mask = 'Calcarine'
#mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
#c_list = [-2, -1, -0.5, 0, 0.5, 1, 10]
#for c in c_list:
#    clf = LinearCSVMC(C=c)
#    temp_score = mvpa_utils.roiClass_1Ss(fds, mask_name)
#    print 'C=',c,' Score=',temp_score
#    print '\n'