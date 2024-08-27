# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:36:17 2021

@author: locro
"""

#from mvpa2.suite import *
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/Users/locro/Documents/Bundle_Value/mvpa/")
import os
os.chdir("/Users/locro/Documents/Bundle_Value/mvpa/")
import mvpa_utils 
import seaborn as sns
import scipy

bundle_path = '/Users/locro/Documents/Bundle_Value/'

subj_list = ['104','105','107','108','109','110','111','113','114']

mask_loop = ['Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial',
    'Insula','OFCant','OFClat','OFCmed','OFCpost',
    'Paracentral_Lobule','Precentral','Supp_Motor_Area','ACC_pre','ACC_sub','ACC_sup', #motor, frontal
    'Calcarine', 'Lingual','Occipital_Inf','Occipital_Mid','Occipital_Sup','Fusiform','Temporal_Inf','Temporal_Mid','Temporal_Pole_Mid','Temporal_Pole_Sup','Temporal_Sup', #visual areas
    'Amygdala','Caudate','Putamen','N_Acc','Hippocampus','ParaHippocampal', #subcortical areas
    'Cingulate_Mid','Cingulate_Post','Cuneus','Precuneus','Parietal_Inf','Parietal_Sup','Postcentral','SupraMarginal','Angular'] #parietal, cingulate

scores_by_roi = np.zeros([len(subj_list), len(mask_loop)])

subj_count = 0
for subj in subj_list:

    scores_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/decode_choice_rois.npy'
    subj_score_dict = np.load(scores_file, encoding = 'latin1', allow_pickle=True)
    subj_score_dict = subj_score_dict.item()
    
    score_list_temp = []
    
    for mask in mask_loop:
        score_list_temp.append(subj_score_dict[mask])
        
    plt.bar(np.arange(len(mask_loop)), score_list_temp, edgecolor='black')
    plt.xticks(np.arange(len(mask_loop)), mask_loop, rotation=90)
    plt.ylabel('Classification Accuracy', fontsize=16)
    plt.title('MVPA Scores Sub'+subj, fontsize=18)
    plt.show()
    
    scores_by_roi[subj_count,:] = score_list_temp
    
    subj_count+=1
    
#plot the average
avg_mvpa_scores = np.mean(scores_by_roi, axis=0)
sem_mvpa_scores = scipy.stats.sem(scores_by_roi)

plt.figure(figsize=(10, 5)) 
plt.bar(np.arange(len(mask_loop)), avg_mvpa_scores, yerr=sem_mvpa_scores, edgecolor='black')
plt.xticks(np.arange(len(mask_loop)), mask_loop, rotation=90, fontsize=10)
plt.ylabel('Prediction Accuracy (r)', fontsize=16)
plt.title('MVPA Scores All Subjects', fontsize=18)
plt.show()