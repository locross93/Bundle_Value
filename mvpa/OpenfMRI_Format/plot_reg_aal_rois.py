#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:39:55 2021

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

analysis_name = 'abs_value'

subj_list = ['104','105','107','108','109','110','111','113','114']

mask_loop = ['ACC_pre','ACC_sub','ACC_sup','Amygdala','Caudate','Cingulate_Mid','Cingulate_Post','Cuneus',
    	'Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial','Fusiform',
    	'Hippocampus','Insula','N_Acc','OFCant','OFClat','OFCmed','OFCpost','Paracentral_Lobule','Precentral','Precuneus','Putamen','Supp_Motor_Area']

mask_loop = ['ACC_pre','ACC_sub','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri',
             'Supp_Motor_Area','Precentral']

mask_names = ['ACCpre','ACCsub','ACCsup',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC/SFG','dlPFC/SFG','MFG','IFG',
              'SMA','Motor Cortex']

mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']
              
            
scores_by_roi = np.zeros([len(subj_list), len(mask_loop)])

subj_count = 0
for subj in subj_list:

    scores_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/aal_reg_scores_'+analysis_name+'.npy'
    subj_score_dict = np.load(scores_file)
    subj_score_dict = subj_score_dict.item()
    
    score_list_temp = []
    
    for mask in mask_loop:
        score_list_temp.append(subj_score_dict[mask])
        
    plt.bar(np.arange(len(mask_loop)), score_list_temp, edgecolor='black')
    plt.xticks(np.arange(len(mask_loop)), mask_loop, rotation=90)
    plt.ylabel('Prediction Accuracy (r)', fontsize=16)
    plt.title('MVPA Scores Sub'+subj, fontsize=18)
    plt.show()
    
    scores_by_roi[subj_count,:] = score_list_temp
    
    subj_count+=1
    
#plot the average
avg_mvpa_scores = np.mean(scores_by_roi, axis=0)
sem_mvpa_scores = scipy.stats.sem(scores_by_roi)

plt.figure(figsize=(10, 5)) 
plt.bar(np.arange(len(mask_loop)), avg_mvpa_scores, yerr=sem_mvpa_scores, edgecolor='black')
plt.xticks(np.arange(len(mask_loop)), mask_names, rotation=0, fontsize=10)
plt.ylabel('Prediction Accuracy (r)', fontsize=16)
plt.title('MVPA Scores All Subjects', fontsize=18)
plt.show()
    