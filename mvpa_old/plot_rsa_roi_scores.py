#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:03:34 2021

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
#subj_list = ['105','106','107','108','109','110','111','112','113','114']

#scores_by_roi = np.zeros([len(subj_list), 44, 8])
#scores_by_roi = np.zeros([len(subj_list), 14, 6])
scores_by_roi = np.zeros([len(subj_list), 6, 6])

subj_count = 0
for subj in subj_list:
    #temp_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/rsa_roi_scores2.csv')
    temp_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/rsa_reg_subcortical_scores.csv')
    scores_by_roi[subj_count,:,:] = temp_df.iloc[:,1:].values
    subj_count+=1
    
avg_scores = np.mean(scores_by_roi, axis=0)
sem_scores = scipy.stats.sem(scores_by_roi, axis=0)
    
column_names = list(temp_df.columns)[1:]

#mask_loop = ['Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial',
#    'Insula','OFCant','OFClat','OFCmed','OFCpost',
#    'Paracentral_Lobule','Precentral','Supp_Motor_Area','ACC_pre','ACC_sub','ACC_sup', #motor, frontal
#    'Calcarine', 'Lingual','Occipital_Inf','Occipital_Mid','Occipital_Sup','Fusiform','Temporal_Inf','Temporal_Mid','Temporal_Pole_Mid','Temporal_Pole_Sup','Temporal_Sup', #visual areas
#    'Amygdala','Caudate','Putamen','N_Acc','Hippocampus','ParaHippocampal', #subcortical areas
#    'Cingulate_Mid','Cingulate_Post','Cuneus','Precuneus','Parietal_Inf','Parietal_Sup','Postcentral','SupraMarginal','Angular'] #parietal, cingulate
#
#mask_loop = ['ACC_pre','ACC_sup',
#             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
#             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri',
#             'Calcarine', 'Fusiform']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG',
              'V1','Fusiform']

mask_names = ['Amygdala','Caudate','Putamen','N_Acc','Hippocampus','ParaHippocampal']
             
#legend_label = 'Spearman Correlation ($\\rho$)'
legend_label = 'Coefficient'
vmax = 0.1
mask_mat = np.zeros_like(avg_scores)
mask_mat[np.triu_indices_from(mask_mat)] = True   
plt.figure(figsize=(10, 10))
f = sns.heatmap(avg_scores, annot=True, annot_kws={"size": 7},  
                xticklabels=column_names, yticklabels=mask_names, vmin=0.0, vmax=vmax, cbar_kws={'label':legend_label}) 
f.set_title('Group Average RSA Regression Scores', fontsize = 20)   
plt.show()