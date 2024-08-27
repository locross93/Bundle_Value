#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:36:05 2021

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
import pandas as pd

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj = '105'

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

mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

scores_by_roi = []
avg_scores_xcat = []
perc_sig_all_cat = []
pref_scores_by_roi = []
mask_count = 0
#ADD SEMS TO ALL GRAPHS
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
    
    mask_label = mask_names[mask_count]
    
    mask_scores = [mask_label, np.mean(scores_s2s[mask_inds]), np.mean(scores_s2b[mask_inds]), np.mean(scores_b2b[mask_inds]), np.mean(scores_b2s[mask_inds])]
    scores_by_roi.append(mask_scores)
    
    #take average scores across category for every voxel in mask
    scores_all_cat = np.column_stack((scores_s2s[mask_inds], scores_s2b[mask_inds], scores_b2b[mask_inds], scores_b2s[mask_inds]))
    avg_scores_xcat_mask = np.mean(scores_all_cat, axis=1)
    avg_scores_xcat.append(np.mean(avg_scores_xcat_mask))
    
    #percentage of voxels in a roi over a r threshold
    r_thr = 0.05
    sig_voxs = scores_all_cat > r_thr
    sig_voxs_all_cat = np.where(np.mean(sig_voxs, axis=1) == 1)[0]
    perc_sig = float(sig_voxs_all_cat.shape[0]) / scores_all_cat.shape[0]
    perc_sig_all_cat.append(perc_sig)
    
    #which voxels are single item specific or bundle specific
    single_item_pref = scores_s2s[mask_inds] - scores_s2b[mask_inds]
    bundle_pref = scores_b2b[mask_inds] - scores_b2s[mask_inds]
    cat_pref_scores = [mask_label, np.mean(single_item_pref), np.mean(bundle_pref)]
    pref_scores_by_roi.append(cat_pref_scores)
    
    mask_count += 1
    
df_data = pd.DataFrame(scores_by_roi, columns = ['Mask','S2S','S2B','B2B','B2S'])  
df_plot = pd.melt(df_data, id_vars=["Mask"], var_name="Decoding Type", value_name="Accuracy")

ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot, ci=None)
plt.xticks(rotation=90)
plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('Cross Decoding Sub'+subj)
plt.show()

plt.bar(mask_names, avg_scores_xcat)
plt.xticks(rotation=90)
plt.title('Cross Decoding Sub'+subj)
plt.show()

plt.bar(mask_names, perc_sig_all_cat)
plt.xticks(rotation=90)
plt.title('Percent Sig Sub'+subj)
plt.show()

df_data_pref = pd.DataFrame(pref_scores_by_roi, columns = ['Mask','Single Item Pref.','Bundle Pref'])  
df_plot_pref = pd.melt(df_data_pref, id_vars=["Mask"], var_name="Trial Type", value_name="Category Preference")
ax = sns.barplot(x="Mask", y="Category Preference", hue="Trial Type", data=df_plot_pref, ci=None)
plt.xticks(rotation=90)
plt.legend(title='Trial Type', bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('Cross Decoding Preference Sub'+subj)
plt.show()