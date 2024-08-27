#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:48:31 2019

@author: logancross
"""
from mvpa2.suite import *
import matplotlib.pyplot as plt
from mvpa2.measures import rsa
from mvpa2.measures.searchlight import sphere_searchlight
from scipy.spatial.distance import squareform
import mvpa_utils

# little helper function to plot dissimilarity matrices
# since we are using correlation-distance, we use colorbar range of [0,2]
def plot_mtx(mtx, labels, title):
    pl.figure()
    pl.imshow(mtx, interpolation='nearest')
    pl.xticks(range(len(mtx)), labels, rotation=-45)
    pl.yticks(range(len(mtx)), labels)
    pl.title(title)
    pl.clim((0, 2))
    pl.colorbar()

###SCRIPT ARGUMENTS

subj = 101

analysis_name = 'abs_value'

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
#conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']   
conditions = ['Food item', 'Trinket item']

###SCRIPT ARGUMENTS END

#which ds to use and which mask to use
glm_ds_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'

fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions)

item_list = np.genfromtxt('/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/task_info/item_list.txt')

inds_in_conds = np.where(item_list[:,1] == -1)[0]
ind_item_list = item_list[inds_in_conds, 0]

num_trials = len(ind_item_list)
ds_item_identity = dataset_wizard(ind_item_list, targets=np.zeros(num_trials))

dsm = rsa.PDist(pairwise_metric='matching', square=False)
#dsm = rsa.PDist(pairwise_metric='matching', square=True)
res_stim_id = dsm(ds_item_identity)

#matrix = squareform(res_stim_id.samples.reshape(-1))

#plot_mtx(squareform(res_stim_id.samples.reshape(-1)), ds_item_identity.sa.targets, 'ROI pattern correlation distances')

tdsm = rsa.PDistTargetSimilarity(res_stim_id, pairwise_metric='correlation', comparison_metric='spearman')

if __debug__:
    debug.active += ["SLC"]

sl_tdsm = sphere_searchlight(ChainLearner([tdsm, TransposeMapper()]), radius=4)
slres_tdsm = sl_tdsm(fds)

## reverse map scores back into nifti format
#scores_per_voxel = slres_tdsm.samples[0,:]
#vector_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/rsa_stim_identity'
#h5save(vector_file,scores_per_voxel)
#nimg = map2nifti(fds, scores_per_voxel)
#nii_file = vector_file+'.nii.gz'
#nimg.to_filename(nii_file)
