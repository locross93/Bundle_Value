#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:00:43 2019

@author: logancross
"""

from mvpa2.suite import *
#from pymvpaw import *
import matplotlib.pyplot as plt
from mvpa2.measures import rsa
from mvpa2.measures.rsa import PDist
from mvpa2.measures.searchlight import sphere_searchlight
from scipy.spatial.distance import squareform
from scipy.stats import zscore
from scipy.stats import rankdata, pearsonr
from sklearn.preprocessing import MinMaxScaler
import mvpa_utils

###SCRIPT ARGUMENTS

subj = 101

analysis_name = 'abs_value'

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']   
#conditions = ['Food item', 'Trinket item']

#which ds to use and which mask to use
#glm_ds_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii'
glm_ds_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/tstat_all_trials_4D.nii'
#glm_ds_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel_smooth/tstat_all_trials_4D.nii'
mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
#mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'

if analysis_name == 'rel_value':
    fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions, train_on='default', test_on='default', relative_value=True)
else:
    fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions)

#h5save('/Users/logancross/Documents/Bundle_Value/mvpa/datasets/sub'+str(subj)+'/all_trials_4D_pfc.hdf5',fds)

#fds = h5load('/Users/logancross/Documents/Bundle_Value/mvpa/datasets/sub'+str(subj)+'/all_trials_4D_fullbrain')

square_dsm_bool = False

remove_within_day = True

###SCRIPT ARGUMENTS END

#control dsm for day
num_trials = fds.shape[0]
trials_per_day = num_trials/3
day_array = np.array([c/trials_per_day for c in range(num_trials)])
ds_day = dataset_wizard(day_array, targets=np.zeros(num_trials))

dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
res_day = dsm(ds_day)

#control dsm for run
ds_run = dataset_wizard(fds.chunks, targets=np.zeros(num_trials))

dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
res_run = dsm(ds_run)

#plot_mtx(res_run, ds_value.sa.targets, 'ROI pattern correlation distances')

#stimulus identity
item_list = np.genfromtxt('/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/task_info/item_list.txt')

#if only individual item trials, only include these trials
if conditions == ['Food item', 'Trinket item']:
    inds_in_conds = np.where(item_list[:,1] == -1)[0]
    item_list = item_list[inds_in_conds, :]

num_items = len(item_list)
square_rdm = np.ones([num_items, num_items])

for i in range(num_items):
    for j in range(num_items):
        items_row = item_list[i,:]
        items_col = item_list[j,:]
        if items_row[0] in items_col:
            square_rdm[i,j] = 0
        elif items_row[1] > 0 and items_row[1] in items_col:
            square_rdm[i,j] = 0

if square_dsm_bool:
    res_stim_id = square_rdm
else:
    res_stim_id = squareform(square_rdm)

#plot_mtx(square_rdm, np.arange(num_items), 'ROI pattern correlation distances')

#value
value = fds.targets
#value_norm = zscore(value)
scaler = MinMaxScaler()
value_norm = scaler.fit_transform(value.reshape(-1,1)).reshape(-1)
ds_value = dataset_wizard(value_norm, targets=np.zeros(num_trials))
dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
res_value = dsm(ds_value)

#food value
trial_categ = fds.sa.trial_categ
food_inds = np.where(trial_categ == 1)[0]
food_value = value[food_inds]
#make the trinket value trials have a value of the mean value so they are in the middle of a dissimilarity matrix with minmaxscaling
mean_value = np.mean(food_value)
food_value_norm = mean_value*np.ones(num_trials)
food_value_norm[food_inds] = food_value
food_value_norm = scaler.fit_transform(food_value_norm.reshape(-1,1))
ds_fvalue = dataset_wizard(food_value_norm, targets=np.zeros(num_trials))
dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
res_fvalue = dsm(ds_fvalue)

#plot_mtx(res_fvalue, ds_fvalue.sa.targets, 'ROI pattern correlation distances')

#trinket value
trinket_inds = np.where(trial_categ == 2)[0]
trinket_value = value[trinket_inds]
#make the food value trials have a value of the mean value so they are in the middle of a dissimilarity matrix with minmaxscaling
mean_value = np.mean(trinket_value)
trinket_value_norm = mean_value*np.ones(num_trials)
trinket_value_norm[trinket_inds] = trinket_value
trinket_value_norm = scaler.fit_transform(trinket_value_norm.reshape(-1,1))
ds_tvalue = dataset_wizard(trinket_value_norm, targets=np.zeros(num_trials))
dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
res_tvalue = dsm(ds_tvalue)

#plot_mtx(res_tvalue, ds_tvalue.sa.targets, 'ROI pattern correlation distances')

#food vs trinket category
#ds_item_categ = dataset_wizard(trial_categ, targets=np.zeros(num_trials))
#dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
#res_item_categ = dsm(ds_item_categ)

#bundle value
bundle_inds = np.where(trial_categ > 2)[0]
bundle_value = value[bundle_inds]
#make the ind item value trials have a value of the mean value so they are in the middle of a dissimilarity matrix with minmaxscaling
ind_item_inds = np.where(trial_categ < 3)[0]
mean_value = np.mean(value[ind_item_inds])
bundle_value_norm = mean_value*np.ones(num_trials)
bundle_value_norm[bundle_inds] = bundle_value
bundle_value_norm = scaler.fit_transform(bundle_value_norm.reshape(-1,1))
ds_bvalue = dataset_wizard(bundle_value_norm, targets=np.zeros(num_trials))
dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
res_bvalue = dsm(ds_bvalue)

if remove_within_day:
    #btwn_run_inds = np.where(res_run.samples == 1)[0]
    btwn_run_inds = np.where(res_day.samples == 1)[0]
    #btwn_run_inds = np.where(res_day.samples == 0)[0]
    #res_stim_id = res_stim_id[btwn_run_inds]
    #res_fvalue = res_fvalue[btwn_run_inds]
    #res_tvalue = res_tvalue[btwn_run_inds]
    #res_bvalue = res_bvalue[btwn_run_inds]
    #res_day = res_day[btwn_run_inds]
    
#if __debug__:
#    debug.active += ["SLC"]
#
#tdsm_fvalue = rsa.PDistTargetSimilarity(res_fvalue, pairwise_metric=pairwise_metric, comparison_metric='spearman')
#sl_rsa_fvalue = sphere_searchlight(ChainLearner([tdsm_fvalue, TransposeMapper()]), radius=3)
#sl_fmri_value = sl_rsa_fvalue(fds_mask[:,:20])
#
#rsa1 = mvpa_utils.rsa_custom(res_fvalue, square_dsm_bool, pairwise_metric, comparison_metric)
#sl_rsa_fvalue = sphere_searchlight(ChainLearner([rsa1, TransposeMapper()]), radius=3)
#sl_fmri_value = sl_rsa_fvalue(fds_mask[:,:20])


model_dsms = np.column_stack((res_stim_id, res_fvalue.samples.reshape(-1), \
                              res_tvalue.samples.reshape(-1), res_bvalue.samples.reshape(-1)))


rsa_reg = rsa.Regression(model_dsms, pairwise_metric='correlation') #, keep_pairs=btwn_run_inds)
#
#if __debug__:
#    debug.active += ["SLC"]
#
#sl_rsa_reg = sphere_searchlight(rsa_reg, radius=3)
#sl_fmri_value = sl_rsa_reg(fds)
#
##save
#scores_per_voxel = sl_fmri_value.samples
#vector_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/rsa_value_pcday'
#h5save(vector_file,scores_per_voxel)
#
#nimg0 = map2nifti(fds, scores_per_voxel[0,:])
#nii_file0 = vector_file+'_stim_id.nii.gz'
#nimg0.to_filename(nii_file0)
#
#nimg1 = map2nifti(fds, scores_per_voxel[1,:])
#nii_file1 = vector_file+'_fvalue.nii.gz'
#nimg1.to_filename(nii_file1)
#
#nimg2 = map2nifti(fds, scores_per_voxel[2,:])
#nii_file2 = vector_file+'_tvalue.nii.gz'
#nimg2.to_filename(nii_file2)
#
#nimg3 = map2nifti(fds, scores_per_voxel[3,:])
#nii_file3 = vector_file+'_bvalue.nii.gz'
#nimg3.to_filename(nii_file3)

#frontal
#mask_loop = ['sup_frontal_gyr', 'acc', 'paracingulate', 'frontal_pole', 'm_OFC', 'l_OFC', 'posterior_OFC']
#
mask_loop = ['m_OFC', 'l_OFC', 'posterior_OFC']

#visual
#mask_loop = ['occipital_pole', 'loc_inferior', 'it_gyrus_occip', 'mt_gyrus_occip', 
#             'it_gyrus_post', 'mt_gyrus_post', 'it_gyrus_ant', 'mt_gyrus_ant', 'temporal_pole']

for mask in mask_loop:
    
    mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
    fds_mask = mask_dset(fds, mask_name)
    
    #remove voxels with no variance
    X_var = np.var(fds_mask.samples, axis=0)
    pos_var_inds = np.where(X_var != 0)[0]
    fds_mask = fds_mask[:,pos_var_inds]


    res_mask = rsa_reg(fds_mask)
    
    print 'Mask ',mask
    print 'Stim ID Coef ',res_mask.samples[0]
    print 'Food Value Coef ',res_mask.samples[1]
    print 'Trinket Value Coef ',res_mask.samples[2]
    print 'Bundle Value Coef ',res_mask.samples[3]
    #print 'Run Coef ',res_mask.samples[4]
    #print 'Day Coef ',res_mask.samples[4]
    print '\n'
    
#    dsm_fmri_mask = PDist(pairwise_metric='correlation', square=square_dsm_bool)
#    res_fmri = dsm_fmri_mask(fds_mask)
#    res_fmri = rankdata(res_fmri)
#    if remove_within_day:
#        res_fmri = res_fmri[btwn_run_inds]
#    
#    print 'Mask ',mask
#    #print 'Stim ID Coef ',round(pearsonr(res_fmri, rankdata(res_stim_id))[0], 3)
#    print 'Food Value Coef ',round(pearsonr(res_fmri, rankdata(res_fvalue))[0], 3)
#    #print 'Trinket Value Coef ',round(pearsonr(res_fmri, rankdata(res_tvalue))[0], 3)
#    #print 'Bundle Value Coef ',round(pearsonr(res_fmri, rankdata(res_bvalue))[0], 3)
#    #print 'Run Coef ',res_mask.samples[4]
#    #print 'Day Coef ',round(pearsonr(res_fmri, rankdata(res_day))[0], 3)
#    print '\n'
    
#    #KENDALL TAU
#    print 'Mask ',mask
#    print 'Stim ID Coef ',round(kendalltau(res_fmri, rankdata(res_stim_id))[0], 3)
##    print 'Food Value Coef ',round(kendalltau(res_fmri, rankdata(res_fvalue))[0], 3)
##    print 'Trinket Value Coef ',round(kendalltau(res_fmri, rankdata(res_tvalue))[0], 3)
##    print 'Bundle Value Coef ',round(kendalltau(res_fmri, rankdata(res_bvalue))[0], 3)
##    #print 'Run Coef ',res_mask.samples[4]
##    #print 'Day Coef ',round(pearsonr(res_fmri, rankdata(res_day))[0], 3)
#    print '\n'


#    fmri_fvalue = roiRSA_1Ss(fds_mask,mask_name,res_fvalue,partial_dsm=res_day,cmetric='spearman')
#    fmri_tvalue = roiRSA_1Ss(fds_mask,mask_name,res_tvalue,partial_dsm=res_day,cmetric='spearman')
#    fmri_bvalue = roiRSA_1Ss(fds_mask,mask_name,res_bvalue,partial_dsm=res_day,cmetric='spearman')
#    
#    print '\n'
#    print 'Mask ',mask
#    print 'Food Value Correlation ',fmri_fvalue[0]
#    print 'Trinket Value Correlation ',fmri_tvalue[0]
#    print 'Bundle Value Correlation ',fmri_bvalue[0]
#    print '\n'

#    pairwise_metric = 'correlation'
#    comparison_metric = 'spearman'
#
#    print 'Mask ',mask
#    rsa1 = mvpa_utils.rsa_custom(res_fvalue, square_dsm_bool, remove_within_day, pairwise_metric, comparison_metric)
#    print 'Food Value Coef ',rsa1(fds_mask).samples
###    print 'Trinket Value Coef ',mvpa_utils.rsa_custom(fds_mask, res_tvalue, square_dsm_bool, pairwise_metric, comparison_metric).samples
###    print 'Bundle Value Coef ',mvpa_utils.rsa_custom(fds_mask, res_bvalue, square_dsm_bool, pairwise_metric, comparison_metric).samples
#    print '\n'
    


 