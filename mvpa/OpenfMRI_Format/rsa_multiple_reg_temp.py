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
from sklearn.preprocessing import MinMaxScaler
import mvpa_utils

###SCRIPT ARGUMENTS

subj = 101

analysis_name = 'abs_value'

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
#conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']   
conditions = ['Food item', 'Trinket item']

###SCRIPT ARGUMENTS END

#which ds to use and which mask to use
#glm_ds_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
glm_ds_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/tstat_all_trials_4D.nii'
mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'

fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions)

#zscore features by run
#zscorer = ZScoreMapper(chunks_attr='chunks')
#zscorer.train(fds)
#fds_zscore = zscorer(fds)

#remove mean for each day
#X = fds.samples
#mean_day1 = np.mean(X[:100,:], axis=0)
#X[:100,:] = X[:100,:] - mean_day1
#mean_day2 = np.mean(X[100:200,:], axis=0)
#X[100:200,:] = X[100:200,:] - mean_day2
#mean_day3 = np.mean(X[200:,:], axis=0)
#X[200:,:] = X[200:,:] - mean_day3
#
#fds.samples = X

square_dsm_bool = False

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
inds_in_conds = np.where(item_list[:,1] == -1)[0]
ind_item_list = item_list[inds_in_conds, 0]

ds_item_identity = dataset_wizard(ind_item_list, targets=np.zeros(num_trials))
dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
res_stim_id = dsm(ds_item_identity)

#plot_mtx(res_stim_id, ds_value.sa.targets, 'ROI pattern correlation distances')

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
ds_item_categ = dataset_wizard(trial_categ, targets=np.zeros(num_trials))
dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
res_item_categ = dsm(ds_item_categ)

#res = roiRSA_1Ss(fds,mask_name,res_stim_id,control_dsms=[res_day],cmetric='pearson')

#model_dsms = np.column_stack((res_stim_id.samples.reshape(-1), res_value.samples.reshape(-1), res_fvalue.samples.reshape(-1), \
#                              res_tvalue.samples.reshape(-1), res_item_categ, res_day.samples.reshape(-1), res_run.samples.reshape(-1)))

#model_dsms = np.column_stack((res_stim_id.samples.reshape(-1), res_value.samples.reshape(-1), res_fvalue.samples.reshape(-1), \
#                              res_tvalue.samples.reshape(-1), res_item_categ))
#
#rsa_reg = rsa.Regression(model_dsms, pairwise_metric='correlation')

#make target dsm functions
pw_metric = 'correlation'
tdsm_stim_id = rsa.PDistTargetSimilarity(res_stim_id, pairwise_metric=pw_metric, comparison_metric='spearman')
tdsm_value = rsa.PDistTargetSimilarity(res_value, pairwise_metric=pw_metric, comparison_metric='spearman')
tdsm_fvalue = rsa.PDistTargetSimilarity(res_fvalue, pairwise_metric=pw_metric, comparison_metric='spearman')
tdsm_tvalue = rsa.PDistTargetSimilarity(res_tvalue, pairwise_metric=pw_metric, comparison_metric='spearman')
tdsm_item_categ = rsa.PDistTargetSimilarity(res_item_categ, pairwise_metric=pw_metric, comparison_metric='spearman')
tdsm_run = rsa.PDistTargetSimilarity(res_run, pairwise_metric=pw_metric, comparison_metric='spearman')
tdsm_day = rsa.PDistTargetSimilarity(res_day, pairwise_metric=pw_metric, comparison_metric='spearman')

mask_loop = ['acc', 'paracingulate', 'frontal_pole', 'm_OFC', 'l_OFC', 'posterior_OFC', 'sup_frontal_gyr']

for mask in mask_loop:
    
    mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
    fds_mask = mask_dset(fds, mask_name)
    
    #SVD dim reduction
#    svdmapper = SVDMapper()
#    #get_SVD_sliced = lambda x: ChainMapper([svdmapper, StaticFeatureSelection(x)])
#    svdmapper.train(fds_mask)
#    fds_svd = svdmapper.forward(fds_mask)
#    #take first 50 dims
#    fds_svd = fds_svd[:,:100]

    #remove voxels with no variance
    X_var = np.var(fds_mask.samples, axis=0)
    pos_var_inds = np.where(X_var != 0)[0]
    fds_mask = fds_mask[:,pos_var_inds]
#
#    res_mask = rsa_reg(fds_mask)
#    
#    print 'Mask ',mask
#    print 'Stim ID Coef ',res_mask.samples[0]
#    print 'Value Coef ',res_mask.samples[1]
#    print 'Food Value Coef ',res_mask.samples[2]
#    print 'Trinket Value Coef ',res_mask.samples[3]
#    print 'Item Category Coef ',res_mask.samples[4]
#    #print 'Day Coef ',res_mask.samples[5]
#    #print 'Run Coef ',res_mask.samples[6]
#    print '\n'

#    fmri_stim_id = tdsm_stim_id(fds_mask)
#    fmri_value = tdsm_value(fds_mask)
#    fmri_fvalue = tdsm_fvalue(fds_mask)
#    fmri_tvalue = tdsm_tvalue(fds_mask)
#    fmri_item_categ = tdsm_item_categ(fds_mask)
#    fmri_run = tdsm_run(fds_mask)
#    fmri_day = tdsm_day(fds_mask)
#    
#    print 'Mask ',mask
#    print 'Stim ID Correlation ',fmri_stim_id.samples[0][0]
#    print 'Value Correlation ',fmri_value.samples[0][0]
#    print 'Food Value Correlation ',fmri_fvalue.samples[0][0]
#    print 'Trinket Value Correlation ',fmri_tvalue.samples[0][0]
#    print 'Item Category Correlation ',fmri_item_categ.samples[0][0]
#    print 'Run Correlation ',fmri_run.samples[0][0]
#    print 'Day Correlation ',fmri_day.samples[0][0]
#    print '\n'
    
    data = {}
    data['sub'+str(subj)] = fds_mask
    fmri_stim_id = roiRSA_nSs(data,mask_name,res_stim_id,partial_dsm=res_day,cmetric='spearman')
    fmri_value = roiRSA_nSs(data,mask_name,res_value,partial_dsm=res_day,cmetric='spearman')
    fmri_fvalue = roiRSA_nSs(data,mask_name,res_fvalue,partial_dsm=res_day,cmetric='spearman')
    fmri_tvalue = roiRSA_nSs(data,mask_name,res_tvalue,partial_dsm=res_day,cmetric='spearman')
    fmri_item_categ = roiRSA_nSs(data,mask_name,res_item_categ,partial_dsm=res_day,cmetric='spearman')
    fmri_run = roiRSA_nSs(data,mask_name,res_run,partial_dsm=res_day,cmetric='spearman')
    fmri_day = roiRSA_nSs(data,mask_name,res_day,partial_dsm=res_day,cmetric='spearman')
    
    print '\n'
    print 'Mask ',mask
    print 'Stim ID Correlation ',fmri_stim_id[1]['sub'+str(subj)][0]
    print 'Value Correlation ',fmri_value[1]['sub'+str(subj)][0]
    print 'Food Value Correlation ',fmri_fvalue[1]['sub'+str(subj)][0]
    print 'Trinket Value Correlation ',fmri_tvalue[1]['sub'+str(subj)][0]
    print 'Item Category Correlation ',fmri_item_categ[1]['sub'+str(subj)][0]
    print 'Run Correlation ',fmri_run[1]['sub'+str(subj)][0]
    print 'Day Correlation ',fmri_day[1]['sub'+str(subj)][0]
    print '\n'
    
    
#plot fds mask RDM
#dsm = PDist(pairwise_metric='correlation', square=True)
#res_fmri = dsm(fds_mask)
#
#plot_mtx(res_fmri, fds_mask.sa.targets, 'ROI pattern correlation distances')
#    
#if __debug__:
#    debug.active += ["SLC"]
#    
##sl_rsa_reg = sphere_searchlight(ChainLearner([rsa_reg, TransposeMapper()]), radius=4)
#sl_rsa_reg = sphere_searchlight(rsa_reg, radius=4)
#
#slres_rsa_reg = sl_rsa_reg(fds_mask[:,:1000])

 