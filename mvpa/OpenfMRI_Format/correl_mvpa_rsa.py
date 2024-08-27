#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:54:07 2019

@author: logancross
"""

subj = 103
mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
#rsa_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/whole_brain_rsa_reg_btwn_day_rel_value_bvalue.nii.gz'
rsa_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/pfc_rsa_bundle_value_pcday.nii.gz'
fds_rsa = fmri_dataset(samples=rsa_file, targets=0, chunks=0, mask=mask_name)
svr_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_ridge_bundle_value.nii.gz'
fds_svr = fmri_dataset(samples=svr_file, targets=0, chunks=0, mask=mask_name)
univariate_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/value_alldays_ICA_nomotion/spmT_0002.nii'
fds_uni = fmri_dataset(samples=univariate_file, targets=0, chunks=0, mask=mask_name)

rsa_samples = fds_rsa.samples.reshape(-1,)
svr_samples = fds_svr.samples.reshape(-1,)
uni_samples = fds_uni.samples.reshape(-1,)
abs_uni_samples = np.absolute(uni_samples)

scipy.stats.pearsonr(rsa_samples, svr_samples)

scipy.stats.pearsonr(rsa_samples, abs_uni_samples)

scipy.stats.pearsonr(svr_samples, abs_uni_samples)
