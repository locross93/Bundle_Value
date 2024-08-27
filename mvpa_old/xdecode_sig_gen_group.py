#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:47:43 2021

@author: logancross
"""

from mvpa2.suite import *

#find voxels sig in s2b and b2s

s2b_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/S2B/spmT_0001.nii'
s2b = fmri_dataset(samples=s2b_file)

b2s_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/B2S/spmT_0001.nii'
b2s = fmri_dataset(samples=b2s_file)

#this changes depending on number of subjects/df
p_01_thr = 2.65

sig_s2b = np.where(s2b.samples > p_01_thr)[1]
sig_b2s = np.where(b2s.samples > p_01_thr)[1]

sig_gen = np.intersect1d(sig_s2b, sig_b2s)

scores_gen = np.zeros(s2b.shape)
scores_gen[0,sig_gen] = 1

nimg = map2nifti(s2b, scores_gen)
nii_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/sig_generalize_p01.nii.gz'
nimg.to_filename(nii_file)

s2b_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/S2B/spmT_0001.nii'
s2b = fmri_dataset(samples=s2b_file)

b2s_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/B2S/spmT_0001.nii'
b2s = fmri_dataset(samples=b2s_file)

#find voxels sig only in s2s and b2b only - at < 0.001
s2s_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/S2S/spmT_0001.nii'
s2s = fmri_dataset(samples=s2s_file)

b2b_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/B2B/spmT_0001.nii'
b2b = fmri_dataset(samples=b2b_file)

sig_s2s = np.where(s2s.samples > p_001_thr)[1]
sig_b2b = np.where(b2b.samples > p_001_thr)[1]
sig_s2b = np.where(s2b.samples > p_001_thr)[1]
sig_b2s = np.where(b2s.samples > p_001_thr)[1]

sig_voxs_not_s2s = np.concatenate((sig_s2b, sig_b2b, sig_b2s), axis=None)
sig_s2s_only = np.array([ind for ind in sig_s2s if ind not in sig_voxs_not_s2s])

scores_s2s_only = np.zeros(s2s.shape)
scores_s2s_only[0,sig_s2s_only] = 1

nimg = map2nifti(s2s, scores_s2s_only)
nii_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/sig_s2s_only_p001.nii.gz'
nimg.to_filename(nii_file)

sig_voxs_not_b2b = np.concatenate((sig_s2s, sig_s2b, sig_b2s), axis=None)
sig_b2b_only = np.array([ind for ind in sig_b2b if ind not in sig_voxs_not_b2b])

scores_b2b_only = np.zeros(b2b.shape)
scores_b2b_only[0,sig_b2b_only] = 1

nimg = map2nifti(b2b, scores_b2b_only)
nii_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/sig_b2b_only_p001.nii.gz'
nimg.to_filename(nii_file)