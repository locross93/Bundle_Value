#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:56:52 2019

@author: logancross
"""
def DMshuffle_custom(DM):
    inds2shuffle = np.arange(len(DM))
    return random.sample(inds2shuffle, len(DM))
    #return squareform(random.sample(squareform( DM ),len(squareform( DM ))))

square_dsm_bool = False

remove_within_day = True

###SCRIPT ARGUMENTS END

target_dsms = mvpa_utils.get_target_dsm(subj, fds, conditions, square_dsm_bool)

res_stim_id = target_dsms['stim_id']
res_fvalue = target_dsms['fvalue']
res_tvalue = target_dsms['tvalue']
res_bvalue = target_dsms['bvalue']

#TEMPORARY
vox_inds = fds.fa.voxel_indices
sample_vox = [84, 120, 32]
for i in range(len(vox_inds)):
    temp_inds = vox_inds[i,:]
    if np.array_equal(temp_inds, sample_vox):
        vox2use = [i]
    
sl_rsa_reg = sphere_searchlight(rsa_reg, radius=3, center_ids=vox2use)
sl_fmri_value = sl_rsa_reg(fds)

res_stim_id = target_dsms['stim_id']
res_fvalue = target_dsms['fvalue']
res_tvalue = target_dsms['tvalue']
res_bvalue = target_dsms['bvalue']

if remove_within_day:
    res_day = target_dsms['day']
    btwn_run_inds = np.where(res_day.samples == 1)[0]
    
model_dsms = np.column_stack((res_stim_id, res_fvalue.samples.reshape(-1), \
                              res_tvalue.samples.reshape(-1), res_bvalue.samples.reshape(-1)))


rsa_reg = rsa.Regression(model_dsms, pairwise_metric='correlation', keep_pairs=btwn_run_inds)

#PERMUTATION TESTS FOR SINGLE SUBJECT LEVEL
#CLASS LABELS ARE SHUFFLED 100 TIMES TO CREATE A NONPARAMETRIC NULL DISTRIBUTION
target_dsms = mvpa_utils.get_target_dsm(subj, fds, conditions, square_dsm_bool)

res_stim_id = target_dsms['stim_id']
res_fvalue = target_dsms['fvalue']
res_tvalue = target_dsms['tvalue']
res_bvalue = target_dsms['bvalue']

num_perms = 500

num_voxs = fds.shape[1]
nulls = np.zeros([num_perms,4])
for i in range(num_perms):
    print 'Permutation ',i
    shuffled_inds = DMshuffle_custom(res_stim_id)
    p_res_stim_id = res_stim_id[shuffled_inds]
    p_res_fvalue = res_fvalue[shuffled_inds]
    p_res_tvalue = res_tvalue[shuffled_inds]
    p_res_bvalue = res_bvalue[shuffled_inds]
    p_res_day = res_day[shuffled_inds]
    btwn_run_inds = np.where(p_res_day.samples == 1)[0]
    
    model_dsms = np.column_stack((p_res_stim_id, p_res_fvalue.samples.reshape(-1), \
                              p_res_tvalue.samples.reshape(-1), p_res_bvalue.samples.reshape(-1)))
    
    rsa_reg = rsa.Regression(model_dsms, pairwise_metric='correlation', keep_pairs=btwn_run_inds)
    
    sl_rsa_reg = sphere_searchlight(rsa_reg, radius=3, center_ids=vox2use)
    sl_fmri_value = sl_rsa_reg(fds)
    
    nulls[i,:] = sl_fmri_value.samples[:-1].reshape(-1)