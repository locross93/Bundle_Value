#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:23:08 2021

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
from nipy import load_image
from nipy.labs import statistical_mapping

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj = '109'

analysis_name = 'cross_decoding_rel_value'

s2s_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_s2s'
brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
myimg = load_image(s2s_file+'.nii.gz')
mask = load_image(brain_mask)
clusters = statistical_mapping.cluster_stats(myimg, mask, 0.2, height_control='none', cluster_th=10)
cluster_table = nilearn.reporting.get_clusters_table(myimg, 0.2, 10)

def mm2vox_coord_trans(coord):
    new_coord = np.zeros(3)
    #X
    new_coord[0] = (90 - coord[0])/2.5
    #Y
    new_coord[1] = (coord[1] + 126)/2.5
    #Z
    new_coord[2] = (coord[2] + 72)/2.5
    
    return new_coord
    
    
voxel_indices = fds.fa.voxel_indices

count = 0
for cluster in clusters[0]:
    cluster_scores = []
    for coord in cluster['maxima']:
        new_coord = mm2vox_coord_trans(coord)
        temp_ind = np.where((voxel_indices == new_coord).all(axis=1))[0]
        cluster_scores.append(scores_s2s[temp_ind])
    print 'Cluster #',count
    print np.mean(cluster_scores)
    print '\n'
    
    count+=1
    