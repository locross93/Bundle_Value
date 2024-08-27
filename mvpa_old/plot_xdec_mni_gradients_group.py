#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:31:52 2021

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
import nibabel as nib
import nilearn

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

plot = False

s2s_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/S2S/spmT_0001_pfc.nii.gz'
s2b_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/S2B/spmT_0001_pfc.nii.gz'
b2b_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/B2B/spmT_0001_pfc.nii.gz'
b2s_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/B2S/spmT_0001_pfc.nii.gz'

scores_s2s = nib.load(s2s_file).get_fdata()
scores_s2b = nib.load(s2b_file).get_fdata()
scores_b2b = nib.load(b2b_file).get_fdata()
scores_b2s = nib.load(b2s_file).get_fdata()

nz_inds = np.where(scores_s2s != 0)
num_nz_inds = len(nz_inds[0])

bundle_pref = []
x_inds = []
y_inds = []
z_inds = []
for i in range(num_nz_inds):
    x = nz_inds[0][i]
    y = nz_inds[1][i]
    z = nz_inds[2][i]
    
    #bundle_pref.append(scores_b2b[x,y,z] - scores_b2s[x,y,z])
    #bundle_pref.append(scores_s2b[x,y,z] - scores_s2s[x,y,z])
    bundle_pref.append(scores_b2b[x,y,z] - scores_s2s[x,y,z])
    x_inds.append(x)
    y_inds.append(y)
    z_inds.append(z)

#take z inds for dorsal to ventral gradient
corr_z = pearsonr(bundle_pref, z_inds)[0]
print 'ventral/dorsal:',round(corr_z,3)

if plot:
    ax = plt.axes()
    plt.scatter(z_inds, bundle_pref)
    plt.text(0.8,0.15,'Correlation: '+'%.3f'%corr_z, horizontalalignment='center',
    verticalalignment='center', transform=ax.transAxes)
    plt.xlabel('Z Index')
    plt.ylabel('Bundle Score - Single Item Score')
    plt.title('Dorsal/Ventral Gradient Sub'+subj, fontsize=18)
    plt.show()

#take y inds too for posterior - anterior
corr_y = pearsonr(bundle_pref, y_inds)[0]
print 'posterior/anterior:',round(corr_y,3)

if plot:
    ax = plt.axes()
    plt.scatter(y_inds, bundle_pref)
    plt.text(0.8,0.15,'Correlation: '+'%.3f'%corr_y, horizontalalignment='center',
    verticalalignment='center', transform=ax.transAxes)
    plt.xlabel('Y Index')
    plt.ylabel('Bundle Score - Single Item Score')
    plt.title('Posterior/Anterior Gradient Sub'+subj, fontsize=18)
    plt.show()

#take x inds and center at x=36 for medial - lateral
middle_x_ind = 90
laterality = np.absolute(np.array(x_inds) - middle_x_ind)
corr_lat = pearsonr(bundle_pref, laterality)[0]
print 'medial/lateral:',round(corr_lat,3)

if plot:
    ax = plt.axes()
    plt.scatter(y_inds, bundle_pref)
    plt.text(0.8,0.15,'Correlation: '+'%.3f'%corr_lat, horizontalalignment='center',
    verticalalignment='center', transform=ax.transAxes)
    plt.xlabel('Laterality')
    plt.ylabel('Bundle Score - Single Item Score')
    plt.title('Medial/Lateral Gradient Sub'+subj, fontsize=18)
    plt.show()
    
print '\n'