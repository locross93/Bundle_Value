# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:53:02 2021

@author: locro
"""

from mvpa2.suite import *
from mvpa2.base.hdf5 import h5load, h5save
#from mvpa2.datasets.mri import map2nifti
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/Users/locro/Documents/Bundle_Value/mvpa/")
import os
os.chdir("/Users/locro/Documents/Bundle_Value/mvpa/")
import mvpa_utils
from mvpa2.misc.neighborhood import Sphere
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def get_voxel_sphere(center_coords, voxel_indices):
    radius = 4
    sphere = Sphere(radius)
    all_coords = sphere(center_coords)
    inds2use = []
    for coords in all_coords:
        coords = np.array(coords)
        temp_ind = np.where((voxel_indices == coords).all(axis=1))[0]
        if len(temp_ind) > 0:
            assert len(temp_ind) == 1
            inds2use.append(temp_ind[0])
            
    return inds2use

bundle_path = '/Users/locro/Documents/Bundle_Value/'

decode_choice = h5load(bundle_path+'mvpa/analyses/sub111/decode_choice')
decode_choice_logit = h5load(bundle_path+'mvpa/analyses/sub111/decode_choice_logit')
decode_choice_logit_sk = h5load(bundle_path+'mvpa/analyses/sub111/decode_choice_logit_sk')

best_voxs_dc = np.flip(np.argsort(decode_choice))

for vox in best_voxs_dc[:10]:
    print vox
    print decode_choice[vox]
    print decode_choice_logit[vox]
    print decode_choice_logit_sk[vox]
    print '\n'
    
subj = '111'

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle'] 

#which ds to use and which mask to use
glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii'
mask_name = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'

#make targets with mvpa utils    
fds = mvpa_utils.make_targets_choice(subj, glm_ds_file, mask_name, conditions, system='win')

#balance dataset, subsample from bigger class
choice = fds.targets
ref_choices = np.where(choice == 0)[0]
item_choices = np.where(choice == 1)[0]
if ref_choices.shape[0] > item_choices.shape[0]:
    num_examples = item_choices.shape[0]
    ref_choices_sub = np.sort(np.random.choice(ref_choices, num_examples, replace=False))
    inds2use = np.sort(np.concatenate((ref_choices_sub, item_choices)))
    fds = fds[inds2use,:]
elif item_choices.shape[0] > ref_choices.shape[0]:
    num_examples = ref_choices.shape[0]
    item_choices_sub = np.sort(np.random.choice(item_choices, num_examples, replace=False))
    inds2use = np.sort(np.concatenate((ref_choices, item_choices_sub)))
    fds = fds[inds2use,:]
    
y = fds.targets
cv_groups = fds.chunks
voxel_indices = fds.fa.voxel_indices

clf = PLR(lm=1e-2)
#clf = LogisticRegression(C=1e-2)
clf = LogisticRegression(C=1e-5, max_iter=1000)
clf = SVC(C=1e-5, kernel='linear')
clf = LinearSVC(C=1e-10)

scores_per_voxel = np.zeros([10])

count = 0
for vox in best_voxs_dc[:10]:
    vox_coord = voxel_indices[vox,:]
    sphere_inds = get_voxel_sphere(vox_coord, voxel_indices)
    fds_sphere = fds[:,sphere_inds]
    
    #scores_per_voxel[vox] = mvpa_utils_lab.roiClass_1Ss(fds_sphere, mask_name)
    #scores_per_voxel[vox] = mvpa_utils_lab.roiClass_1Ss(fds_sphere, mask_name, clf)
    
    X = fds_sphere.samples
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    #cv = CrossValidation(clf, NFoldPartitioner(), enable_ca=['stats'], errorfx=lambda p, t: np.mean(p == t))
    #scores_per_voxel[count] = np.mean(cv(fds_sphere))
    
    scores_per_voxel[count] = np.mean(cross_val_score(clf, X, y, groups=cv_groups, cv=15))
    print decode_choice[vox]
    print decode_choice_logit[vox]
    print vox,scores_per_voxel[count]
    print '\n'
    
    count+=1