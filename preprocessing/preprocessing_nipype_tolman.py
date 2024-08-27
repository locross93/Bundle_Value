#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:50:28 2018

@author: logancross
"""

import math
import os
import nipype
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.algorithms.modelgen as model   # model generation
import nipype.algorithms.rapidart as ra      # artifact detection
import nibabel as nib
import nipype.interfaces.c3 as c3
import nipype.interfaces.ants as ants
#from nipype.interfaces.fsl import fix, ICA_AROMA       # fix automated denoising
from nipype.interfaces.utility import Function
from IPython.display import Image
#from nilearn import plotting
import sys
import time

start_time = time.time()

reload(sys)
sys.setdefaultencoding('utf8')

# remove square brackets (['filename'] -> 'filename')
def unlist(mylist):
    #print "Mylist"
    #print mylist
    r = mylist[0]
    #print r
    return r

# remove square brackets (['filename'] -> 'filename') long type
def unlist_long(mylist):
    #print "Mylist"
    #print mylist
    r = mylist[0]
    #print r
    return long(r)

def reverse(mylist):
    mylist.reverse()
    return mylist

# 7/23/2018 Determine resolution of fMRI scan volume(s)
def get_voxel_size(f):
    from nibabel import load
    in_file = f
    if isinstance(f, list):
        in_file = f[0]
    img_header = load(in_file).get_header().get_zooms()
    return float(img_header[0])

#7/24/2018 WG Create option string for highpass filtering
def create_highpass_op_string(f):
    from nibabel import load
    in_file = f
    if isinstance(f, list):
        in_file = f[0]
    img_header = load(in_file).get_header().get_zooms()
    print 'In highpass function'
    print img_header[3]
    print (100 / (2 * img_header[3]))
    # wolfgangs code had a 2 
    #return "-bptf %s -1" % (100 / (2 * img_header[3]))
    return "-bptf %s -1" % (100 / (img_header[3]))

#7/24/2018 WG Generate option string to create mask for functional
def to_opt_string(out_stat):
    r = []
    for i in xrange(len(out_stat)):
        r.append("-thr %s -bin -fillh -dilF" % (float(out_stat[i][0]) + .5))
    #print out_stat
    #print r
    return r

def to_bp_string(f):
    in_file = f
    if isinstance(f, list):
        in_file = f[0]
    TR = 1.1
    r = '-bptf '+str((100/(TR * 2))) + ' -1 -add ' + in_file

    return r

def get_first(mylist):
    #print "get_first"
    r = mylist[0]
    #print mylist
    return r

def get_sec(mylist):
    #print "get_sec"
    r = mylist[1]
    #print mylist
    return r

def get_sec_tfm(in_list):
    #print "get_sec_tfm"
    #print in_list
    out_list = []
    for i in xrange(len(in_list)):
        out_list.append([in_list[i][0]])
    return out_list 

def getthreshop(thresh):
    return '-thr %.10f -Tmin -bin'%(0.1*thresh[0][1])

def getinormscale(medianvals):
    return ['-mul %.10f'%(10000./val) for val in medianvals]


# Optional: Use the following lines to increase verbosity of output
#nipype.config.set('logging', 'workflow_level',  'DEBUG')
#nipype.config.set('logging', 'interface_level', 'DEBUG')
#nipype.logging.update_logging(nipype.config)

#Prevent script from having to rerun unnecessary outputs
nipype.config.set('execution','remove_unnecessary_outputs', False)

#Log the outputs
nipype.config.set('logging','log_to_file',True)

# Run the test: Increase verbosity parameter for more info
#nipype.test(verbose=1)

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

data_dir = '/home/lcross/Bundle_Value/preprocessing/bidsdata/'
fmri_dir = '/home/lcross/Bundle_Value/code/'

# create preproc workflow
preproc = pe.Workflow(name='preproc')
preproc.base_dir = os.path.abspath('/home/lcross/Bundle_Value/preprocessing/')

preproc.config['execution']['crashfile_format'] = 'txt'

# WG 7/12/2018 Defining templates of files to load for anatomical (T1 and T2 normalized), field maps (positive and negative), BOLD, and BOLD_Ref
templates={'t1':'sub-{subID}/ses-day{dayID}/anat/sub-{subID}_ses-day{dayID}_T1w_run-02*.nii.gz',
           't2':'sub-{subID}/ses-day{dayID}/anat/sub-{subID}_ses-day{dayID}_*T2*.nii.gz',
           'func':'sub-{subID}/ses-day{dayID}/func/sub-{subID}_ses-day{dayID}_*run-0{runID}_bold.nii.gz',
          'func_ref':'sub-{subID}/ses-day{dayID}/func/sub-{subID}_ses-day{dayID}_*run-0{runID}_bold_SBRef.nii.gz',
          'fm_pos':'sub-{subID}/ses-day{dayID}/fmap/sub-{subID}_ses-day{dayID}_*run-0{runID}*AP*.nii.gz',
          'fm_neg':'sub-{subID}/ses-day{dayID}/fmap/sub-{subID}_ses-day{dayID}_*run-0{runID}*PA*.nii.gz'}

# scan parameters
TR = 1.1
#fugue_dwell_time = 2.1 * 10**-4
#fugue_dwell_time = 6.8 * 10**-4
fugue_dwell_time = 0.000393121

#WG 7/12/2018 Creating node to load files and specifying necessary options.
datasource = pe.Node(nio.SelectFiles(templates),
                    name='selectfiles')
datasource.inputs.base_directory = data_dir
datasource.inputs.subID='102' #WG 7/12/2018 Can modify later to allow for all subject IDs.
datasource.inputs.dayID=['1','2','3']
datasource.inputs.runID=['1','2','3','4','5']
datasource.inputs.sort_filelist = True

datasource.run().outputs #Demonstrates output of the node above as method of debugging

#WG 7/12/2018 Sets up node to feed respective image types into different sub-pipelines
inputnode = pe.Node(interface=util.IdentityInterface(fields=['func', 'func_ref', 't1', 't2', 'fm_pos', 'fm_neg']), name='inputnode')

#WG 7/12/2018 Connects each output from the SelectFile node (datasource) to the IdentityInterface node (inputnode)
preproc.connect(datasource, 'func', inputnode, 'func')
preproc.connect(datasource, 'func_ref', inputnode, 'func_ref')
preproc.connect(datasource, 't1', inputnode, 't1')
preproc.connect(datasource, 't2', inputnode, 't2')
preproc.connect(datasource, 'fm_pos', inputnode, 'fm_pos')
preproc.connect(datasource, 'fm_neg', inputnode, 'fm_neg')

###########################
###PREPROCESS ANATOMICALS##
###########################

# reorient t1 to standard
T1_reoriented = pe.Node(interface=fsl.Reorient2Std(output_type = "NIFTI_GZ"), name='T1_reoriented') 
preproc.connect(inputnode, 't1', T1_reoriented, 'in_file')

# reorient t2 to standard
T2_reoriented = pe.Node(interface=fsl.Reorient2Std(output_type = "NIFTI_GZ"), name='T2_reoriented') 
preproc.connect(inputnode, 't2', T2_reoriented, 'in_file')

# coreg t2 to t1
t2tot1 = pe.Node(interface=fsl.FLIRT(dof=6, output_type = "NIFTI_GZ"), name='t2tot1')
preproc.connect(T2_reoriented, 'out_file', t2tot1, 'in_file')
preproc.connect(T1_reoriented, 'out_file', t2tot1, 'reference')

# # downsample caltech atlas to t1/t2 resolution
# WG 7/12/2018 Currently using local copies of ident.mat and reference atlases
# WG 7/23/2018 May want to switch to using appropriate atlas for cortical areas
downsample_atlas_t1 = pe.Node(interface=fsl.FLIRT(dof=6, output_type = "NIFTI_GZ", in_file=fmri_dir+'CIT_brains/CIT168_T1w_700um_MNI.nii.gz', reference=fmri_dir+'CIT_brains/CIT168_T1w_700um_MNI.nii.gz', in_matrix_file='/usr/share/fsl/5.0/etc/flirtsch/ident.mat'), name='downsample_atlas_t1')
preproc.connect(inputnode, ('t1', get_voxel_size), downsample_atlas_t1, 'apply_isoxfm')

downsample_atlas_t2 = pe.Node(interface=fsl.FLIRT(dof=6, output_type = "NIFTI_GZ", in_file=fmri_dir+'CIT_brains/CIT168_T1w_700um_MNI.nii.gz', reference=fmri_dir+'CIT_brains/CIT168_T1w_700um_MNI.nii.gz', in_matrix_file='/usr/share/fsl/5.0/etc/flirtsch/ident.mat'), name='downsample_atlas_t2')
preproc.connect(inputnode, ('t2', get_voxel_size), downsample_atlas_t2, 'apply_isoxfm')

# # create mask file for atlas
mask_atlas = pe.Node(interface=fsl.ImageMaths(op_string = '-thr .001 -bin -dilF', suffix = '_mask'), name='mask_atlas')
preproc.connect(downsample_atlas_t1, 'out_file', mask_atlas, 'in_file')

# skull strip t1
#7/24/2018 LC May need to modify frac subject by subject 
skullstrip = pe.Node(interface=fsl.BET(mask = True, frac=0.4, reduce_bias = True),
                     name = 'stripstruct')
preproc.connect(T1_reoriented, 'out_file', skullstrip, 'in_file')

inflate_mask = pe.Node(interface=fsl.ImageMaths(op_string = '-dilF', suffix = '_dilF'), name='inflate_mask')
preproc.connect(skullstrip, 'mask_file', inflate_mask, 'in_file')

# apply t1 mask to t2
maskT2 = pe.Node(interface=fsl.ImageMaths(op_string = '-mas', suffix = '_bet'), name='maskT2')
preproc.connect(t2tot1, 'out_file', maskT2, 'in_file')
preproc.connect(inflate_mask, 'out_file', maskT2, 'in_file2')

#7/23/2018 WG Reapply dilated mask to T1, such that T1 and T2 have same brain mask.
maskT1 = pe.Node(interface=fsl.ImageMaths(op_string = '-mas', suffix = '_bet'), name='maskT1')
preproc.connect(T1_reoriented, 'out_file', maskT1, 'in_file')
preproc.connect(inflate_mask, 'out_file', maskT1, 'in_file2')

# affine co-reg (flirt) of T1 to caltech atlas
init_t1_to_atlas_coreg = pe.Node(interface=fsl.FLIRT(dof=6, output_type = "NIFTI_GZ"), name='init_t1_to_atlas_coreg')
preproc.connect(maskT1, 'out_file', init_t1_to_atlas_coreg, 'in_file')
preproc.connect(downsample_atlas_t1, 'out_file', init_t1_to_atlas_coreg, 'reference')

# convert transform matrix to ants format (ITK)
fsl2ras = pe.Node(interface=c3.C3dAffineTool(fsl2ras = True, itk_transform=True), name='fsl2ras')
preproc.connect(downsample_atlas_t1, 'out_file', fsl2ras, 'reference_file')
preproc.connect(maskT1, 'out_file', fsl2ras, 'source_file')
preproc.connect(init_t1_to_atlas_coreg, 'out_matrix_file', fsl2ras, 'transform_file')

# merge fixed and moving images into list
merge_fixed = pe.Node(interface=util.Merge(2, axis='hstack'), name='merge_fixed')
preproc.connect(downsample_atlas_t1, 'out_file', merge_fixed, 'in1')
preproc.connect(downsample_atlas_t2, 'out_file', merge_fixed, 'in2')
merge_moving = pe.Node(interface=util.Merge(2, axis='hstack'), name='merge_moving')
preproc.connect(maskT1, 'out_file', merge_moving, 'in1')
preproc.connect(maskT2, 'out_file', merge_moving, 'in2')

structs_to_atlas_coreg_input = pe.Node(interface=util.IdentityInterface(fields=['fixed_image', 'moving_image', 'fixed_image_mask', 'moving_image_mask', 'initial_moving_transform']), name='structs_to_atlas_coreg_input')
preproc.connect(merge_fixed, ('out', unlist), structs_to_atlas_coreg_input, 'fixed_image')
preproc.connect(merge_moving, ('out', unlist), structs_to_atlas_coreg_input, 'moving_image')
preproc.connect(mask_atlas, 'out_file', structs_to_atlas_coreg_input, 'fixed_image_mask')
preproc.connect(inflate_mask, 'out_file', structs_to_atlas_coreg_input, 'moving_image_mask')
preproc.connect(fsl2ras, 'itk_transform', structs_to_atlas_coreg_input, 'initial_moving_transform')

# diffeomorphic mapping of t2/t1 to caltech atlas
structs_to_atlas_coreg = pe.Node(interface=ants.Registration(dimension=3, transforms=['SyN'], metric=[['CC'] * 2], radius_or_number_of_bins=[[4] * 2], metric_weight = [[.5] * 2], transform_parameters=[[.1,3.0,0.0]], number_of_iterations=[[100,100,70,50,20]], convergence_threshold=[1.e-6], convergence_window_size=[10], smoothing_sigmas=[[5.0,3.0,2.0,1.0,0.0]], shrink_factors=[[10,6,4,2,1]], use_histogram_matching=True, interpolation='Linear', invert_initial_moving_transform=False, sampling_strategy = [['Random']*2], sampling_percentage = [[0.05]*2]), name='structs_to_atlas_coreg')
#structs_to_atlas_coreg = pe.Node(interface=ants.Registration(dimension=3, transforms=['SyN'], metric=[['CC'] * 2], radius_or_number_of_bins=[[4] * 2], metric_weight = [[.5] * 2], transform_parameters=[[.1,3.0,0.0]], number_of_iterations=[[100,100,70,50,20]], convergence_threshold=[1.e-6], convergence_window_size=[10], smoothing_sigmas=[[5.0,3.0,2.0,1.0,0.0]], shrink_factors=[[10,6,4,2,1]], use_histogram_matching=True, interpolation='Linear', invert_initial_moving_transform=False, sampling_strategy = [['Random']*2], sampling_percentage = [[0.05]*2], verbose=True), name='structs_to_atlas_coreg')
preproc.connect(structs_to_atlas_coreg_input, 'fixed_image', structs_to_atlas_coreg, 'fixed_image')
preproc.connect(structs_to_atlas_coreg_input, 'moving_image', structs_to_atlas_coreg, 'moving_image')
preproc.connect(structs_to_atlas_coreg_input, 'fixed_image_mask', structs_to_atlas_coreg, 'fixed_image_mask')
preproc.connect(structs_to_atlas_coreg_input, 'moving_image_mask', structs_to_atlas_coreg, 'moving_image_mask')
preproc.connect(structs_to_atlas_coreg_input, 'initial_moving_transform', structs_to_atlas_coreg, 'initial_moving_transform')

structs_to_atlas_coreg_output = pe.Node(interface=util.IdentityInterface(fields=['forward_transforms', 'warped_image', 'reverse_transforms', 'inverse_warped_image']), name='structs_to_atlas_coreg_output')
preproc.connect(structs_to_atlas_coreg, 'forward_transforms', structs_to_atlas_coreg_output, 'forward_transforms')
preproc.connect(structs_to_atlas_coreg, 'warped_image', structs_to_atlas_coreg_output, 'warped_image')
preproc.connect(structs_to_atlas_coreg, 'reverse_transforms', structs_to_atlas_coreg_output, 'reverse_transforms')
preproc.connect(structs_to_atlas_coreg, 'inverse_warped_image', structs_to_atlas_coreg_output, 'inverse_warped_image')

#apply transformation from anat all the way to caltech atlas (in anat resolution)
t1_to_ds_atlas = pe.Node(interface=ants.WarpImageMultiTransform(use_nearest = True), name = 't1_to_ds_atlas')
preproc.connect(maskT1, 'out_file', t1_to_ds_atlas, 'input_image')
preproc.connect(downsample_atlas_t1, 'out_file', t1_to_ds_atlas, 'reference_image')
preproc.connect(structs_to_atlas_coreg_output, ('forward_transforms', reverse), t1_to_ds_atlas, 'transformation_series')

#preproc.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
#
#time_dif = time.time() - start_time
#print 'Finished script at ',time_dif

###########################
###PREPROCESS FUNCTIONALS##
###########################

start_time = time.time()

#######################
# MCFLIRT, skull strip, then high pass filter as would be done in FEAT/MELODIC

# use SB reference for alternative reference in MCFLIRT, so scans are aligned to an independent, higher SNR scan 
# n4 bias correction of reference
n4biasCorrMbRef = pe.MapNode(interface=ants.N4BiasFieldCorrection(dimension = 3, 
                bspline_fitting_distance = 300, shrink_factor = 3, n_iterations = [50,50,30,20], save_bias = True)
                , name='n4biasCorrMbRef', iterfield=['input_image'])
preproc.connect(inputnode, 'func_ref', n4biasCorrMbRef, 'input_image')

#skull strip reference
stripref = pe.MapNode(interface=fsl.BET(mask = False, frac=0.3, robust = True),
                     name = 'stripref', iterfield=['in_file'])
preproc.connect(n4biasCorrMbRef, 'output_image', stripref, 'in_file')

# Motion Correction
motion_correct = pe.MapNode(interface=fsl.MCFLIRT(save_mats = True, save_plots = True, interpolation = 'spline'), name='motion_correct', iterfield = ['in_file','ref_file'])
preproc.connect(inputnode, 'func', motion_correct, 'in_file')
preproc.connect(stripref, 'out_file', motion_correct, 'ref_file')

# plot est motion params
plot_motion = pe.MapNode(interface=fsl.PlotMotionParams(in_source='fsl'),
                        name='plot_motion',
                        iterfield=['in_file'])
plot_motion.iterables = ('plot_type', ['rotations', 'translations'])
preproc.connect(motion_correct, 'par_file', plot_motion, 'in_file')

# get a mean func for skull stripping
meanfunc_filt = pe.MapNode(interface=fsl.ImageMaths(op_string = '-Tmean', suffix='_mean'), name='meanfunc_filt', iterfield = ['in_file'])
preproc.connect(motion_correct, 'out_file', meanfunc_filt, 'in_file')

# skull strip functionals
#7/24/2018 LC May need to modify frac subject by subject 
stripfunc = pe.MapNode(interface=fsl.BET(mask = True, frac=0.3, robust = False),
                     name = 'stripfunc', iterfield = ['in_file'])
preproc.connect(meanfunc_filt, 'out_file', stripfunc, 'in_file')


# apply BET mask to whole set of functionals
betfunc = pe.MapNode(interface=fsl.ImageMaths(op_string = '-mas', suffix = '_bet'), name='betfunc', iterfield = ['in_file','in_file2'])
preproc.connect(motion_correct, 'out_file', betfunc, 'in_file')
preproc.connect(stripfunc, 'mask_file', betfunc, 'in_file2')

# Determine the 2nd and 98th percentile intensities of each functional run
getthresh = pe.MapNode(interface=fsl.ImageStats(op_string='-p 2 -p 98'),
                       iterfield = ['in_file'],
                       name='getthreshold')
preproc.connect(betfunc, 'out_file', getthresh, 'in_file')

#Threshold the first run of the functional data at 10% of the 98th percentile
threshold = pe.MapNode(interface=fsl.ImageMaths(out_data_type='char',
                                             suffix='_thresh'),
                    iterfield = ['in_file'],
                    name='threshold')
preproc.connect(betfunc, 'out_file', threshold, 'in_file')
preproc.connect(getthresh, ('out_stat', getthreshop), threshold, 'op_string')

# Determine the median value of the functional runs using the mask
medianval = pe.MapNode(interface=fsl.ImageStats(op_string='-k %s -p 50'),
                       iterfield = ['in_file','mask_file'],
                       name='medianval')
preproc.connect(betfunc, 'out_file', medianval, 'in_file')
preproc.connect(threshold, 'out_file', medianval, 'mask_file')

# Dilate the mask. This is the final mask for the level 1.
dilatemask = pe.MapNode(interface=fsl.ImageMaths(suffix='_dil',
                                              op_string='-dilF'),
                    iterfield=['in_file'],
                     name='dilatemask')

preproc.connect(threshold, 'out_file', dilatemask, 'in_file')

# Mask the motion corrected functional runs with the dilated mask
prefiltered_func_data_thresh = pe.MapNode(interface=fsl.ImageMaths(suffix='_mask',
                                                op_string='-mas'),
                       iterfield=['in_file','in_file2'],
                       name='prefiltered_func_data_thresh')

preproc.connect(betfunc, 'out_file', prefiltered_func_data_thresh, 'in_file')
preproc.connect(dilatemask, 'out_file', prefiltered_func_data_thresh, 'in_file2')

# Scale each volume of the run so that the median value of the run is set to 10000 - intensity normalization
intnorm = pe.MapNode(interface=fsl.ImageMaths(suffix='_intnorm'),
                     iterfield=['in_file','op_string'],
                     name='intnorm')
preproc.connect(prefiltered_func_data_thresh, 'out_file', intnorm, 'in_file')
preproc.connect(medianval, ('out_stat', getinormscale), intnorm, 'op_string')

# Create tempMean
tempMean = pe.MapNode(interface=fsl.ImageMaths(op_string='-Tmean',
                                                suffix='_mean'),
                       iterfield=['in_file'],
                       name='tempMean')
preproc.connect(intnorm, 'out_file', tempMean, 'in_file')

# Perform temporal highpass filtering on the data. This is the same as filtered_func_data in FSL output
highpass = pe.MapNode(interface=fsl.ImageMaths(op_string= '-bptf %d -1 -add'%(100/(2*TR)), suffix='_tempfilt'),
                      iterfield=['in_file','in_file2'],
                      name='highpass')
preproc.connect(tempMean, 'out_file', highpass, 'in_file2')
preproc.connect(intnorm, 'out_file', highpass, 'in_file')

#######################
# put functionals and fieldmaps in the same voxel space
# Transform fieldmap to reference space then to functional space 

#######################
# get transformation matrix for fm -> ref

# merge fm_pos and fm_neg into one list
mergenode = pe.Node(interface=util.Merge(2, axis='hstack'), name='merge', iterfield=['in1','in2'])
preproc.connect(inputnode, 'fm_pos', mergenode, 'in1')
preproc.connect(inputnode, 'fm_neg', mergenode, 'in2')

# concatenate pos and neg image into one for topup
mergeposneg = pe.MapNode(interface=fsl.Merge(dimension='t'), name='mergeposneg', iterfield=['in_files'])
preproc.connect(mergenode, 'out',  mergeposneg, 'in_files')

# create fieldmap and field_cof
# subtle change to configuration file, to use subsampling of 5, instead of 2, to get started. This is necessary because of the un-even number of slices (35)
topup = pe.MapNode(interface=fsl.TOPUP(config='b02b0.cnf', encoding_file = "/home/lcross/Bundle_Value/code/preprocessing/fieldmaps_datain_Bundles.txt", output_type = "NIFTI_GZ"), name='topup', iterfield=['in_file'])
preproc.connect(mergeposneg, 'merged_file', topup, 'in_file')

# convert fielmaps to rad/s
fm_to_fmRads = pe.MapNode(interface=fsl.ImageMaths(op_string = ('-mul %s' % (math.pi * 2)), suffix = '_mask'), name='fm_fmRads', iterfield=['in_file'])
preproc.connect(topup, 'out_field', fm_to_fmRads, 'in_file')

# co-register mbRef to merged fm epis
mbRef_to_corrected_fm_epi = pe.MapNode(interface=fsl.FLIRT(dof=6, output_type = "NIFTI_GZ"), name='mbRef_to_corrected_fm_epi', iterfield=['in_file', 'reference'])
preproc.connect(n4biasCorrMbRef, 'output_image', mbRef_to_corrected_fm_epi, 'in_file')
preproc.connect(topup, 'out_corrected', mbRef_to_corrected_fm_epi, 'reference')

# unwarp co-registered mb reference image - DWELL TIME varies by sequence
unwarp = pe.MapNode(interface=fsl.FUGUE(save_shift = True, unwarp_direction = 'y', output_type = "NIFTI_GZ", dwell_time=fugue_dwell_time), name='unwarp', iterfield=['fmap_in_file', 'in_file'])
preproc.connect(fm_to_fmRads, 'out_file', unwarp, 'fmap_in_file')
preproc.connect(mbRef_to_corrected_fm_epi, 'out_file', unwarp, 'in_file')

# invert xfm - since we used the magnitude as a reference we need to invert the transformation matrix before applying it
invert_tfm = pe.MapNode(interface=fsl.ConvertXFM(output_type = "NIFTI_GZ", invert_xfm = True), name='invert_tfm', iterfield=['in_file'])
preproc.connect(mbRef_to_corrected_fm_epi, 'out_matrix_file', invert_tfm, 'in_file')

# rev apply mbRef to topup trans mat to out_shift_file
shift_file2mbRef = pe.MapNode(interface=fsl.FLIRT(apply_xfm = True, output_type = "NIFTI_GZ"), name='shift_file2mbRef', iterfield=['in_file','reference','in_matrix_file'])
preproc.connect(unwarp, 'shift_out_file', shift_file2mbRef, 'in_file')
preproc.connect(inputnode, 'func_ref', shift_file2mbRef, 'reference')
preproc.connect(invert_tfm, 'out_file', shift_file2mbRef, 'in_matrix_file')

########################
## get transformation matrix for ref -> func

#extract mean func - get func scans from hihgpass filter output
meanfunc = pe.MapNode(interface=fsl.ImageMaths(op_string = '-Tmean', suffix='_mean'), name='meanfunc', iterfield = ['in_file'])
preproc.connect(stripfunc, 'out_file', meanfunc, 'in_file')

# get intensity range of mean func
func_intensity_range = pe.MapNode(interface=fsl.ImageStats(op_string= '-r'), name='func_intensity_range', iterfield=['in_file'])
preproc.connect(meanfunc, 'out_file', func_intensity_range, 'in_file')

# create func mask
func_mask = pe.MapNode(interface=fsl.ImageMaths(suffix='_mask'), name='func_mask', iterfield=['in_file', 'op_string'])
preproc.connect(meanfunc, 'out_file', func_mask, 'in_file')
preproc.connect(func_intensity_range, ('out_stat', to_opt_string), func_mask, 'op_string')

# n4 bias correction of mean
n4biasCorrMFunc = pe.MapNode(interface=ants.N4BiasFieldCorrection(dimension = 3, bspline_fitting_distance = 300, shrink_factor = 3, n_iterations = [50,50,30,20], save_bias = True), name='n4biasCorrMFunc', iterfield=['input_image', 'mask_image'])
preproc.connect(meanfunc, 'out_file', n4biasCorrMFunc, 'input_image')
preproc.connect(func_mask, 'out_file', n4biasCorrMFunc, 'mask_image')

# co-register mean functional to biased corrected MbRef 
mFunc_to_mbref = pe.MapNode(interface=fsl.FLIRT(dof=6, output_type = "NIFTI_GZ"), name='mFunc_to_mbref', iterfield=['in_file', 'reference'])
preproc.connect(n4biasCorrMFunc, 'output_image', mFunc_to_mbref, 'in_file')
preproc.connect(n4biasCorrMbRef, 'output_image', mFunc_to_mbref, 'reference')

#######################
# apply transformation fm -> ref -> func 

# concat transformation file for mFunc to mbRef and mbRef to FM
concat_tfm_mFunc_mbRef_FM = pe.MapNode(interface=fsl.ConvertXFM(output_type = "NIFTI_GZ", concat_xfm = True), name='concat_tfm_mFunc_mbRef_FM', iterfield=['in_file', 'in_file2'])
preproc.connect(mbRef_to_corrected_fm_epi, 'out_matrix_file', concat_tfm_mFunc_mbRef_FM, 'in_file')
preproc.connect(mFunc_to_mbref, 'out_matrix_file', concat_tfm_mFunc_mbRef_FM, 'in_file2')

# invert concatenated transformation matrix
invert_tfm_mFunc_mbRef_FM = pe.MapNode(interface=fsl.ConvertXFM(output_type = "NIFTI_GZ", invert_xfm = True), name='invert_tfm_mFunc_mbRef_FM', iterfield=['in_file'])
preproc.connect(concat_tfm_mFunc_mbRef_FM, 'out_file', invert_tfm_mFunc_mbRef_FM, 'in_file')

# transform fm from FM to mFunc
shift_file2mFunc = pe.MapNode(interface=fsl.FLIRT(apply_xfm = True, output_type = "NIFTI_GZ"), name='shift_file2mFunc', iterfield=['in_file','reference','in_matrix_file'])
preproc.connect(unwarp, 'shift_out_file', shift_file2mFunc, 'in_file')
preproc.connect(inputnode, 'func', shift_file2mFunc, 'reference')
preproc.connect(invert_tfm_mFunc_mbRef_FM, 'out_file', shift_file2mFunc, 'in_matrix_file')

#######################
# UNWARP

# unwarp mbRef in place
unwarp_mbRef = pe.MapNode(interface=fsl.FUGUE(save_shift = True, unwarp_direction = 'y', output_type = "NIFTI_GZ", dwell_time=fugue_dwell_time), name='unwarp_mbRef', iterfield=['shift_in_file','in_file'])
preproc.connect(shift_file2mbRef, 'out_file', unwarp_mbRef, 'shift_in_file')
preproc.connect(n4biasCorrMbRef, 'output_image', unwarp_mbRef, 'in_file')

# unwarp meanFunc in place
unwarp_meanfunc = pe.MapNode(interface=fsl.FUGUE(save_shift = True, unwarp_direction = 'y', output_type = "NIFTI_GZ", dwell_time=fugue_dwell_time), name='unwarp_meanfunc', iterfield=['shift_in_file','in_file'])
preproc.connect(shift_file2mFunc, 'out_file', unwarp_meanfunc, 'shift_in_file')
preproc.connect(meanfunc, 'out_file', unwarp_meanfunc, 'in_file')

# unwarp func in place
unwarp_func = pe.MapNode(interface=fsl.FUGUE(save_shift = True, unwarp_direction = 'y', output_type = "NIFTI_GZ", dwell_time=fugue_dwell_time), name='unwarp_func', iterfield=['shift_in_file','in_file'])
preproc.connect(shift_file2mFunc, 'out_file', unwarp_func, 'shift_in_file')
preproc.connect(highpass, 'out_file', unwarp_func, 'in_file')

###############################################################
##### EXPERIMENTAL AND UNTESTED BELOW THIS ####################
###############################################################

# # # ----------- coregistation to standard

# resample caltech atlas to resolution of functional (from size of structural)
downsample_atlas_t1_to_func = pe.Node(interface=fsl.FLIRT(dof=6, output_type = "NIFTI_GZ", reference=fmri_dir + 'CIT_brains/CIT168_T1w_700um_MNI.nii.gz', in_matrix_file='/usr/share/fsl/5.0/etc/flirtsch/ident.mat'), name='downsample_atlas_t1_to_func')
preproc.connect(inputnode, ('func', get_voxel_size), downsample_atlas_t1_to_func, 'apply_isoxfm')
preproc.connect(downsample_atlas_t1, 'out_file', downsample_atlas_t1_to_func, 'in_file')

downsample_atlas_t2_to_func = pe.Node(interface=fsl.FLIRT(dof=6, output_type = "NIFTI_GZ", reference=fmri_dir + 'CIT_brains/CIT168_T2w_700um_MNI.nii.gz', in_matrix_file='/usr/share/fsl/5.0/etc/flirtsch/ident.mat'), name='downsample_atlas_t2_to_func')
preproc.connect(inputnode, ('func', get_voxel_size), downsample_atlas_t2_to_func, 'apply_isoxfm')
preproc.connect(downsample_atlas_t2, 'out_file', downsample_atlas_t2_to_func, 'in_file')

# invert xfm 
downsample_atlas_t2_to_func_inv = pe.MapNode(interface=fsl.ConvertXFM(output_type = "NIFTI_GZ", invert_xfm = True), name='downsample_atlas_t2_to_func_inv', iterfield=['in_file'])
preproc.connect(downsample_atlas_t2_to_func, 'out_matrix_file', downsample_atlas_t2_to_func_inv, 'in_file')

# convert to itk format
fsl2ras_downsample_atlas_from_struct_to_func = pe.Node(interface=c3.C3dAffineTool(fsl2ras = True, itk_transform=True, reference_file=fmri_dir + 'CIT_brains/CIT168_T1w_700um_MNI.nii.gz'), name='fsl2ras_downsample_atlas_from_struct_to_func')
preproc.connect(downsample_atlas_t2_to_func, 'out_matrix_file', fsl2ras_downsample_atlas_from_struct_to_func, 'transform_file')
preproc.connect(downsample_atlas_t2, 'out_file', fsl2ras_downsample_atlas_from_struct_to_func, 'source_file')

# convert to itk format
fsl2ras_downsample_atlas_from_struct_to_func_inv = pe.Node(interface=c3.C3dAffineTool(fsl2ras = True, itk_transform=True, source_file=fmri_dir + 'CIT_brains/CIT168_T1w_700um_MNI.nii.gz'), name='fsl2ras_downsample_atlas_from_struct_to_func_inv')
preproc.connect(downsample_atlas_t2_to_func_inv, ('out_file', unlist), fsl2ras_downsample_atlas_from_struct_to_func_inv, 'transform_file')
preproc.connect(downsample_atlas_t2, 'out_file', fsl2ras_downsample_atlas_from_struct_to_func_inv, 'reference_file')

# co-register mbRef to anatomical
mbRef_to_t2 = pe.MapNode(interface=fsl.FLIRT(dof=6, output_type = "NIFTI_GZ"), name='mbRef_to_t2', iterfield=['in_file'])
preproc.connect(unwarp_mbRef, 'unwarped_file', mbRef_to_t2, 'in_file')
preproc.connect(maskT2, 'out_file', mbRef_to_t2, 'reference')

# invert xfm 
invert_t2_to_mbRef_tfm = pe.MapNode(interface=fsl.ConvertXFM(output_type = "NIFTI_GZ", invert_xfm = True), name='invert_t2_to_mbRef_tfm', iterfield=['in_file'])
preproc.connect(mbRef_to_t2, 'out_matrix_file', invert_t2_to_mbRef_tfm, 'in_file')

# transform t2 mask to func space
t2Mask_to_mbRef = pe.MapNode(interface=fsl.FLIRT(apply_xfm = True, output_type = "NIFTI_GZ", interp='nearestneighbour'), name='t2Mask_to_mbRef', iterfield=['reference','in_matrix_file'])
preproc.connect(inflate_mask, 'out_file', t2Mask_to_mbRef, 'in_file')
preproc.connect(unwarp_mbRef, 'unwarped_file', t2Mask_to_mbRef, 'reference')
preproc.connect(invert_t2_to_mbRef_tfm, 'out_file', t2Mask_to_mbRef, 'in_matrix_file')

# apply t2 mask to mbRef
mask_mbRef = pe.MapNode(interface=fsl.ImageMaths(op_string = '-mas', suffix = '_bet'), name='mask_mbRef', iterfield=['in_file','in_file2'])
preproc.connect(unwarp_mbRef, 'unwarped_file', mask_mbRef, 'in_file')
preproc.connect(t2Mask_to_mbRef, 'out_file', mask_mbRef, 'in_file2')

# convert to itk format
fsl2ras_mbRef_to_t2 = pe.MapNode(interface=c3.C3dAffineTool(fsl2ras = True, itk_transform=True), name='fsl2ras_mbRef_to_t2', iterfield=['source_file','transform_file'])
preproc.connect(maskT2, 'out_file', fsl2ras_mbRef_to_t2, 'reference_file')
preproc.connect(unwarp_mbRef, 'unwarped_file', fsl2ras_mbRef_to_t2, 'source_file')
preproc.connect(mbRef_to_t2, 'out_matrix_file', fsl2ras_mbRef_to_t2, 'transform_file')

# assemble inputs for mbref to t2 coregistration
ants_mbRef_to_t2_input = pe.MapNode(interface=util.IdentityInterface(fields=['fixed_image', 'moving_image', 'fixed_image_mask', 'moving_image_mask', 'initial_moving_transform']), name='ants_mbRef_to_t2_input', iterfield=['moving_image','moving_image_mask','initial_moving_transform'])
preproc.connect(mask_mbRef, 'out_file', ants_mbRef_to_t2_input, 'moving_image')
preproc.connect(maskT2, 'out_file', ants_mbRef_to_t2_input, 'fixed_image')
preproc.connect(t2Mask_to_mbRef, 'out_file', ants_mbRef_to_t2_input, 'moving_image_mask')
preproc.connect(inflate_mask, 'out_file', ants_mbRef_to_t2_input, 'fixed_image_mask')
preproc.connect(fsl2ras_mbRef_to_t2, 'itk_transform', ants_mbRef_to_t2_input, 'initial_moving_transform')

# perform mbRef to t2 coregistration
ants_mbRef_to_t2 = pe.MapNode(interface=ants.Registration(dimension=3, transforms=['Affine'], metric=[['Mattes']], radius_or_number_of_bins=[[32]], metric_weight = [[1]], transform_parameters=[[.01]], number_of_iterations=[[100]], convergence_threshold=[1.e-6], convergence_window_size=[10], smoothing_sigmas=[[2.0]], shrink_factors=[[2]], use_histogram_matching=True, interpolation='Linear', invert_initial_moving_transform=False, sampling_strategy = [['Random']], sampling_percentage = [[0.05]], output_warped_image = 'output_warped_image.nii.gz'), name='ants_mbRef_to_t2', iterfield=['moving_image','moving_image_mask','initial_moving_transform'])
preproc.connect(ants_mbRef_to_t2_input, 'fixed_image', ants_mbRef_to_t2, 'fixed_image')
preproc.connect(ants_mbRef_to_t2_input, 'moving_image', ants_mbRef_to_t2, 'moving_image')
preproc.connect(ants_mbRef_to_t2_input, 'fixed_image_mask', ants_mbRef_to_t2, 'fixed_image_mask')
preproc.connect(ants_mbRef_to_t2_input, 'moving_image_mask', ants_mbRef_to_t2, 'moving_image_mask')
preproc.connect(ants_mbRef_to_t2_input, 'initial_moving_transform', ants_mbRef_to_t2, 'initial_moving_transform')

# assemble output of this
ants_mbRef_to_t2_output = pe.MapNode(interface=util.IdentityInterface(fields=['forward_transforms', 'warped_image', 'reverse_transforms', 'inverse_warped_image']), name='ants_mbRef_to_t2_output', iterfield=['forward_transforms','warped_image','reverse_transforms','inverse_warped_image'])
preproc.connect(ants_mbRef_to_t2, 'forward_transforms', ants_mbRef_to_t2_output, 'forward_transforms')
preproc.connect(ants_mbRef_to_t2, 'warped_image', ants_mbRef_to_t2_output, 'warped_image')
preproc.connect(ants_mbRef_to_t2, 'reverse_transforms', ants_mbRef_to_t2_output, 'reverse_transforms')
preproc.connect(ants_mbRef_to_t2, 'inverse_warped_image', ants_mbRef_to_t2_output, 'inverse_warped_image')

## apply transformation from mbRef all the way to caltech atlas (in func resolution)
#ants_mbRef_to_t2_do = pe.MapNode(interface=ants.WarpImageMultiTransform(use_nearest = True), name = 'ants_mbRef_to_t2_do')
#preproc.connect(mask_mbRef, 'out_file', ants_mbRef_to_t2_do, 'input_image')
#preproc.connect(maskT2, 'out_file', ants_mbRef_to_t2_do, 'reference_image')
#preproc.connect(ants_mbRef_to_t2, 'forward_transforms', ants_mbRef_to_t2_do, 'transformation_series')

# mbRef_to_t2_1 = pe.Node(interface=fsl.FLIRT(dof=6, output_type = "NIFTI_GZ"), name='mbRef_to_t2_1')
# preproc.connect(mask_mbRef, 'out_file', mbRef_to_t2_1, 'in_file')
# preproc.connect(maskT2, 'out_file', mbRef_to_t2_1, 'reference')

sep_tfms = pe.Node(interface=Function(input_names=['in_list'], output_names=['out_list'], function=get_sec_tfm),name='sep_tfms')
preproc.connect(ants_mbRef_to_t2_output, 'forward_transforms', sep_tfms, 'in_list')

# combine transformations for (func to atlas)
cmb_tfm_func_to_ds_atlas = pe.MapNode(interface=util.Merge(4, axis='vstack'), name='cmb_tfm_func_to_ds_atlas', iterfield=['in4'])
# from atlas in struct resolution to func resolution
preproc.connect(fsl2ras_downsample_atlas_from_struct_to_func, 'itk_transform', cmb_tfm_func_to_ds_atlas, 'in1')
# from structural to atlas (with struct resolution)
preproc.connect(structs_to_atlas_coreg_output, ('forward_transforms', get_sec), cmb_tfm_func_to_ds_atlas, 'in2')
preproc.connect(structs_to_atlas_coreg_output, ('forward_transforms', get_first), cmb_tfm_func_to_ds_atlas, 'in3')
# transform from functional to structural
#preproc.connect(ants_mbRef_to_t2_output, ('forward_transforms', get_sec), cmb_tfm_func_to_ds_atlas, 'in4')
preproc.connect(sep_tfms, 'out_list', cmb_tfm_func_to_ds_atlas, 'in4')

# apply transformation from mbRef all the way to caltech atlas (in func resolution)
mbRef_to_ds_atlas = pe.MapNode(interface=ants.WarpImageMultiTransform(use_nearest = True), name = 'mbRef_to_ds_atlas', iterfield=['input_image', 'transformation_series'])
preproc.connect(mask_mbRef, 'out_file', mbRef_to_ds_atlas, 'input_image')
preproc.connect(downsample_atlas_t2_to_func, 'out_file', mbRef_to_ds_atlas, 'reference_image')
preproc.connect(cmb_tfm_func_to_ds_atlas, 'out', mbRef_to_ds_atlas, 'transformation_series')

# apply transformation to 4D func
func_to_ds_atlas = pe.MapNode(interface=ants.WarpTimeSeriesImageMultiTransform(use_nearest = True), name = 'func_to_ds_atlas', iterfield=['transformation_series','input_image'])
preproc.connect(unwarp_func, 'unwarped_file', func_to_ds_atlas, 'input_image')
preproc.connect(downsample_atlas_t2_to_func, 'out_file', func_to_ds_atlas, 'reference_image')
preproc.connect(cmb_tfm_func_to_ds_atlas, 'out', func_to_ds_atlas, 'transformation_series')

# combine transformations (t2 to atlas)
cmb_tfm_t2_to_ds_atlas = pe.Node(interface=util.Merge(3, axis='hstack'), name='cmb_tfm_t2_to_ds_atlas')
# from structural to atlas (with struct resolution)
preproc.connect(structs_to_atlas_coreg_output, ('forward_transforms', get_sec), cmb_tfm_t2_to_ds_atlas, 'in1')
preproc.connect(structs_to_atlas_coreg_output, ('forward_transforms', get_first), cmb_tfm_t2_to_ds_atlas, 'in2')
# from atlas in struct resolution to t2 resolution
preproc.connect(fsl2ras_downsample_atlas_from_struct_to_func, 'itk_transform', cmb_tfm_t2_to_ds_atlas, 'in3')

# apply transformation from t2 all the way to caltech atlas (in func resolution)
t2_to_ds_atlas_func = pe.Node(interface=ants.WarpImageMultiTransform(use_nearest = True), name = 't2_to_ds_atlas_func')
preproc.connect(maskT2, 'out_file', t2_to_ds_atlas_func, 'input_image')
preproc.connect(downsample_atlas_t2_to_func, 'out_file', t2_to_ds_atlas_func, 'reference_image')
preproc.connect(cmb_tfm_t2_to_ds_atlas, ('out', unlist), t2_to_ds_atlas_func, 'transformation_series')

# apply transformation from t1 all the way to caltech atlas (in func resolution)
t1_to_ds_atlas_func = pe.Node(interface=ants.WarpImageMultiTransform(use_nearest = True), name = 't1_to_ds_atlas_func')
preproc.connect(maskT1, 'out_file', t1_to_ds_atlas_func, 'input_image')
preproc.connect(downsample_atlas_t1_to_func, 'out_file', t1_to_ds_atlas_func, 'reference_image')
preproc.connect(cmb_tfm_t2_to_ds_atlas, ('out', unlist), t1_to_ds_atlas_func, 'transformation_series')

# smooth - create a mask first so smoothing doesnt go outside brain
mask_standard_func = pe.MapNode(interface=fsl.ImageMaths(op_string = '-thr 100 -bin'), name = 'mask_standard_func', iterfield=['in_file'])
preproc.connect(func_to_ds_atlas, 'output_image',  mask_standard_func, 'in_file')

smooth = pe.MapNode(interface=fsl.ImageMaths(op_string = '-s 2.12 -mas', suffix = '_smooth'), name = 'smooth', iterfield=['in_file', 'in_file2'])
preproc.connect(func_to_ds_atlas, 'output_image',  smooth, 'in_file')
preproc.connect(mask_standard_func, 'out_file',  smooth, 'in_file2')

preproc.run(plugin='MultiProc', plugin_args={'n_procs' : 5})

time_dif = time.time() - start_time
print 'Finished script at ',time_dif
