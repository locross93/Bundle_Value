#!/usr/bin/env python

import math
import os
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.algorithms.modelgen as model   # model generation
import nipype.algorithms.rapidart as ra      # artifact detection
import nibabel as nib
import nipype.interfaces.c3 as c3 #WG 7/12/2018 Had to install c3 scripts locally for this to work.
import nipype.interfaces.ants as ants
from nipype.interfaces.fsl import fix        # fix automated denoising
from nipype.interfaces.utility import Function
from IPython.display import Image
import time

start_time = time.time()

data_dir = '/home/wgriggs/fmri/data/'
code_dir = '/home/wgriggs/NiPypeCode/'

def Get_nVols(in_file):
    from nipype import logging
    import nibabel as nib
    v = nib.load(in_file)
    data = v.get_data()
    npts = data.shape[3]
    iflogger = logging.getLogger('interface')
    iflogger.info("npts: %s" % npts)
    return npts

def Get_TotalVoxels(in_file):
    from nipype import logging
    import nibabel as nib
    import numpy as np
    v = nib.load(in_file)
    data = v.get_data()
    total_voxels = np.prod(data.shape)
    iflogger = logging.getLogger('interface')
    iflogger.info("Total Voxels: %s" % total_voxels)
    return total_voxels

#WG 7/17/2018 Need to update to reflect proper number of voxels/npts
def Prepare_Design_FSF(feat_files, initial_highres_files, highres_files, npts=99, total_voxels=191):
    from nipype import logging
    import os
    # WG 7/13/18 Using 
    design_fsf = open(code_dir +'MelodicTemplate3.fsf','r')
    
    # assemble dicts from inputs
    str_rep = dict()
    str_rep['feat_files(1)'] = feat_files.strip('.nii.gz')
    str_rep['initial_highres_files(1)'] = initial_highres_files.strip('.nii.gz')
    str_rep['highres_files(1)'] = highres_files.strip('.nii.gz')
    int_rep = dict()
    int_rep['fmri(totalVoxels)'] = total_voxels
    int_rep['fmri(npts)'] = npts

    out_lines = []
    lines = design_fsf.readlines()
    for line in lines:
        items = line.split(' ')
        if len(items) == 3:
            if str_rep.has_key(items[1]):
                # print line
                out_line = "%s %s \"%s\"\n" % (items[0], items[1], str_rep[items[1]])
                # print out_line
            elif int_rep.has_key(items[1]):
                # print line
                out_line = "%s %s %s\n" % (items[0], items[1], int_rep[items[1]])
                # print out_line
            else:
                out_line = line
            out_lines.append(out_line)
        else:
            out_lines.append(line)
    iflogger = logging.getLogger('interface')
    out_file = os.path.abspath('design.fsf')
    iflogger.info(out_file)
    with open(out_file, 'w') as fp:
       fp.writelines(out_lines)
    return out_file

# feat_files = "/home/pauli/Development/core_shell/data/mri/JOD-WP-CS2-005/f1_short"
# initial_highres_files = "/home/pauli/Development/core_shell/data/mri/JOD-WP-CS2-005/f1_ref"
# highres_files = "/home/pauli/Development/core_shell/data/mri/JOD-WP-CS2-005/t2"
# total_voxels = 35000000
# npts = 100
# design_fsf = prepare_design_fsf(feat_files, initial_highres_files, highres_files, npts, total_voxels)

# remove square brackets (['filename'] -> 'filename')
def unlist(mylist):
    #print "Mylist"
    #print mylist
    r = mylist[0]
    #print r
    return r

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

# create preproc workflow
preproc = pe.Workflow(name='preproc')
preproc.base_dir = os.path.abspath('./')

# preproc.config['execution']['crashfile_format'] = 'txt'
# try:
#     preproc.run()
# except(RuntimeError) as err:
#     print("RuntimeError:", err)
# else:
#     raise

# WG 7/11/2018 Unclear where to get this file from. Not important until very last steps of preprocessing though, so currently ignoring.
#ds_ventricle_mask = '/home/pauli/Development/core_shell/data/openfmri/group/ventricle_mask.nii.gz'


# WG 7/12/2018 Defining templates of files to load for anatomical (T1 and T2 normalized), field maps (positive and negative), BOLD, and BOLD_Ref
templates={'t1':'sub{subID}/sub-{subID}/ses-day{dayID}/anat/sub-{subID}_ses-day{dayID}_T1w_run-02*.nii.gz',
           't2':'sub{subID}/sub-{subID}/ses-day{dayID}/anat/sub-{subID}_ses-day{dayID}_*T2*.nii.gz',
           'func':'sub{subID}/sub-{subID}/ses-day{dayID}/func/sub-{subID}_ses-day{dayID}_*bold.nii.gz',
          'func_ref':'sub{subID}/sub-{subID}/ses-day{dayID}/func/sub-{subID}_ses-day{dayID}_*bold_SBRef.nii.gz',
          'fm_pos':'sub{subID}/sub-{subID}/ses-day{dayID}/fmap/sub-{subID}_ses-day{dayID}_*AP*.nii.gz',
          'fm_neg':'sub{subID}/sub-{subID}/ses-day{dayID}/fmap/sub-{subID}_ses-day{dayID}_*PA*.nii.gz'}

#WG 7/12/2018 Creating node to load files and specifying necessary options.
datasource = pe.Node(nio.SelectFiles(templates),
                    name='selectfiles')
datasource.inputs.base_directory = data_dir
datasource.inputs.subID='101' #WG 7/12/2018 Can modify later to allow for all subject IDs.
datasource.inputs.dayID='1'
datasource.inputs.sort_filelist = True

#WG 7/12/2018 Sets up node to feed respective image types into different sub-pipelines
inputnode = pe.Node(interface=util.IdentityInterface(fields=['func', 'func_ref', 't1', 't2', 'fm_pos', 'fm_neg']), name='inputnode')

#WG 7/12/2018 Connects each output from the SelectFile node (datasource) to the IdentityInterface node (inputnode)
preproc.connect(datasource, 'func', inputnode, 'func')
preproc.connect(datasource, 'func_ref', inputnode, 'func_ref')
preproc.connect(datasource, 't1', inputnode, 't1')
preproc.connect(datasource, 't2', inputnode, 't2')
preproc.connect(datasource, 'fm_pos', inputnode, 'fm_pos')
preproc.connect(datasource, 'fm_neg', inputnode, 'fm_neg')

# ------------- prepare anatomical data -------------

# WG 7/12/2018 Believe prep_anatomicals is currently unused, so commenting out
#prep_anatomicals = pe.Workflow(name='prep_anatomicals')
#prep_anatomicals.base_dir = os.path.abspath('./nipype')

# reorient t1 to standard
T1_to_standard = pe.Node(interface=fsl.Reorient2Std(output_type = "NIFTI_GZ"), name='T1_to_standard') 
preproc.connect(inputnode, 't1', T1_to_standard, 'in_file')


# reorient t2 to standard
T2_to_standard = pe.Node(interface=fsl.Reorient2Std(output_type = "NIFTI_GZ"), name='T2_to_standard') 
preproc.connect(inputnode, 't2', T2_to_standard, 'in_file')


# coreg t2 to t1
t2tot1 = pe.Node(interface=fsl.FLIRT(dof=6, output_type = "NIFTI_GZ"), name='t2tot1')
preproc.connect(T2_to_standard, 'out_file', t2tot1, 'in_file')
preproc.connect(T1_to_standard, 'out_file', t2tot1, 'reference')

time_dif = time.time() - start_time
print 'Finished reorienting and coreg at ',time_dif


# determine resolution of t1/t2
# get the middle volume of the session
def get_voxel_size(f):
    from nibabel import load
    in_file = f
    if isinstance(f, list):
        in_file = f[0]
    img_header = load(in_file).get_header().get_zooms()
    return float(img_header[0])
    

# # downsample caltech atlas to t1/t2 resolution
# WG 7/12/2018 Currently using local copies of ident.mat and reference atlases
downsample_atlas_t1 = pe.Node(interface=fsl.FLIRT(dof=6, output_type = "NIFTI_GZ", in_file= data_dir + 'CIT168_T1w_700um_MNI.nii.gz', reference=data_dir + 'CIT168_T1w_700um_MNI.nii.gz', in_matrix_file='/usr/share/fsl/5.0/etc/flirtsch/ident.mat'), name='downsample_atlas_t1')
preproc.connect(inputnode, ('t1', get_voxel_size), downsample_atlas_t1, 'apply_isoxfm')

downsample_atlas_t2 = pe.Node(interface=fsl.FLIRT(dof=6, output_type = "NIFTI_GZ", in_file=data_dir + 'CIT168_T2w_700um_MNI.nii.gz', reference=data_dir + 'CIT168_T2w_700um_MNI.nii.gz', in_matrix_file='/usr/share/fsl/5.0/etc/flirtsch/ident.mat'), name='downsample_atlas_t2')
preproc.connect(inputnode, ('t2', get_voxel_size), downsample_atlas_t2, 'apply_isoxfm')

# # create mask file for atlas
# WG 7/12/2018 Keeping same parameters as used by Wolfgang currently.
mask_atlas = pe.Node(interface=fsl.ImageMaths(op_string = '-thr .001 -bin -dilF', suffix = '_mask'), name='mask_atlas')
preproc.connect(downsample_atlas_t1, 'out_file', mask_atlas, 'in_file')



# skull strip t1
#nosestrip = pe.Node(interface=fsl.BET(frac=0.3),
#                     name = 'nosestrip')
skullstrip = pe.Node(interface=fsl.BET(mask = True, frac=0.2, reduce_bias = True),
                     name = 'stripstruct')
preproc.connect(T1_to_standard, 'out_file', skullstrip, 'in_file')
#preproc.connect(nosestrip, 'out_file', skullstrip, 'in_file')

inflate_mask = pe.Node(interface=fsl.ImageMaths(op_string = '-dilF', suffix = '_dilF'), name='inflate_mask')
preproc.connect(skullstrip, 'mask_file', inflate_mask, 'in_file')


# apply t1 mask to t2
maskT2 = pe.Node(interface=fsl.ImageMaths(op_string = '-mas', suffix = '_bet'), name='maskT2')
preproc.connect(t2tot1, 'out_file', maskT2, 'in_file')
preproc.connect(inflate_mask, 'out_file', maskT2, 'in_file2')

maskT1 = pe.Node(interface=fsl.ImageMaths(op_string = '-mas', suffix = '_bet'), name='maskT1')
preproc.connect(T1_to_standard, 'out_file', maskT1, 'in_file')
preproc.connect(inflate_mask, 'out_file', maskT1, 'in_file2')


# # convert func to float (TODO: WHY?)
# # img2float = pe.MapNode(interface=fsl.ImageMaths(out_data_type='float', op_string = '', suffix='_dtype'), iterfield=['in_file'], name='img2float')
# # preproc.connect(inputnode, 'func', img2float, 'in_file')



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
# WG 7/12/2018 Currently needs to be an older version of ants 2.1.0. Cannot use latest ANTS 2.2.0
structs_to_atlas_coreg = pe.Node(interface=ants.Registration(dimension=3, transforms=['SyN'], metric=[['CC'] * 2], radius_or_number_of_bins=[[4] * 2], metric_weight = [[.5] * 2], transform_parameters=[[.1,3.0,0.0]], number_of_iterations=[[100,100,70,50,20]], convergence_threshold=[1.e-6], convergence_window_size=[10], smoothing_sigmas=[[5.0,3.0,2.0,1.0,0.0]], shrink_factors=[[10,6,4,2,1]], use_histogram_matching=True, interpolation='Linear', invert_initial_moving_transform=False, sampling_strategy = [['Random']*2], sampling_percentage = [[0.05]*2]), name='structs_to_atlas_coreg')
#structs_to_atlas_coreg = pe.Node(interface=ants.Registration(dimension=3, transforms=['SyN'], metric=[['CC'] * 2], radius_or_number_of_bins=[[4] * 2], metric_weight = [[.5] * 2], transform_parameters=[[.1,3.0,0.0]], number_of_iterations=[[10]], convergence_threshold=[1.e-1], convergence_window_size=[10], smoothing_sigmas=[[5.0]], shrink_factors=[[10]], use_histogram_matching=True, interpolation='Linear', output_warped_image=True, invert_initial_moving_transform=False, verbose=False), name='structs_to_atlas_coreg')
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


def reverse(mylist):
    mylist.reverse()
    return mylist

# # apply transformation from func all the way to caltech atlas (in func resolution)
# t1_to_ds_atlas = pe.Node(interface=ants.WarpImageMultiTransform(use_nearest = True), name = 't1_to_ds_atlas')
# preproc.connect(maskT1, 'out_file', t1_to_ds_atlas, 'input_image')
# preproc.connect(downsample_atlas_t1, 'out_file', t1_to_ds_atlas, 'reference_image')
# preproc.connect(structs_to_atlas_coreg_output, ('forward_transforms', reverse), t1_to_ds_atlas, 'transformation_series')

preproc.run(plugin='MultiProc', plugin_args={'n_procs' : 5})

time_dif = time.time() - start_time
print 'Finished script at ',time_dif

