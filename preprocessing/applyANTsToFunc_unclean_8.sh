# set this to the directory containing antsRegistration
ANTSPATH=/usr/local/ANTs/build/bin/

# ITK thread count
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS

# Check args
if [ $# -lt 1 ]; then
echo "USAGE: $0 <sbj>"
  exit
fi

sbj=$1
ses=$2
echo Sbject: $sbj
echo Session: $ses

# path to afine transform tool
c3d_affine_tool=/usr/local/c3d/c3d-1.0.0-Linux-x86_64/bin/c3d_affine_tool

# paths to the T1/T2 structurals, and the standard anatomical images
subID_short=${sbj:0:3}
struct_path=/home/lcross/Bundle_Value/preprocessing/prepForICA/sub${subID_short}-1
func_path_og=/home/lcross/Bundle_Value/preprocessing/FSL_ICA_BET/sub${sbj}/Session${ses}.ica
func_path=/home/lcross/Bundle_Value/preprocessing/FSL_ICA_BET/sub${sbj}/Session${ses}.ica/unclean
anat_path=/home/lcross/Atari/FSL_Code_BetICA/preprocessing 

# Assign arguments
fixed_t1=`imglob -extension .nii.gz ${anat_path}/CIT168_T1w_MNI.nii.gz`
fixed_t2=`imglob -extension .nii.gz ${anat_path}/CIT168_T2w_MNI.nii.gz`
moving_t1=`imglob -extension .nii.gz ${struct_path}/T1_reoriented_brain.nii.gz`
moving_t2=`imglob -extension .nii.gz ${struct_path}/T2_reoriented_brain_aligned_T1.nii.gz`
# Prefix for output transform files
outPrefix=${moving_t1%%.nii.gz}
echo ${outPrefix}


# # extract a sample to speed up processing
# echo "Extract volume samples to speed processing at $(date +"%T")"
# fslroi ${func_path}/filtered_func_data_clean_unwarped_unclean.nii.gz ${func_path}/filtered_func_data_sample_1.nii.gz 0 1

# # added by Eva, because with eyetracking head move is so close to the coil
# fslmaths ${func_path}/filtered_func_data_clean_unwarped_unclean.nii.gz -Tmean ${func_path}/filtered_func_mean.nii.gz

# # do a voxel-value bia correction
# echo "Running bias correction at $(date +"%T")"
# /usr/local/ANTs/build/bin/N4BiasFieldCorrection -i ${func_path}/filtered_func_mean.nii.gz -o ${func_path}/filtered_func_data_bias.nii.gz --convergence [50x50x30x20,0.0] -d 3 -v 1

# # get the functionals into alignment with the structural
# echo "Running Flirt Sbj func to struct $(date +"%T")"
# flirt -ref $moving_t2 -in ${func_path}/filtered_func_data_bias -out ${func_path}/tmp_func_to_struct -omat ${func_path}/tmp_func_to_struct.mat

# # convert functionals into a format ANTs can use
# echo "Converting fsl transformation to ras format at $(date +"%T")"
# $c3d_affine_tool -ref $moving_t2 -src ${func_path}/filtered_func_data_bias ${func_path}/tmp_func_to_struct.mat -fsl2ras -oitk ${func_path}/itk_transformation_func_to_struct.txt

# # convert T2 into a format ANTs can use
# echo "Converting fsl transformation to ras format at $(date +"%T")"
# $c3d_affine_tool -ref ${anat_path}/CIT168_T2w_MNI_lowres.nii.gz -src $fixed_t2 ${anat_path}/CIT168_T2w_MNI_lowres.mat -fsl2ras -oitk ${func_path}/itk_transformation_hires_to_lowres.txt



# for a single volume transform
# echo "Apply series of transformations all the way from func to lowres atlas (in MNI space)"
# /usr/local/ANTs/build/bin/WarpImageMultiTransform 3 ${func_path}/filtered_func_data_bias.nii.gz ${func_path}/filtered_func_data_lowres_atlas_bias.nii.gz \
# 	-R ${anat_path}/CIT168_T2w_MNI_lowres.nii.gz \
# 	${func_path}/itk_transformation_hires_to_lowres.txt \
#   	${outPrefix}_xfm1Warp.nii.gz ${outPrefix}_xfm0GenericAffine.mat \
#   	${func_path}/itk_transformation_func_to_struct.txt


    
# for multi-volume transform
echo "Apply series of transformations all the way from func to lowres atlas (in MNI space)"
/usr/local/ANTs/build/bin/WarpTimeSeriesImageMultiTransform 4 ${func_path}/filtered_func_data_unwarped_unclean.nii.gz ${func_path}/filtered_func_data_unclean_unwarped_ANTsReg.nii.gz \
	-R ${anat_path}/CIT168_T2w_MNI_lowres.nii.gz \
	${func_path_og}/itk_transformation_hires_to_lowres.txt \
    ${outPrefix}_xfm1Warp.nii.gz ${outPrefix}_xfm0GenericAffine.mat \
    ${func_path_og}/itk_transformation_func_to_struct.txt
echo "done ants  (in MNI space) at $(date +"%T")"
