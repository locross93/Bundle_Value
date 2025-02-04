%-----------------------------------------------------------------------
% Job saved on 26-Jan-2022 12:22:21 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
analysis_dir = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/';
analysis_name1 = 'wtp_minus_ref';
analysis_name2 = 'rel_value';
contrast_file = 'con_0001.nii';

matlabbatch{1}.spm.stats.factorial_design.dir = {'/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/2nd_level/wtp_minus_ref/Paired_T'};
count = 1;
sub_list = {'101','102','103'};
for sub_num=1:length(sub_list)
    subID = sub_list{sub_num};
    temp_file1 = [analysis_dir,'sub',subID,'/',analysis_name1,'_downsample/',contrast_file,',1'];
    temp_file2 = [analysis_dir,'sub',subID,'/',analysis_name2,'_downsample/',contrast_file,',1'];
    scans = {temp_file1, temp_file2};
    matlabbatch{1}.spm.stats.factorial_design.des.pt.pair(count).scans = scans';
    count = count + 1;
end
sub_list = {'104','105','106','107','108','109','110','111','112','113','114'};
for sub_num=1:length(sub_list)
    subID = sub_list{sub_num};
    temp_file1 = [analysis_dir,'sub',subID,'/',analysis_name1,'/',contrast_file,',1'];
    temp_file2 = [analysis_dir,'sub',subID,'/',analysis_name2,'/',contrast_file,',1'];
    scans = {temp_file1, temp_file2};
    matlabbatch{1}.spm.stats.factorial_design.des.pt.pair(count).scans = scans';
    count = count + 1;
end

matlabbatch{1}.spm.stats.factorial_design.des.pt.gmsca = 0;
matlabbatch{1}.spm.stats.factorial_design.des.pt.ancova = 0;
matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;
