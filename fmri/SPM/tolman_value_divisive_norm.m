function tolman_value_divisive_norm(subID)
%put all trial type values in one regressor
%motion regressors
%regressors for choosing reference and choosing item

%run ConvertChoiceDataToText.m first to get timing struct

first_ds = true;

%add spm12 to matlab path
addpath(genpath('/usr/local/matlab/R2014a/toolbox/spm12'))

spm('defaults','FMRI');
spm_jobman('initcfg');

clear matlabbatch % Every preprocessing step needs this line

if first_ds
    TR = 1.1; %specify TR (in secs)
else
    TR = 1.12; %specify TR (in secs)
end

time_file_dir = '/home/lcross/Bundle_Value/analysis/SPM/timing_files/';
spm_dir = ['/home/lcross/Bundle_Value/analysis/SPM/sub',subID,'/div_norm_value'];

if isdir(spm_dir) == 0
    mkdir(spm_dir)
end

%initialize batch
matlabbatch{1}.spm.stats.fmri_spec.dir = {spm_dir};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

%create a new session for every day, concatenate every run within a day 
for day=1:3
    if first_ds
        %first dataset
        func_dir = ['/home/lcross/Bundle_Value/analysis/sub',subID,'/func/smooth/day',num2str(day)];
    else
        %second dataset
        func_dir = ['/home/lcross/Bundle_Value/analysis/sub',subID,'/day',num2str(day)];
    end
    
    %Concatenate onsets from all blocks
    r1 = spm_select('ExtFPList', func_dir,'run1.nii', Inf);
    r2 = spm_select('ExtFPList', func_dir,'run2.nii', Inf);
    r3 = spm_select('ExtFPList', func_dir,'run3.nii', Inf);
    r4 = spm_select('ExtFPList', func_dir,'run4.nii', Inf);
    r5 = spm_select('ExtFPList', func_dir,'run5.nii', Inf);
    
    %Calculate cumulated block durations
    r1_dur = (length(r1)*TR);
    r2_dur = r1_dur + (length(r2)*TR);
    r3_dur = r2_dur + (length(r3)*TR);
    r4_dur = r3_dur + (length(r4)*TR);
    run_dur_mat = [0, r1_dur, r2_dur, r3_dur, r4_dur];
    
    %load subj struct/matlab file with regressors for left vs right choice
    load([time_file_dir,'sub',subID,'-',num2str(day),'_timing'])
    
    trial_onsets = [subj_struct.value_onset{1}; subj_struct.value_onset{2} + r1_dur;...
    subj_struct.value_onset{3}+ r2_dur; subj_struct.value_onset{4} + r3_dur;...
    subj_struct.value_onset{5} + r4_dur];

    trial_dur = [subj_struct.value_dur{1}; subj_struct.value_dur{2};...
    subj_struct.value_dur{3}; subj_struct.value_dur{4};...
    subj_struct.value_dur{5}];

    trial_value = [subj_struct.value_param{1}; subj_struct.value_param{2};...
    subj_struct.value_param{3}; subj_struct.value_param{4};...
    subj_struct.value_param{5}];
    
    %trial category (0 single item - 1 bundle)
    trial_cat = double([subj_struct.value_categ{1}; subj_struct.value_categ{2};...
    subj_struct.value_categ{3}; subj_struct.value_categ{4};...
    subj_struct.value_categ{5}]);

    %divisive normalize value by category for relative value
    sitem_inds = find(trial_cat == 0);
    bundle_inds = find(trial_cat == 1);
    trial_value(sitem_inds) = trial_value(sitem_inds) / mean(trial_value(sitem_inds));
    trial_value(bundle_inds) = trial_value(bundle_inds) / mean(trial_value(bundle_inds));

    %turn 0s to -1s for interaction
    trial_cat(sitem_inds) = -1;
    
    left_onsets = [subj_struct.left_button{1}; subj_struct.left_button{2} + r1_dur; subj_struct.left_button{3} + r2_dur;...
        subj_struct.left_button{4} + r3_dur; subj_struct.left_button{5} + r4_dur];

    right_onsets = [subj_struct.right_button{1}; subj_struct.right_button{2} + r1_dur; subj_struct.right_button{3} + r2_dur;...
        subj_struct.right_button{4} + r3_dur; subj_struct.right_button{5} + r4_dur];
    
    %add two more regressors for choosing reference and choosing item
    
    choose_ref_onsets = [subj_struct.value_onset{1}(~subj_struct.trial_choice{1}); subj_struct.value_onset{2}(~subj_struct.trial_choice{2}) + r1_dur;...
        subj_struct.value_onset{3}(~subj_struct.trial_choice{3}) + r2_dur; subj_struct.value_onset{4}(~subj_struct.trial_choice{4}) + r3_dur;...
        subj_struct.value_onset{5}(~subj_struct.trial_choice{5}) + r4_dur];
    
    choose_item_onsets = [subj_struct.value_onset{1}(subj_struct.trial_choice{1}); subj_struct.value_onset{2}(subj_struct.trial_choice{2}) + r1_dur;...
        subj_struct.value_onset{3}(subj_struct.trial_choice{3}) + r2_dur; subj_struct.value_onset{4}(subj_struct.trial_choice{4}) + r3_dur;...
        subj_struct.value_onset{5}(subj_struct.trial_choice{5}) + r4_dur];
    
    %start creating matlab batch
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).scans = cellstr([r1;r2;r3;r4;r5;]);
    
    %stimulus onset
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).name = 'Stim onset';
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).onset = trial_onsets;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).duration = trial_dur;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).pmod(1).name = 'Value of item/bundle';
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).pmod(1).param = trial_value;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).pmod(1).poly = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).pmod(2).name = 'Trial type';
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).pmod(2).param = trial_cat;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).pmod(2).poly = 1;
    %interaction regressor
    interaction = trial_value.*trial_cat;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).pmod(3).name = 'Interaction';
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).pmod(3).param = interaction;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).pmod(3).poly = 1;
    %add RT - zscore first
    choice_rt = zscore(trial_dur);
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).pmod(4).name = 'RT';
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).pmod(4).param = choice_rt;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).pmod(4).poly = 1;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(1).orth = 1;
    
    %left hand choice 
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(2).name = 'LH onset';
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(2).onset = left_onsets;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(2).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(2).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(2).pmod = struct('name', {}, 'param', {}, 'poly', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(2).orth = 0;

    %right hand choice 
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(3).name = 'RH onset';
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(3).onset = right_onsets;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(3).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(3).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(3).pmod = struct('name', {}, 'param', {}, 'poly', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(3).orth = 0;
    
    %reference choice 
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(4).name = 'Choose ref. onset';
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(4).onset = choose_ref_onsets;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(4).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(4).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(4).pmod = struct('name', {}, 'param', {}, 'poly', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(4).orth = 0;
    
    %item choice 
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(5).name = 'Choose item onset';
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(5).onset = choose_item_onsets;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(5).duration = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(5).tmod = 0;
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(5).pmod = struct('name', {}, 'param', {}, 'poly', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).cond(5).orth = 0;

    matlabbatch{1}.spm.stats.fmri_spec.sess(day).multi = {''};
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).regress = struct('name', {}, 'val', {});
    
    %motion regressors
    M = [];
    for run_id=1:5
        if first_ds
            motion_file = ['/home/lcross/Bundle_Value/preprocessing/sub',subID,'/ICA/day',num2str(day),'/run',num2str(run_id),'.ica/mc/prefiltered_func_data_mcf.par'];
        else
            motion_file = ['/home/lcross/Bundle_Value/preprocessing/FSL_ICA_BET/sub',subID,'-',num2str(day),'/Session',num2str(run_id),'.ica/mc/prefiltered_func_data_mcf.par'];
        end 
        M_temp = dlmread(motion_file);
        M = [M; M_temp];
    end
    
    %add regressors to motion to account for collapsing across sessions
    R_temp = [ones(length(r1),4);zeros(length(r2),1) ones(length(r2),3); ...
        zeros(length(r3),2) ones(length(r3),2);zeros(length(r4),3) ones(length(r4),1); ...
        zeros(length(r5),4)];
    
    R = [M R_temp];
    conf_sess = [spm_dir,'/sessions_regressors_day',num2str(day),'.mat'];
    save(conf_sess,'R');
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).multi_reg = cellstr(conf_sess);
    matlabbatch{1}.spm.stats.fmri_spec.sess(day).hpf = 128;
end

matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.1; %change this in case there are 'holes' in the mask
matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

matlabbatch{2}.spm.stats.fmri_est.spmmat = {[spm_dir,'/SPM.mat']};
matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;  

save([spm_dir,'/batch_model-',subID,'.mat'],'matlabbatch');
spm_jobman('run',matlabbatch)

end