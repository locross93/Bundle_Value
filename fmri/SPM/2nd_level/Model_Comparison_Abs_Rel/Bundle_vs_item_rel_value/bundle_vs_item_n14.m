% List of open inputs
nrun = X; % enter the number of runs here
jobfile = {'/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/2nd_level/Model_Comparison_Abs_Rel/Bundle_vs_item_rel_value/bundle_vs_item_n14_job.m'};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(0, nrun);
for crun = 1:nrun
end
spm('defaults', 'FMRI');
spm_jobman('run', jobs, inputs{:});
