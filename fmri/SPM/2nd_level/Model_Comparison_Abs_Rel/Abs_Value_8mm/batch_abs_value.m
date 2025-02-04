% List of open inputs
nrun = 1; % enter the number of runs here
jobfile = {'/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/2nd_level/Model_Comparison_Abs_Rel/Abs_Value_8mm/batch_abs_value_job.m'};
jobs = repmat(jobfile, 1, nrun);
inputs = cell(0, nrun);
for crun = 1:nrun
end
spm('defaults', 'FMRI');
spm_jobman('run', jobs, inputs{:});
