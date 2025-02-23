addpath('C:\Users\locro\Documents\MATLAB\cbm\')
% hierarchical Bayesian inference using cbm_hbi. cbm_hbi needs 4 inputs.
% 1st input: data for all subjects
data = load('data/all_subj_data.mat');
data = data.data;

% 2nd input: a cell input containing function handle to models
models = {@fit_softmax, @fit_dv_norm, @fit_mc_norm, @fit_rw_norm};

% 3rd input: another cell input containing file-address to files saved by cbm_lap
fcbm_maps = {'results/fixed_effects_abs_val.mat', 'results/fixed_effects_dv_norm.mat',...
            'results/fixed_effects_mc_norm.mat', 'results/fixed_effects_rw_norm.mat'};

% note that they corresponds to models (so pay attention to the order)
% 4th input: a file address for saving the output
fname_hbi = 'hbi_norm_codes.mat';

cbm_hbi(data,models,fcbm_maps,fname_hbi);

% look at output
cbm_hbi = load(fname_hbi);
cbm = cbm_hbi.cbm;
cbm.output

% plot 
% 1st input is the file-address of the file saved by cbm_hbi
% 2nd input: a cell input containing model names
model_names = {'Abs Value', 'Div Norm', 'MC Norm', 'RW Norm'};
% note that they corresponds to models (so pay attention to the order)
% 3rd input: another cell input containing parameter names of the winning model
param_names = {'\beta'};
% note that '\alpha^+' is in the latex format, which generates a latin alpha
% 4th input: another cell input containing transformation function associated with each 
transform = {'none'};
% note that if you use a less usual transformation function, you should pass the handle 
cbm_hbi_plot(fname_hbi, model_names, param_names, transform)
% this function creates a model comparison plot (exceednace probability and model frequency) % a plot of transformed parameters of the most frequent model.

% save responsibility
responsibility = cbm_hbi.cbm.output.responsibilitycbm_hbi.cbm.output.responsibility;
T = array2table(responsibility, 'VariableNames', model_names);
filename = 'results/bayesian_model_comparison.csv';
writetable(T, filename);
