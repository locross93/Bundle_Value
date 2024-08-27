addpath('C:\Users\locro\Documents\MATLAB\cbm\')

% load data
data = load('data/all_subj_data.mat');
data = data.data;

models = {@fit_softmax, @fit_wtp_minus_ref, @fit_dv_norm, @fit_mc_norm, @fit_rw_norm};
model_names = {'abs_val', 'wtp_minus_ref', 'dv_norm', 'mc_norm', 'rw_norm'};

%models = {@fit_wtp_minus_ref};
%model_names = {'wtp_minus_ref'};

num_models = length(models);
lls_all_models = zeros(14, num_models);

for i=1:num_models
    v = 6.25;
    if i < 4
        prior_softmax = struct('mean',0,'variance',v);
    else
        prior_softmax = struct('mean',zeros(2,1),'variance',v);
    end
    
    model_name = model_names{i};
    fname_softmax = ['results/fixed_effects_',model_name,'.mat'];

    model = models{i};
    cbm_lap(data, model, prior_softmax, fname_softmax);

    % Let%s take a look at the file saved by the cbm_lap:
    fname = load(fname_softmax);
    cbm = fname.cbm;
    lls = cbm.math.loglik;
    lls_all_models(:,i) = lls;
end

df_lls = array2table(lls_all_models,'VariableNames',model_names);

