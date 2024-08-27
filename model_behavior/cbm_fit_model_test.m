% load data
data = load('data/all_subj_data.mat');
data = data.data;

v = 6.25;
%prior_softmax = struct('mean',0,'variance',v);
prior_softmax = struct('mean',zeros(2,1),'variance',v);
fname_softmax = 'results/softmax_test_rw.mat';

%cbm_lap(data, @fit_softmax, prior_softmax, fname_softmax);
%cbm_lap(data, @fit_dv_norm, prior_softmax, fname_softmax);
%cbm_lap(data, @fit_mc_norm, prior_softmax, fname_softmax);
cbm_lap(data, @fit_rw_norm, prior_softmax, fname_softmax);

% Let%s take a look at the file saved by the cbm_lap:
fname = load(fname_softmax);
cbm = fname.cbm;
lls = cbm.math.loglik;