function [loglik] = fit_rw_norm(parameters,subj,by_day,nd_params)
% Set the default value for by_day to true if it is not provided
if nargin < 3
    by_day = true;
end

% Set the default value for nd_params to true if it is not provided
if nargin < 4
    nd_params = true;
end

if nd_params
    nd_alpha = parameters(2); % normally-distributed alpha
    alpha = 1/(1+exp(-nd_alpha)); % alpha (transformed to be between zero and one)
else
    alpha = parameters(2);
end

% Rescorla Wagner type update of EV for divisive normalization
values = subj.Value;
ref_amount = subj.Ref_Amount;
day_array = subj.Day;
trial_type = subj.Trial_type;
sitem_inds = find(trial_type == 0);
bundle_inds = find(trial_type == 1);

rwnorm_values = zeros(size(values));
rwnorm_ref = zeros(size(ref_amount));

if by_day
    for day=1:3
        day_inds = find(day_array == day);
        day_sinds = intersect(day_inds, sitem_inds);
        day_binds = intersect(day_inds, bundle_inds);
        sitem_ev0 = mean(values(day_sinds));
        bundle_ev0 = mean(values(day_binds));
        [rwnorm_values(day_sinds) , rwnorm_ref(day_sinds)] = compute_rw_norm_behavior(values(day_sinds), ref_amount(day_sinds), alpha, sitem_ev0); 
        [rwnorm_values(day_binds) , rwnorm_ref(day_binds)] = compute_rw_norm_behavior(values(day_binds), ref_amount(day_binds), alpha, bundle_ev0);
    end
else
    sitem_ev0 = mean(values(sitem_inds));
    bundle_ev0 = mean(values(bundle_inds));
    [rwnorm_values(sitem_inds) , rwnorm_ref(sitem_inds)] = compute_rw_norm_behavior(values(sitem_inds), ref_amount(sitem_inds), alpha, sitem_ev0); 
    [rwnorm_values(bundle_inds) , rwnorm_ref(bundle_inds)] = compute_rw_norm_behavior(values(bundle_inds), ref_amount(bundle_inds), alpha, bundle_ev0); 
end

subj.Value = rwnorm_values;
subj.Ref_Amount = rwnorm_ref;
loglik = fit_softmax(parameters,subj);

end

function [rw_normed_values, rw_normed_ref] = compute_rw_norm_behavior(values, ref_amount, alpha, ev0)
ev = ev0;
rw_normed_values = zeros(size(values));
rw_normed_ref = zeros(size(ref_amount));

for i=1:length(values)
    val = values(i);
    ref = ref_amount(i);
    pe = val - ev;
    ev = ev + (alpha*pe);
    rw_normed_values(i) = val / ev;
    rw_normed_ref(i) = ref / ev;
end

end