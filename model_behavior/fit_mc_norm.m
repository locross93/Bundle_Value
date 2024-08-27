function [loglik] = fit_mc_norm(parameters,subj)
values = subj.Value;
ref_amount = subj.Ref_Amount;
day_array = subj.Day;
trial_type = subj.Trial_type;
sitem_inds = find(trial_type == 0);
bundle_inds = find(trial_type == 1);

mcnorm_values = zeros(size(values));
mcnorm_ref = zeros(size(ref_amount));

for day=1:3
    day_inds = find(day_array == day);
    day_sinds = intersect(day_inds, sitem_inds);
    day_binds = intersect(day_inds, bundle_inds);
    [mcnorm_values(day_sinds) , mcnorm_ref(day_sinds)] = compute_monte_carlo_norm_behavior(values(day_sinds), ref_amount(day_sinds)); 
    [mcnorm_values(day_binds) , mcnorm_ref(day_binds)] = compute_monte_carlo_norm_behavior(values(day_binds), ref_amount(day_binds));
end

subj.Value = mcnorm_values;
subj.Ref_Amount = mcnorm_ref;
loglik = fit_softmax(parameters,subj);

end

function [mc_normed_values, mc_normed_ref] = compute_monte_carlo_norm_behavior(values, ref_amount)
mc_normed_values = zeros(size(values));
mc_normed_ref = zeros(size(ref_amount));

for i=1:length(values)
    val = values(i);
    ref = ref_amount(i);
    mc_ev = mean(values(1:i));
    if mc_ev == 0
        mc_normed_values(i) = 0;
        mc_normed_ref(i) = ref;
    else
        mc_normed_values(i) = val / mc_ev;
        mc_normed_ref(i) = ref / mc_ev;
    end
end

end