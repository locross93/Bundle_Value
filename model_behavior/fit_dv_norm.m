function [loglik] = fit_dv_norm(parameters,subj)
values = subj.Value;
ref_amount = subj.Ref_Amount;
day_array = subj.Day;
trial_type = subj.Trial_type;
sitem_inds = find(trial_type == 0);
bundle_inds = find(trial_type == 1);

dvnorm_values = zeros(size(values));
dvnorm_ref = zeros(size(ref_amount));

for day=1:3
    day_inds = find(day_array == day);
    day_sinds = intersect(day_inds, sitem_inds);
    day_binds = intersect(day_inds, bundle_inds);
    [dvnorm_values(day_sinds) , dvnorm_ref(day_sinds)] = compute_divisive_norm_behavior(values(day_sinds), ref_amount(day_sinds)); 
    [dvnorm_values(day_binds) , dvnorm_ref(day_binds)] = compute_divisive_norm_behavior(values(day_binds), ref_amount(day_binds));
end

subj.Value = dvnorm_values;
subj.Ref_Amount = dvnorm_ref;
loglik = fit_softmax(parameters,subj);

end

function [div_normed_values, div_normed_ref] = compute_divisive_norm_behavior(values, ref_amount)
avg_value = mean(values);
div_normed_values = values ./ avg_value;
div_normed_ref = ref_amount ./ avg_value;
end