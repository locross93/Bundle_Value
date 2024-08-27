function [div_normed_values, div_normed_ref] = compute_divisive_norm_behavior(values, ref_amount)
avg_value = mean(values);
div_normed_values = values ./ avg_value;
div_normed_ref = ref_amount ./ avg_value;
end