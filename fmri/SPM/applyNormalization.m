function [normed_value, trial_cat_out] = applyNormalization(trial_value, trial_cat, method)
% APPLYNORMALIZATION Allows swapping in different normalization approaches 
% based on the method argument.
%
% Usage:
%   [normed_value, trial_cat_out] = applyNormalization(trial_value, trial_cat, method)
%
% Inputs:
%   trial_value   : Vector of value parameters for each trial
%   trial_cat     : Vector indicating trial category (e.g., 0 = single item, 1 = bundle)
%   method        : String specifying which normalization method to apply
%
% Outputs:
%   normed_value  : Normalized values
%   trial_cat_out : Possibly modified category vector if your method needs it
%
% Author: Your Name

    % Default output for trial_cat in case you need to modify it 
    trial_cat_out = trial_cat; 

    % Figure out the two sets of indices (item vs. bundle in your case)
    sitem_inds = trial_cat == 0;  % single-item indices
    bundle_inds = trial_cat == 1; % bundle indices

    switch lower(method)
        case 'divide_by_mean'
            % Divisive normalize by category mean
            normed_value = trial_value;
            
            % Single items
            normed_value(sitem_inds) = ...
                normed_value(sitem_inds) / mean(normed_value(sitem_inds));

            % Bundles
            normed_value(bundle_inds) = ...
                normed_value(bundle_inds) / mean(normed_value(bundle_inds));

        case 'zscore_within_category'
            % Example: Z-scoring within each category
            normed_value = trial_value;
            
            % Single items
            normed_value(sitem_inds) = zscore(normed_value(sitem_inds));

            % Bundles
            normed_value(bundle_inds) = zscore(normed_value(bundle_inds));

        otherwise
            % Fallback if method is unrecognized
            warning(['Normalization method "', method, '" not recognized. ',...
                     'Returning unmodified values.']);
            normed_value = trial_value;
    end
end
