function [loglik] = fit_softmax(parameters,subj,nd_params)
% Set the default value for nd_params to false if it is not provided
if nargin < 3
    nd_params = false;
end

if nd_params
    nd_beta  = parameters(1); % normally-distributed beta
    beta = exp(nd_beta); % beta (transformed to be between zero and inf)
else
    beta = parameters(1);
end

values = subj.Value;
ref_amount = subj.Ref_Amount;
choices = subj.Choice;

% number of trials
T = size(choices,1);

% to save probability of choice. Currently NaNs, will be filled below
p = nan(T,1);

choice_probs = softmax_actions(values, ref_amount, beta);
for j=1:T
    choice = choices(j);
    if choice == 1
        p(j) = choice_probs(j, 1);
    elseif choice == 0
        p(j) = choice_probs(j, 2);
    end
end

% log-likelihood is defined as the sum of log-probability of choice data 
% (given the parameters).
loglik = sum(log(p));
end

function choice_probs = softmax_actions(values, ref_amount, beta)
action_vals = [values, ref_amount];
choice_probs = zeros(size(action_vals));
numer = exp(action_vals.*beta);
denom = sum(numer, 2);
choice_probs = numer ./ denom;
end
