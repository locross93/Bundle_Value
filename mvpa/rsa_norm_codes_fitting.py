from mvpa2.suite import *
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/Users/locro/Documents/Bundle_Value/mvpa/")
import os
os.chdir("/Users/locro/Documents/Bundle_Value/mvpa/")
import mvpa_utils 
import numpy as np
import time
from lmfit import Model, Parameters
import statsmodels.api as sm
import statsmodels.tools.eval_measures

import pdb

def compute_divisive_norm(values, beta, sigma):
    avg_value = np.mean(values)
    div_normed_values = (beta + values.astype(float)) / (sigma + avg_value)
    
    return div_normed_values


def rsa_regression(temp_fmri, partial_dsms, target_dsm, num_params):
    #model_dsms = partial_dsms
    #model_dsms.append(target_dsm)
    model_dsms = partial_dsms + [target_dsm]
    model_dsm_array = np.column_stack((model_dsms))
    X = sm.add_constant(model_dsm_array)
    mod = sm.OLS(temp_fmri, X)
    res = mod.fit()
    target_dsm_tval = res.tvalues[-1]
    df_modelwc = res.df_model + 1 + num_params
    nobs = nobs = X.shape[0]
    bic = statsmodels.tools.eval_measures.bic(res.llf, nobs, df_modelwc)
    adj_r2 = 1 - (nobs-1)/(res.df_resid-num_params) * (1-res.rsquared)
    coefs = res.params
    
    return target_dsm_tval, bic, adj_r2, coefs


def nonlinear_rsa_regression(temp_fmri, partial_dsms, target_dsm, num_params):
    # Combine all DSMs
    model_dsms = partial_dsms + [target_dsm]
    model_dsm_array = np.column_stack(model_dsms)
    
    # Define the nonlinear model function
    def nonlinear_model(x, a0, a1, a2, a3, a4):
        #return a0 + a1 * x[:, 0] + a2 * x[:, 1] + a3 * x[:, 2] + b * np.exp(c * x[:, 3])
        return a0 + a1 * x[:, 0] + a2 * x[:, 1] + a3 * x[:, 2] + a4 * x[:, 3]
    
    # Create the LMfit model
    lmfit_model = Model(nonlinear_model)
    
    # Set up parameters with initial guesses
    params = Parameters()
    params.add('a0', value=0)
    params.add('a1', value=1)
    params.add('a2', value=1)
    params.add('a3', value=1)
    params.add('a4', value=1)
    
    # Fit the model
    result = lmfit_model.fit(temp_fmri, params, x=model_dsm_array)
    
    
    # Calculate statistics
    #bic = statsmodels.tools.eval_measures.bic(result.chisqr, nobs, df_modelwc)
    bic = result.bic
    
    # Calculate adjusted R-squared
    nobs = len(temp_fmri)
    df_modelwc = len(result.params) + num_params
    ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
    ss_residual = np.sum(result.residual**2)
    r_squared = 1 - (ss_residual / ss_total)
    adj_r2 = 1 - (1 - r_squared) * ((nobs - 1) / (nobs - df_modelwc - 1))
    
    coefs = [result.params['a0'].value, result.params['a1'].value, result.params['a2'].value, result.params['a3'].value, result.params['a4'].value]
    
    return bic, adj_r2, coefs
    

def nonlinear_rsa_regression2(temp_fmri, partial_dsms, abs_value, num_params, trial_type_inds, btwn_day_inds=None):

    # Define the nonlinear model function
    def nonlinear_model_sigma(x, a0, a1, a2, a3, a4, sigma, abs_value, partial_dsms):
        norm_values = np.zeros(len(abs_value))
        
        # Divisively normalize values
        for trial_inds in trial_type_inds:
            avg_value = np.mean(abs_value[trial_inds])
            norm_values[trial_inds] = abs_value[trial_inds] / (sigma + avg_value)
        
        ds_value = dataset_wizard(norm_values, targets=np.zeros(len(norm_values)))
        dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
        value_dsm = dsm(ds_value)
        
        if ranked:
            value_dsm = rankdata(value_dsm)
        else:
            value_dsm = value_dsm.samples.reshape(-1)
            
        if remove_within_day and btwn_day_inds is not None:
            value_dsm = value_dsm[btwn_day_inds]
        
        # transpose partial_dsms
        partial_dsms = partial_dsms.T
        # concatenate partial_dsms with value_dsm by column
        x = np.column_stack((partial_dsms, value_dsm))
        
        return a0 + a1 * x[:, 0] + a2 * x[:, 1] + a3 * x[:, 2] + a4 * x[:, 3] 
    
    # Create the LMfit model
    lmfit_model = Model(nonlinear_model_sigma, independent_vars=['x', 'abs_value', 'partial_dsms'])
    
    # Set up parameters with initial guesses
    params = Parameters()
    params.add('a0', value=0)
    params.add('a1', value=1)
    params.add('a2', value=1)
    params.add('a3', value=1)
    params.add('a4', value=1)
    params.add('sigma', value=0, min=0)  # sigma should be non-negative
    
    # Fit the model
    result = lmfit_model.fit(temp_fmri, params, x=np.zeros(len(temp_fmri)), 
                             abs_value=abs_value, partial_dsms=partial_dsms)
    
    # Calculate statistics
    bic = result.bic
    
    # Calculate adjusted R-squared
    nobs = len(temp_fmri)
    df_modelwc = len(result.params) + num_params
    ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
    ss_residual = np.sum(result.residual**2)
    r_squared = 1 - (ss_residual / ss_total)
    adj_r2 = 1 - (1 - r_squared) * ((nobs - 1) / (nobs - df_modelwc - 1))
    
    coefs = [result.params['a0'].value, result.params['a1'].value, result.params['a2'].value, 
             result.params['a3'].value, result.params['a4'].value, result.params['sigma'].value]
    
    return bic, adj_r2, coefs

bundle_path = '/Users/locro/Documents/Bundle_Value/'

#subj_list = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']
#subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
subj_list = ['104']

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']

mask_loop = ['Frontal_Med_Orb', 'OFCmed', 'Frontal_Sup_Medial']
mask_names = ['vmPFC','OFCmed','dmPFC']

square_dsm_bool = False
ranked = False
remove_within_day = True

model = 'Divisive'

for subj in subj_list:
    start_time = time.time()
    print(subj)

    fmri_dsms_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/fmri_dsm_list'
    fmri_dsm_list = h5load(fmri_dsms_file)
    
    if int(subj) < 104:
        target_dsms_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/target_dsms.csv'
        target_dsms_df = pd.read_csv(target_dsms_file)
        target_dsms = target_dsms_df.to_dict()
        
        subj_info_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/info_dict'
        subj_info_dict = h5load(subj_info_file+'_list')
        abs_value = subj_info_dict[0]
        trial_categ = subj_info_dict[1]
        sitem_inds = subj_info_dict[2]
        bundle_inds = subj_info_dict[3]
        run_array = subj_info_dict[4]
        day_array = subj_info_dict[5]
        
        #item or bundle?
        item_or_bundle = trial_categ
        item_or_bundle[sitem_inds] = 0
        item_or_bundle[bundle_inds] = 1
        assert np.max(item_or_bundle) == 1
        
        num_trials = len(item_or_bundle)
        ds_trial_cat = dataset_wizard(item_or_bundle, targets=np.zeros(num_trials))
        dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
        res_trial_cat = dsm(ds_trial_cat)
        if ranked:
            res_trial_cat = rankdata(res_trial_cat)
        else:
            res_trial_cat = res_trial_cat.samples.reshape(-1)
        target_dsms['item_or_bundle'] = res_trial_cat

        #choice 
        choice = mvpa_utils.get_fmri_choices(subj, conditions)
        ds_choice = dataset_wizard(choice, targets=np.zeros(num_trials))
        dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
        res_choice = dsm(ds_choice)
        if ranked:
            res_choice = rankdata(res_choice)
        else:
            res_choice = res_choice.samples.reshape(-1)
        target_dsms['choice'] = res_choice
        
        #left vs right choice
        lr_choice_list = np.genfromtxt(bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/task_info/lr_choice.txt')
        lr_choice = lr_choice_list[:,1]
        #if only individual item trials, only include these trials
        if len(conditions) < 5:
            lr_choice = lr_choice[inds_in_conds]
        ds_lr_choice = dataset_wizard(lr_choice, targets=np.zeros(num_trials))
        dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
        res_lr_choice = dsm(ds_lr_choice)
        if ranked:
            res_lr_choice = rankdata(res_lr_choice)
        else:
            res_lr_choice = res_lr_choice.samples.reshape(-1)
        target_dsms['lr_choice'] = res_lr_choice

        if remove_within_day:
            res_day = np.array(target_dsms['day'].values())
            if ranked:
                day_values = np.unique(res_day)
                high_rank = np.max(day_values)
                btwn_day_inds = np.where(res_day == high_rank)[0]
            else:
                btwn_day_inds = np.where(res_day == 1)[0]
    else:
        target_dsms_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/target_dsms'
        target_dsms = h5load(target_dsms_file)

        subj_info_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/info_dict'
        subj_info_dict = h5load(subj_info_file)
        abs_value = subj_info_dict['abs_value']
        trial_categ = subj_info_dict['trial_categ']
        sitem_inds = subj_info_dict['sitem_inds']
        bundle_inds = subj_info_dict['bundle_inds']
        run_array = subj_info_dict['run_array']
        day_array = subj_info_dict['day_array']
    
        if remove_within_day:
            res_day = target_dsms['day']
            if ranked:
                day_values = np.unique(res_day)
                high_rank = np.max(day_values)
                btwn_day_inds = np.where(res_day == high_rank)[0]
            else:
                btwn_day_inds = np.where(res_day == 1)[0]

    # set up rsa regression 
    norm_values = np.zeros([len(abs_value)])

    if model == 'Divisive':
        norm_values[sitem_inds] = compute_divisive_norm(abs_value[sitem_inds], beta=0, sigma=0) 
        norm_values[bundle_inds] = compute_divisive_norm(abs_value[bundle_inds], beta=0, sigma=0) 
        num_params = 2

    ds_value = dataset_wizard(norm_values, targets=np.zeros(len(norm_values)))
    dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
    res_value = dsm(ds_value)
    if ranked:
        res_value = rankdata(res_value)
    else:
        res_value = res_value.samples.reshape(-1)

    model_dsm_names = ['choice','item_or_bundle','lr_choice']
    #for mask_num in range(len(mask_loop)):
    for mask_num in range(1):
        partial_dsms = [target_dsms[model_dsm][btwn_day_inds] for model_dsm in model_dsm_names]
        if remove_within_day:
            temp_fmri = fmri_dsm_list[mask_num][btwn_day_inds]
            temp_model = res_value[btwn_day_inds]
        else:
            temp_fmri = fmri_dsm_list[mask_num]
            temp_model = res_value
            btwn_day_inds = None
        linear_tval, linear_bic, linear_adj_r2, linear_coefs = rsa_regression(temp_fmri, partial_dsms, temp_model, num_params)
        nonlinear_bic, nonlinear_adj_r2, nonlinear_coefs = nonlinear_rsa_regression(temp_fmri, partial_dsms, temp_model, num_params)
        trial_type_inds = [sitem_inds, bundle_inds]
        sigma_bic, sigma_adj_r2, sigma_coefs = nonlinear_rsa_regression2(temp_fmri, partial_dsms, abs_value, num_params, trial_type_inds, btwn_day_inds)