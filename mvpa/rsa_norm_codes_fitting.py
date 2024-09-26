#from mvpa2.suite import *
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/Users/locro/Documents/Bundle_Value/mvpa/")
import os
os.chdir("/Users/locro/Documents/Bundle_Value/mvpa/")
#import mvpa_utils 
import numpy as np
import time
from lmfit import Model, Parameters
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist

import os
import json

import pdb
    

def nonlinear_rsa_regression2(temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds):

    # Define the nonlinear model function
    def nonlinear_model_sigma(x, a0, a1, a2, a3, a4, sigma, abs_value, partial_dsms):
        norm_values = np.zeros(len(abs_value))
        
        # Divisively normalize values
        for trial_inds in trial_type_inds:
            avg_value = np.mean(abs_value[trial_inds])
            norm_values[trial_inds] = abs_value[trial_inds] / (sigma + avg_value)
        
        ds_value = norm_values.reshape(-1, 1)
        value_dsm = pdist(ds_value, metric='euclidean')
        
        if ranked:
            value_dsm = rankdata(value_dsm)
            
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
    num_params = len([p for p in result.params.values() if not p.vary])
    df_modelwc = len(result.params) - num_params # degrees of freedom
    ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
    ss_residual = np.sum(result.residual**2)
    r_squared = 1 - (ss_residual / ss_total)
    adj_r2 = 1 - (1 - r_squared) * ((nobs - 1) / (nobs - df_modelwc - 1))

    fit_dict = {'bic': bic, 'adj_r2': adj_r2, 
                    'a0': result.params['a0'].value, 'a1': result.params['a1'].value, 
                    'a2': result.params['a2'].value, 'a3': result.params['a3'].value,
                    'a4': result.params['a4'].value, 'sigma': result.params['sigma'].value}
    
    return fit_dict


def nonlinear_rsa_regression3(temp_fmri, partial_dsms, abs_value, btwn_day_inds, item1_value, item2_value):

    # Define the nonlinear model function
    def nonlinear_model_sigma(x, a0, a1, a2, a3, sigma, w, item1_value, item2_value, partial_dsms):

        # Divisively normalize values
        #norm_values = abs_value / (sigma + abs_value*w)
        norm_values = (item1_value + item2_value) / (sigma + w*(item1_value + item2_value))
        
        ds_value = norm_values.reshape(-1, 1)
        value_dsm = pdist(ds_value, metric='euclidean')
        
        if ranked:
            value_dsm = rankdata(value_dsm)
            
        if remove_within_day and btwn_day_inds is not None:
            value_dsm = value_dsm[btwn_day_inds]
        
        # transpose partial_dsms
        partial_dsms = partial_dsms.T
        # concatenate partial_dsms with value_dsm by column
        x = np.column_stack((partial_dsms, value_dsm))
        
        return a0 + a1 * x[:, 0] + a2 * x[:, 1] + a3 * x[:, 2] + x[:, 3]  
    
    # Create the LMfit model
    lmfit_model = Model(nonlinear_model_sigma, independent_vars=['x', 'item1_value', 'item2_value', 'partial_dsms'])
    
    # Set up parameters with initial guesses
    params = Parameters()
    params.add('a0', value=0)
    params.add('a1', value=1)
    params.add('a2', value=1)
    params.add('a3', value=1)
    params.add('sigma', value=5, min=0)  # sigma should be non-negative
    params.add('w', value=1, min=0)  # w should be non-negative
    
    # Fit the model
    result = lmfit_model.fit(temp_fmri, params, x=np.zeros(len(temp_fmri)), 
                             item1_value=item1_value, item2_value=item2_value, partial_dsms=partial_dsms)
    
    # Calculate statistics
    bic = result.bic
    
    # Calculate adjusted R-squared
    nobs = len(temp_fmri)
    num_params = len([p for p in result.params.values() if not p.vary])
    df_modelwc = len(result.params) - num_params # degrees of freedom
    ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
    ss_residual = np.sum(result.residual**2)
    r_squared = 1 - (ss_residual / ss_total)
    adj_r2 = 1 - (1 - r_squared) * ((nobs - 1) / (nobs - df_modelwc - 1))

    fit_dict = {'bic': bic, 'adj_r2': adj_r2,
                    'a0': result.params['a0'].value, 'a1': result.params['a1'].value, 
                    'a2': result.params['a2'].value, 'a3': result.params['a3'].value,
                    'sigma': result.params['sigma'].value, 'w': result.params['w'].value}
    
    return fit_dict


def abs_value_rsa_regression(temp_fmri, partial_dsms, abs_value, btwn_day_inds, item1_value, item2_value):

    # Define the nonlinear model function
    def nonlinear_model_sigma(x, a0, a1, a2, a3, sigma, abs_value, partial_dsms):

        # Divisively normalize values
        norm_values = abs_value / sigma 
        
        ds_value = norm_values.reshape(-1, 1)
        value_dsm = pdist(ds_value, metric='euclidean')
        
        if ranked:
            value_dsm = rankdata(value_dsm)
            
        if remove_within_day and btwn_day_inds is not None:
            value_dsm = value_dsm[btwn_day_inds]
        
        # transpose partial_dsms
        partial_dsms = partial_dsms.T
        # concatenate partial_dsms with value_dsm by column
        x = np.column_stack((partial_dsms, value_dsm))
        
        return a0 + a1 * x[:, 0] + a2 * x[:, 1] + a3 * x[:, 2] + x[:, 3] 
    
    # Create the LMfit model
    lmfit_model = Model(nonlinear_model_sigma, independent_vars=['x', 'abs_value', 'partial_dsms'])
    
    # Set up parameters with initial guesses
    params = Parameters()
    params.add('a0', value=0)
    params.add('a1', value=1)
    params.add('a2', value=1)
    params.add('a3', value=1)
    params.add('sigma', value=5, min=0)  # sigma should be non-negative
    
    # Fit the model
    result = lmfit_model.fit(temp_fmri, params, x=np.zeros(len(temp_fmri)), 
                             abs_value=abs_value, partial_dsms=partial_dsms)
    
    # Calculate statistics
    bic = result.bic
    
    # Calculate adjusted R-squared
    nobs = len(temp_fmri)
    num_params = len([p for p in result.params.values() if not p.vary])
    df_modelwc = len(result.params) - num_params # degrees of freedom
    ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
    ss_residual = np.sum(result.residual**2)
    r_squared = 1 - (ss_residual / ss_total)
    adj_r2 = 1 - (1 - r_squared) * ((nobs - 1) / (nobs - df_modelwc - 1))
    
    fit_dict = {'bic': bic, 'adj_r2': adj_r2,
                    'a0': result.params['a0'].value, 'a1': result.params['a1'].value, 
                    'a2': result.params['a2'].value, 'a3': result.params['a3'].value,
                    'sigma': result.params['sigma'].value}
    
    return fit_dict


def plot_multi_mask_normalization_comparison(subj, results_dict, mask_names):
    # Create a DataFrame
    data = []
    for mask in mask_names:
        for norm_type, results in results_dict[mask].items():
            data.append({
                'Mask': mask,
                'Normalization': norm_type,
                'Adjusted R2': results['adj_r2']
            })
    
    df = pd.DataFrame(data)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
    sns.barplot(x='Mask', y='Adjusted R2', hue='Normalization', data=df, palette='Set2', hue_order=hue_order, ax=ax)
    
    # Customize the plot
    ax.set_title('Sub{0:03d} RSA Normalized Codes'.format(subj), fontsize=16)
    ax.set_xlabel('ROI', fontsize=12)
    ax.set_ylabel('RSA Adj R2', fontsize=12)
    
    # Adjust legend
    plt.legend(title='Normalization', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    return fig, ax
    
    
def save_results(subj, results_dict, mask_names):
    subj_dir = os.path.join(bundle_path, 'mvpa', 'analyses', 'sub'+str(subj))
    
    # Save parameter fits and statistics
    results_file = os.path.join(subj_dir, 'rsa_norm_results_new.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=4)

    # Save plot
    plot_file = os.path.join(subj_dir, 'rsa_norm_plot_new.png')
    fig, ax = plot_multi_mask_normalization_comparison(subj, results_dict, mask_names)
    
    if ax.has_data():
        fig.savefig(plot_file, bbox_inches='tight')
        plt.close(fig)
        print("Plot saved successfully: {}".format(plot_file))
    else:
        print("Error: Plot is empty. No data found.")

    # Print some debugging information
    print("Sample of results:")
    print(pd.DataFrame(results_dict).head())


bundle_path = '/Users/locro/Documents/Bundle_Value/'

subj_list = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']
#subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
subj_list = ['105','106','107','108','109','110','111','112','113','114']

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']

mask_loop = ['Frontal_Med_Orb', 'OFCmed', 'Frontal_Sup_Medial']
mask_names = ['vmPFC','OFCmed','dmPFC']

square_dsm_bool = False
ranked = False
remove_within_day = True

for subj in subj_list:
    start_time = time.time()
    print(subj)

    fmri_dsms_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/fmri_dsm_list_np.npz'
    data = np.load(fmri_dsms_file)
    fmri_dsm_list = [data['arr_{}'.format(i)] for i in range(0, len(data))]
    
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
        target_dsms_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/target_dsms_np.npz'
        target_dsms = np.load(target_dsms_file)
        target_dsms = {key: target_dsms[key] for key in target_dsms}
        #target_dsms = h5load(target_dsms_file)
        
        subj_info_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/info_all_trials.csv'
        subj_info_dict = pd.read_csv(subj_info_file)
        abs_value = subj_info_dict['Stimulus Value'].values
        trial_categ = subj_info_dict['Trial Categ'].values
        item1_value = subj_info_dict['Item 1 Value'].values
        item2_value = subj_info_dict['Item 2 Value'].values
        sitem_inds = np.where(trial_categ == 0)[0]
        bundle_inds = np.where(trial_categ == 1)[0]
    
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

    model_dsm_names = ['choice','item_or_bundle','lr_choice']
    results_dict = {}
    for mask_num, mask_name in enumerate(mask_names):
        partial_dsms = [target_dsms[model_dsm][btwn_day_inds] for model_dsm in model_dsm_names]
        if remove_within_day:
            temp_fmri = fmri_dsm_list[mask_num][btwn_day_inds]
        else:
            temp_fmri = fmri_dsm_list[mask_num]
            btwn_day_inds = None
    
        trial_type_inds = [sitem_inds, bundle_inds]
    
        # Perform regressions and collect results
        results_dict[mask_name] = {
            'Divisive by Cat': nonlinear_rsa_regression2(temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds),
            'Sigma and w': nonlinear_rsa_regression3(temp_fmri, partial_dsms, abs_value, btwn_day_inds, item1_value, item2_value),
            'Absolute': abs_value_rsa_regression(temp_fmri, partial_dsms, abs_value, btwn_day_inds, item1_value, item2_value)
        }
    
        
    # Save results and plot
    save_results(int(subj), results_dict, mask_names)