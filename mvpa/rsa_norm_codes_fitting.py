#from mvpa2.suite import *
import matplotlib.pyplot as plt
import sys
#sys.path.insert(0, "/Users/locro/Documents/Bundle_Value/mvpa/")
import os
#os.chdir("/Users/locro/Documents/Bundle_Value/mvpa/")
#import mvpa_utils 
import numpy as np
import time
from lmfit import Model, Parameters
from lmfit.model import save_modelresult

import statsmodels.api as sm
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import pdist

import os
import json
import pickle

import pdb
from pathlib import Path

#Divisive By Type and V
def nonlinear_rsa(info, params, temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds):

    # Define the nonlinear model function
    def nonlinear_model_sigma(a0, a1, a2, a3, b0, b1, sigma, w1, w_v, w_avg, abs_value, partial_dsms):
        norm_values = np.zeros(len(abs_value))
        
        if info['model']=='Interaction Full(w)':
            # Interact values
            for trial_inds in trial_type_inds:
                norm_values[trial_inds] = b0 * trial_categ[trial_inds] + b1 * abs_value[trial_inds] + w_v * abs_value[trial_inds]*trial_categ[trial_inds]
                #logan's suggestion
                #norm_values[trial_inds] = b0 * trial_categ[trial_inds] + b1 * abs_value[trial_inds]*(1-trial_categ[trial_inds]) + w_v * abs_value[trial_inds]*trial_categ[trial_inds]
        else:
            # Divisively normalize values
            for trial_inds in trial_type_inds:
                avg_value = np.mean(abs_value[trial_inds])
                norm_values[trial_inds] = b0 * trial_categ[trial_inds] + ((b1 * abs_value[trial_inds]) / (sigma + w_avg * avg_value  + w1 * trial_categ[trial_inds] + w_v *abs_value[trial_inds]))
            
        
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
    lmfit_model = Model(nonlinear_model_sigma, independent_vars=['abs_value', 'partial_dsms'])
    
    # Fit the model
    result = lmfit_model.fit(temp_fmri, params, 
                             abs_value=abs_value, partial_dsms=partial_dsms)
    
    # Calculate statistics
    
    # Calculate adjusted R-squared
          
    ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
    ss_residual = np.sum(result.residual**2)
    r_squared = 1 - (ss_residual / ss_total)
    adj_r2 = 1 - (1 - r_squared) * ((result.ndata - 1) / (result.nfree - 1))

    #fit_dict = result.params.valuesdict()
    fit_dict = result.params.create_uvars(result.covar)
    fit_dict['r2'] = r_squared
    fit_dict['bic'] = result.bic
    fit_dict['adj_r2'] = adj_r2
    
    print('\n')
    print(info['subj'])
    print(info['mask'])
    print(info['model'])
    print(result.fit_report()) 
    print('\n')

    subj_dir = os.path.join(bundle_path, 'mvpa', 'analyses', 'sub'+str(info['subj']))
    
    # Save parameter fits and statistics
    with open(os.path.join(subj_dir, "".join(['rsa_norm_results_11_07_24-', info['model'], info['mask'],'.json'])), 'w') as f:
        result.params.dump(f)
        
    # Save model fit and statistics
    #save_modelresult(result, os.path.join(subj_dir, "".join(['rsa_norm_results_11_07_24-', info['model'], info['mask'],'.sav'])))
    
    return fit_dict


def nonlinear_rsa_items(info, params, temp_fmri, partial_dsms, abs_value, btwn_day_inds, item1_value, item2_value):

    # Define the nonlinear model function
    def nonlinear_model_sigma(x, a0, a1, a2, a3, b0, b1, sigma, w1, w_v, w_avg, item1_value, item2_value, partial_dsms):
        norm_values = np.zeros(len(abs_value))
        
        
        if info['model']=='Items Interaction':
            # Divisively normalize values
            for trial_inds in trial_type_inds:
                avg_value = np.mean(abs_value[trial_inds])
                sum_const_values = (item1_value[trial_inds] + item2_value[trial_inds])
                norm_values[trial_inds] =  b0*trial_categ[trial_inds]  + b1*sum_const_values + w_v*sum_const_values*trial_categ[trial_inds] + w_avg*avg_value
        else:    
            # Divisively normalize values
            for trial_inds in trial_type_inds:
                avg_value = np.mean(abs_value[trial_inds])
                sum_const_values = (item1_value[trial_inds] + item2_value[trial_inds])
                norm_values[trial_inds] =  b0*trial_categ[trial_inds]  + (b1*sum_const_values / (sigma + w1*trial_categ[trial_inds] + w_v*sum_const_values + w_avg*avg_value))
        
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

    # Fit the model
    result = lmfit_model.fit(temp_fmri, params, x=np.zeros(len(temp_fmri)), 
                             item1_value=item1_value, item2_value=item2_value, partial_dsms=partial_dsms)
    
    # # Calculate statistics
    # bic = result.bic
    
    # # Calculate adjusted R-squared
    # nobs = len(temp_fmri)
    # num_params = len([p for p in result.params.values() if not p.vary])
    # df_modelwc = len(result.params) - num_params # degrees of freedom
    # ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
    # ss_residual = np.sum(result.residual**2)
    # r_squared = 1 - (ss_residual / ss_total)
    # adj_r2 = 1 - (1 - r_squared) * ((nobs - 1) / (nobs - df_modelwc - 1))

    # fit_dict = {'r2': r_squared, 'bic': bic, 'adj_r2': adj_r2,
    #                 'a0': result.params['a0'].value, 'a1': result.params['a1'].value, 
    #                 'a2': result.params['a2'].value, 'a3': result.params['a3'].value,
    #                 'w1': result.params['w1'].value, 'w2': result.params['w2'].value, 'sigma': result.params['sigma'].value}
    # print('AverageByType and V and Parameters, Constituent Items\n')
    # print(result.fit_report())
    # print('\n') 
    # return fit_dict

    # Calculate adjusted R-squared
           
    ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
    ss_residual = np.sum(result.residual**2)
    r_squared = 1 - (ss_residual / ss_total)
    adj_r2 = 1 - (1 - r_squared) * ((result.ndata - 1) / (result.nfree - 1))
    
    #fit_dict = result.params.valuesdict()
    fit_dict = result.params.create_uvars(result.covar)
    fit_dict['r2'] = r_squared
    fit_dict['bic'] = result.bic
    fit_dict['adj_r2'] = adj_r2
    
    print('\n')
    print(info['subj'])
    print(info['mask'])
    print(info['model'])
    print(result.fit_report()) 
    print('\n')

    subj_dir = os.path.join(bundle_path, 'mvpa', 'analyses', 'sub'+str(info['subj']))
    
    # Save parameter fits and statistics for this model
    with open(os.path.join(subj_dir, "".join(['rsa_norm_results_11_07_24-', info['model'], info['mask'],'.json'])), 'w') as f:
        result.params.dump(f)
        
    # Save model fit and statistics
    #save_modelresult(result, os.path.join(subj_dir, "".join(['rsa_norm_results_11_07_24-', info['model'], info['mask'],'.sav'])))
    
    return fit_dict



def plot_multi_mask_normalization_comparison(subj, results_dict, mask_names):
    # Create a DataFrame
    data = []
    for norm_type in results_dict.keys():
        for mask, results in results_dict[norm_type].items():
            #for norm_type, results in results_dict[mask_names].items():

            data.append({
                'Mask': mask,
                'Normalization': norm_type,
                'Adjusted R2': results['adj_r2']
            })
 
    df = pd.DataFrame(data)
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    #hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
    sns.barplot(x='Mask', y='Adjusted R2', hue='Normalization', data=df, palette='Set2', ax=ax)
    
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
    results_file = os.path.join(subj_dir, 'rsa_norm_results_11_07_24.pkl')
    with open(results_file, 'wb') as f:
        #json.dump(results_dict, f, indent=4)
        #save_modelresult(modelresult, fname)
        pickle.dump(results_dict, f)
        
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


def get_bundle_path():
    """
    Returns the appropriate bundle path based on the system user.
    Add new user paths as needed.
    """
    user_paths = {
        'ryanwebb': '/Users/ryanwebb/Documents/GitHub/Bundle_Value/',
        'locro': '/Users/locro/Documents/Bundle_Value/'
    }
    
    # Get current user
    current_user = Path.home().name
    
    # Return path for current user, with a helpful error if user not found
    if current_user in user_paths:
        return user_paths[current_user]
    else:
        raise ValueError(f"No path configured for user '{current_user}'. "
                        f"Please add your path to the user_paths dictionary.")

# Replace the hardcoded bundle_path with the dynamic one
bundle_path = get_bundle_path()

# all subjects
# subj_list = ['101', '102', '103', '104','105','106','107','108','109','110','111','112','113','114']
subj_list = ['104']


#subject with "not insane" value regions (i.e. positive code for absolute value)
#subj_list = ['101', '102', '103', '104','107','108','109','110','111','112','113','114']

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']

mask_loop = ['Frontal_Med_Orb', 'OFCmed', 'Frontal_Sup_Medial']
mask_names = ['vmPFC','OFCmed','dmPFC']

#mask_loop = ['Frontal_Med_Orb']
#mask_names = ['OFCmed']

square_dsm_bool = False
ranked = False
remove_within_day = True

for subj in subj_list:
    start_time = time.time()
    print(subj)

    fmri_dsms_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/fmri_dsm_list_np.npz'
    data = np.load(fmri_dsms_file)
    fmri_dsm_list = [data['arr_{}'.format(i)] for i in range(0, len(data))]
    
    target_dsms_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/target_dsms_np.npz'
    target_dsms = np.load(target_dsms_file, allow_pickle=True)
    target_dsms = {key: target_dsms[key] for key in target_dsms}
    
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
        
        info={'subj': subj,'mask': mask_name}
        
        info['model']='Divisive by Cat Average'
        
        # Set up parameters with initial guesses
        params = Parameters()
        params.add('a0', value=0)
        params.add('a1', value=1)
        params.add('a2', value=1)
        params.add('a3', value=1)
        params.add('b0', value=0, vary=False)
        params.add('b1', value=1)
        params.add('sigma', value=1, min=0) # sigma should be non-negative
        params.add('w1', value=0, vary=False)
        params.add('w_v', value=0, vary=False)  
        params.add('w_avg', value=1, vary=False) 
        
        if info['model'] not in results_dict: 
            results_dict[info['model']] = {}       
        results_dict[info['model']][info['mask']] = nonlinear_rsa(info, params, temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds)
        
        # info['model']=''Divisive by Cat Average (sigma and w)'
        
        # # Set up parameters with initial guesses
        # params = Parameters()
        # params.add('a0', value=0)
        # params.add('a1', value=1)
        # params.add('a2', value=1)
        # params.add('a3', value=1)
        # params.add('b', value=1, vary=False)
        # params.add('sigma', value=1, min=0) # sigma should be non-negative
        # params.add('w1', value=0, vary=False)
        # params.add('w_v', value=0, vary=False)  
        # params.add('w_avg', value=0) 
        
        # if info['model'] not in results_dict: 
        #     results_dict[info['model']] = {}        
        # results_dict[info['model']][info['mask']] = nonlinear_rsa(info, params, temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds)
        
        # info['model']='Divisive by Cat Average(w)'
        
        # # Set up parameters with initial guesses
        # params = Parameters()
        # params.add('a0', value=0)
        # params.add('a1', value=1)
        # params.add('a2', value=1)
        # params.add('a3', value=1)
        # params.add('b0', value=0)
        # params.add('b1', value=1)
        # params.add('sigma', value=1, vary=False) # sigma should be non-negative
        # params.add('w1', value=0, vary=False)
        # params.add('w_v', value=0, vary=False )  
        # params.add('w_avg', value=1) 
                
        # if info['model'] not in results_dict: 
        #     results_dict[info['model']] = {}        
        # results_dict[info['model']][info['mask']] = nonlinear_rsa(info, params, temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds)
        
        
        info['model']='Interaction Full(w)'
        
        # Set up parameters with initial guesses
        params = Parameters()
        params.add('a0', value=0)
        params.add('a1', value=1)
        params.add('a2', value=1)
        params.add('a3', value=1)
        params.add('b0', value=0)
        params.add('b1', value=1)
        params.add('sigma', value=0, vary=False ) # sigma should be non-negative
        params.add('w1', value=0, vary=False)
        params.add('w_v', value=0)  
        params.add('w_avg', value=0, vary=False) 
                
        if info['model'] not in results_dict: 
            results_dict[info['model']] = {}        
        results_dict[info['model']][info['mask']] = nonlinear_rsa(info, params, temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds)
        
        
        # info['model']='Divisive by Cat Interaction'
        
        # # Set up parameters with initial guesses
        # params = Parameters()
        # params.add('a0', value=0)
        # params.add('a1', value=1)
        # params.add('a2', value=1)
        # params.add('a3', value=1)
        # params.add('b0', value=0)
        # params.add('b1', value=1, vary=False)
        # params.add('sigma', value=1, min=0) # sigma should be non-negative
        # params.add('w1', value=0)
        # params.add('w_v', value=0, vary=False)  
        # params.add('w_avg', value=0, vary=False) 
        
        # if info['model'] not in results_dict: 
        #     results_dict[info['model']] = {}        
        # results_dict[info['model']][info['mask']] = nonlinear_rsa(info, params, temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds)

        # info['model']='Divisive by Cat Interaction (Diff Spec)'
        # # b_0 + b_1 * v / (1 + w1 I)
        # # Set up parameters with initial guesses
        # params = Parameters()
        # params.add('a0', value=0)
        # params.add('a1', value=1)
        # params.add('a2', value=1)
        # params.add('a3', value=1)
        # params.add('b0', value=0)
        # params.add('b1', value=1)
        # params.add('sigma', value=1, vary=False) # sigma should be non-negative
        # params.add('w1', value=1)
        # params.add('w_v', value=0, vary=False)  
        # params.add('w_avg', value=0, vary=False) 
        
        # if info['model'] not in results_dict: 
        #     results_dict[info['model']] = {}        
        # results_dict[info['model']][info['mask']] = nonlinear_rsa(info, params, temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds)
        
        # info['model']='Divisive by Interaction + V'
        
        # # Set up parameters with initial guesses
        # params = Parameters()
        # params.add('a0', value=0)
        # params.add('a1', value=1)
        # params.add('a2', value=1)
        # params.add('a3', value=1)
        # params.add('b0', value=0, vary=False)
        # params.add('b1', value=1, vary=False)
        # params.add('sigma', value=1, min=0) # sigma should be non-negative
        # params.add('w1', value=1, vary=False)
        # params.add('w_v', value=0, vary=False)  
        # params.add('w_avg', value=0, vary=False) 
        
        # if info['model'] not in results_dict: 
        #     results_dict[info['model']] = {}        
        # results_dict[info['model']][info['mask']] = nonlinear_rsa(info, params, temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds)
        
        info['model']='Absolute'
        
        # Set up parameters with initial guesses
        params = Parameters()
        params.add('a0', value=0)
        params.add('a1', value=1)
        params.add('a2', value=1)
        params.add('a3', value=1)
        params.add('b0', value=0, vary=False)
        params.add('b1', value=1)
        params.add('sigma', value=1, vary=False) # sigma should be non-negative
        params.add('w1', value=0, vary=False)
        params.add('w_v', value=0, vary=False )  
        params.add('w_avg', value=0, vary=False) 
        
        if info['model'] not in results_dict: 
            results_dict[info['model']] = {}        
        results_dict[info['model']][info['mask']] = nonlinear_rsa(info, params, temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds)
        
        # info['model']='Absolute + Bundle'
        
        # # Set up parameters with initial guesses
        # params = Parameters()
        # params.add('a0', value=0)
        # params.add('a1', value=1)
        # params.add('a2', value=1)
        # params.add('a3', value=1)
        # params.add('b0', value=0)
        # params.add('b1', value=1)
        # params.add('sigma', value=0, vary=False) # sigma should be non-negative
        # params.add('w1', value=0, vary=False)
        # params.add('w_v', value=0, vary=False)  
        # params.add('w_avg', value=1, vary=False) 
        
        
        # if info['model'] not in results_dict: 
        #     results_dict[info['model']] = {}        
        # results_dict[info['model']][info['mask']] = nonlinear_rsa(info, params, temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds)
        
        # info['model']='Null'
        
        # # Set up parameters with initial guesses
        # params = Parameters()
        # params.add('a0', value=0)
        # params.add('a1', value=1)
        # params.add('a2', value=1)
        # params.add('a3', value=1)
        # params.add('b0', value=0, vary=False)
        # params.add('b1', value=0, vary=False)
        # params.add('sigma', value=1, vary=False) # sigma should be non-negative
        # params.add('w1', value=0, vary=False)
        # params.add('w_v', value=0, vary=False )  
        # params.add('w_avg', value=0, vary=False) 
        
        # if info['model'] not in results_dict: 
        #     results_dict[info['model']] = {}        
        # results_dict[info['model']][info['mask']] = nonlinear_rsa(info, params, temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds)
        
        #info['model']='Items'
        
        # # Set up parameters with initial guesses
        # params = Parameters()
        # # Set up parameters with initial guesses
        # params = Parameters()
        # params.add('a0', value=0)
        # params.add('a1', value=1)
        # params.add('a2', value=1)
        # params.add('a3', value=1)
        # params.add('b0', value=1)
        # params.add('b1', value=1)
        # params.add('sigma', value=1, vary=False)  # sigma should be non-negative
        # params.add('w1', value=1, min=0)  # w should be non-negative
        # params.add('w_v', value=1, min=0)  # w should be non-negative
        # params.add('w_avg', value=0, vary=False) 
        
        # if info['model'] not in results_dict: 
        #     results_dict[info['model']] = {}        
        # results_dict[info['model']][info['mask']] = nonlinear_rsa_items(info, params, temp_fmri, partial_dsms, abs_value, btwn_day_inds, item1_value, item2_value)
        
        # info['model']='Items Interaction'
        
        # # Set up parameters with initial guesses
        # params = Parameters()
        # # Set up parameters with initial guesses
        # params = Parameters()
        # params.add('a0', value=0)
        # params.add('a1', value=1)
        # params.add('a2', value=1)
        # params.add('a3', value=1)
        # params.add('b0', value=1)
        # params.add('b1', value=1)
        # params.add('sigma', value=1, vary=False)  # sigma should be non-negative
        # params.add('w1', value=0, vary=False)  # w should be non-negative
        # params.add('w_v', value=1)  # w should be non-negative
        # params.add('w_avg', value=0, vary=False) 
        
        # if info['model'] not in results_dict: 
        #     results_dict[info['model']] = {}        
        # results_dict[info['model']][info['mask']] = nonlinear_rsa_items(info, params, temp_fmri, partial_dsms, abs_value, btwn_day_inds, item1_value, item2_value)
           
    # Save results and plot
    save_results(int(subj),results_dict, mask_names)
        
   
        #save_results(int(subj), name, mask_name, nonlinear_rsa(name, params, temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds))

        # Perform regressions and collect results
  #  results_dict[mask_name] = {
   #         name1: out1,
    #        'Interact by Cat Interaction':nonlinear_rsa_regression2_interact(temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds),
     #       'Divisive By Cat Interaction Denom':nonlinear_rsa_regression2_denom(temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds),
      #       name2: out2,
       #     'Divisive by Cat Average and V Denominator': nonlinear_rsa_regression2_plus_vt_denom(temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds),
        #    'Constituent Values': nonlinear_rsa_regression3_new(temp_fmri, partial_dsms, abs_value, btwn_day_inds, item1_value, item2_value),
         #   'Absolute': abs_value_rsa_regression(temp_fmri, partial_dsms, abs_value, btwn_day_inds, item1_value, item2_value)
       # }
    
        # Perform regressions and collect results
  # results_dict[mask_name] = {
   #        name1: out1,
        #    'Divisive By Cat Interaction Denom':nonlinear_rsa_regression2_denom(temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds),
      #  #   'Divisive by Cat Average and V Denominator': nonlinear_rsa_regression2_plus_vt_denom(temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds),
          #  'Constituent Values': nonlinear_rsa_regression3_new(temp_fmri, partial_dsms, abs_value, btwn_day_inds, item1_value, item2_value),
           # 'Absolute': abs_value_rsa_regression(temp_fmri, partial_dsms, abs_value, btwn_day_inds, item1_value, item2_value)
        #}

    # Save results and plot
        