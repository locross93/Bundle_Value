import numpy as np
import pandas as pd
from pathlib import Path
import os
from lmfit import Model, Parameters
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def get_bundle_path():
    """Returns the appropriate bundle path based on the system user."""
    user_paths = {
        'ryanwebb': '/Users/ryanwebb/Documents/GitHub/Bundle_Value/',
        'locro': '/Users/locro/Documents/Bundle_Value/'
    }
    current_user = Path.home().name
    if current_user in user_paths:
        return user_paths[current_user]
    else:
        raise ValueError(f"No path configured for user '{current_user}'")

def load_subject_data(subject, bundle_path):
    """Load DSMs and info for a single subject."""
    # Load fMRI DSMs (already flattened)
    fmri_dsms_file = os.path.join(bundle_path, f'mvpa/presaved_data/sub{subject}/fmri_dsm_list_np.npz')
    data = np.load(fmri_dsms_file)
    fmri_dsm_list = [data[f'arr_{i}'] for i in range(len(data))]
    
    # Load target DSMs (already flattened)
    target_dsms_file = os.path.join(bundle_path, f'mvpa/presaved_data/sub{subject}/target_dsms_np.npz')
    target_dsms = np.load(target_dsms_file, allow_pickle=True)
    target_dsms = {key: target_dsms[key] for key in target_dsms}
    
    # Load subject info
    subj_info_file = os.path.join(bundle_path, f'mvpa/presaved_data/sub{subject}/info_all_trials.csv')
    subj_info_dict = pd.read_csv(subj_info_file)
    abs_value = subj_info_dict['Stimulus Value'].values
    trial_categ = subj_info_dict['Trial Categ'].values
    item1_value = subj_info_dict['Item 1 Value'].values
    item2_value = subj_info_dict['Item 2 Value'].values
    sitem_inds = np.where(trial_categ == 0)[0]
    bundle_inds = np.where(trial_categ == 1)[0]

    res_day = target_dsms['day']
    btwn_day_inds = np.where(res_day == 1)[0]
    
    return fmri_dsm_list, target_dsms, abs_value, sitem_inds, bundle_inds, btwn_day_inds

def create_subject_nuisance_dsm(n_samples_per_subject):
    """
    Create flattened subject nuisance DSM using vectorized operations.
    
    Parameters:
    -----------
    n_samples_per_subject : list
        List containing number of samples for each subject
        
    Returns:
    --------
    subject_dsm : ndarray
        Flattened DSM where 0 indicates within-subject comparisons and 
        1 indicates between-subject comparisons
    """
    # Create subject labels vector
    subject_labels = np.repeat(np.arange(len(n_samples_per_subject)), n_samples_per_subject)

    X = np.array([subject_labels, subject_labels]).T
    subject_dsm = pdist(X, 'matching')
    
    return subject_dsm

def concatenate_group_data(subjects, bundle_path, mask_index, model_dsm_names):
    """Concatenate flattened DSMs across subjects."""
    all_fmri_dsms = []
    all_model_dsms = {name: [] for name in model_dsm_names}
    all_abs_value = []
    all_sitem_inds = []
    all_bundle_inds = []
    all_btwn_day_inds = []
    n_samples_per_subject = []

    cumulative_trials = 0  # Keep track of total trials processed
    
    for subject in subjects:
        # Load subject data
        fmri_dsm_list, target_dsms, abs_value, sitem_inds, bundle_inds, btwn_day_inds = load_subject_data(subject, bundle_path)
        
        # Get number of trials for this subject
        n_trials = len(abs_value)
        
        # Get indices and offset them by cumulative trials
        sitem_inds = sitem_inds + cumulative_trials
        bundle_inds = bundle_inds + cumulative_trials
        btwn_day_inds = btwn_day_inds + cumulative_trials
        n_samples_per_subject.append([subject for n in range(len(btwn_day_inds))])
        
        # Store the offset indices
        all_sitem_inds.append(sitem_inds)
        all_bundle_inds.append(bundle_inds)
        all_btwn_day_inds.append(btwn_day_inds)
        
        # Update cumulative trials
        cumulative_trials += n_trials
        
        # Store DSMs
        all_fmri_dsms.append(fmri_dsm_list[mask_index])
        for name in model_dsm_names:
            all_model_dsms[name].append(target_dsms[name])
        all_abs_value.append(abs_value)
    
    # Concatenate all data
    group_fmri_dsm = np.concatenate(all_fmri_dsms)
    group_model_dsms = {name: np.concatenate(values) for name, values in all_model_dsms.items()}
    group_abs_value = np.concatenate(all_abs_value)
    group_sitem_inds = np.concatenate(all_sitem_inds)
    group_bundle_inds = np.concatenate(all_bundle_inds)
    trial_type_inds = [group_sitem_inds, group_bundle_inds]
    group_btwn_day_inds = np.concatenate(all_btwn_day_inds)
    group_partial_dsms = [group_model_dsms[name][group_btwn_day_inds] for name in model_dsm_names]
    group_fmri_dsm = group_fmri_dsm[group_btwn_day_inds]
    
    # Create subject nuisance regressors
    #subject_nuisance_dsms = create_subject_nuisance_dsm(n_samples_per_subject)
    #group_partial_dsms.append(subject_nuisance_dsms)
    subject_nuisance_dsms = np.concatenate(n_samples_per_subject)
    breakpoint()
    
    # Verify all DSMs have the same length
    dsm_length = len(group_fmri_dsm)
    assert len(group_partial_dsms[0]) == dsm_length, "Model DSMs have different lengths"
    #assert all(len(dsm) == dsm_length for dsm in subject_nuisance_dsms), "Nuisance DSMs have different lengths"
    
    return group_fmri_dsm, group_partial_dsms, group_abs_value, trial_type_inds, group_btwn_day_inds, subject_nuisance_dsms

def nonlinear_rsa_items_group(info, params, temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds):
    """Group-level nonlinear RSA with subject nuisance regressors."""
    
    def nonlinear_model_sigma(a0, a1, a2, a3, a4, b0, b1, sigma, w1, w_v, w_avg, abs_value, partial_dsms):
        norm_values = np.zeros(len(abs_value))
        # trial_categ is same length as norm_values with 0s for items and 1s for bundles
        trial_categ = np.zeros(len(abs_value))
        trial_categ[trial_type_inds[1]] = 1
        
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
        
        # Create value DSM (already flattened)
        ds_value = norm_values.reshape(-1, 1)
        value_dsm = pdist(ds_value, metric='euclidean')

        if btwn_day_inds is not None:
            value_dsm = value_dsm[btwn_day_inds]
        
        # Combine all regressors
        partial_dsms = partial_dsms.T
        x = np.column_stack((partial_dsms, value_dsm))
        #X = np.column_stack((partial_dsms, value_dsm, subject_nuisance_dsms))

        return a0 + a1 * x[:, 0] + a2 * x[:, 1] + a3 * x[:, 2] + a4 * x[:, 3] + x[:, 4] 
    
    # Create the LMfit model
    lmfit_model = Model(nonlinear_model_sigma, independent_vars=['abs_value', 'partial_dsms'])

    # Fit the model
    result = lmfit_model.fit(temp_fmri, params, 
                             abs_value=abs_value, partial_dsms=partial_dsms)
    
    # Calculate statistics
    ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
    ss_residual = np.sum(result.residual**2)
    r_squared = 1 - (ss_residual / ss_total)
    adj_r2 = 1 - (1 - r_squared) * ((result.ndata - 1) / (result.nfree - 1))
    
    fit_dict = result.params.create_uvars(result.covar)
    fit_dict['r2'] = r_squared
    fit_dict['bic'] = result.bic
    fit_dict['adj_r2'] = adj_r2
    
    print('\n')
    print(f"Group Results:")
    print(f"Mask: {info['mask']}")
    print(f"Model: {info['model']}")
    print(result.fit_report())
    print('\n')
    
    return fit_dict, result


def plot_multi_mask_normalization_comparison(results_dict):
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
    ax.set_title('RSA Normalized Codes - Group Level', fontsize=16)
    ax.set_xlabel('ROI', fontsize=12)
    ax.set_ylabel('RSA Adj R2', fontsize=12)
    
    # Adjust legend
    plt.legend(title='Normalization', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def save_results(bundle_path, results_dict):
    save_dir = os.path.join(bundle_path, 'mvpa', 'analyses', 'group')
    
    # Save parameter fits and statistics
    results_file = os.path.join(save_dir, 'rsa_norm_results_1_23_25.pkl')
    with open(results_file, 'wb') as f:
        #json.dump(results_dict, f, indent=4)
        #save_modelresult(modelresult, fname)
        pickle.dump(results_dict, f)
        
    # Save plot
    plot_file = os.path.join(save_dir, 'rsa_norm_plot_new.png')
    fig, ax = plot_multi_mask_normalization_comparison(results_dict)
    
    if ax.has_data():
        fig.savefig(plot_file, bbox_inches='tight')
        plt.close(fig)
        print("Plot saved successfully: {}".format(plot_file))
    else:
        print("Error: Plot is empty. No data found.")

    # Print some debugging information
    print("Sample of results:")
    print(pd.DataFrame(results_dict).head())

if __name__ == "__main__":
    # Setup
    bundle_path = get_bundle_path()
    subjects = [104, 107, 108]  # Add all subject numbers
    mask_names = ['vmPFC', 'OFCmed', 'dmPFC']
    
    # Specify which model DSMs to use (same as in individual subject analysis)
    model_dsm_names = ['choice', 'item_or_bundle', 'lr_choice']
    results_dict = {}
    
    for mask_idx, mask_name in enumerate(mask_names):
        print(f"Processing {mask_name}...")
        
        # Load and concatenate data - now including model_dsm_names
        group_fmri_dsm, group_model_dsms, group_abs_value, trial_type_inds, btwn_day_inds, subject_nuisance_dsms = concatenate_group_data(
            subjects, 
            bundle_path, 
            mask_idx,
            model_dsm_names  # Added this argument
        )
        
        # Set up parameters with initial guesses for Divisive by Cat Average
        info = {'mask': mask_name, 'model': 'Divisive by Cat Average'}
        params = Parameters()
        params.add('a0', value=0)
        params.add('a1', value=1)
        params.add('a2', value=1)
        params.add('a3', value=1)
        params.add('a4', value=1)
        params.add('b0', value=0, vary=False)
        params.add('b1', value=1)
        params.add('sigma', value=1, min=0) # sigma should be non-negative
        params.add('w1', value=0, vary=False)
        params.add('w_v', value=0, vary=False)  
        params.add('w_avg', value=1, vary=False)

        if info['model'] not in results_dict: 
            results_dict[info['model']] = {}        
        results_dict[info['model']][info['mask']], model_fit = nonlinear_rsa_items_group(info, params, group_fmri_dsm, group_model_dsms, group_abs_value, trial_type_inds, btwn_day_inds)

        # Set up parameters for Interaction Full(w)
        info = {'mask': mask_name, 'model': 'Interaction Full(w)'}
        params = Parameters()
        params.add('a0', value=0)
        params.add('a1', value=1)
        params.add('a2', value=1)
        params.add('a3', value=1)
        params.add('a4', value=1)
        params.add('b0', value=0)
        params.add('b1', value=1)
        params.add('sigma', value=0, vary=False ) # sigma should be non-negative
        params.add('w1', value=0, vary=False)
        params.add('w_v', value=0)  
        params.add('w_avg', value=0, vary=False)
        
        if info['model'] not in results_dict: 
            results_dict[info['model']] = {}        
        results_dict[info['model']][info['mask']], model_fit = nonlinear_rsa_items_group(info, params, group_fmri_dsm, group_model_dsms, group_abs_value, trial_type_inds, btwn_day_inds)
        
        # Set up parameters for absolute model
        info = {'mask': mask_name, 'model': 'Absolute'}
        params.add('a0', value=0)
        params.add('a1', value=1)
        params.add('a2', value=1)
        params.add('a3', value=1)
        params.add('a4', value=1)
        params.add('b0', value=0, vary=False)
        params.add('b1', value=1)
        params.add('sigma', value=1, vary=False) # sigma should be non-negative
        params.add('w1', value=0, vary=False)
        params.add('w_v', value=0, vary=False )  
        params.add('w_avg', value=0, vary=False) 

        if info['model'] not in results_dict: 
            results_dict[info['model']] = {}        
        results_dict[info['model']][info['mask']], model_fit = nonlinear_rsa_items_group(info, params, group_fmri_dsm, group_model_dsms, group_abs_value, trial_type_inds, btwn_day_inds)
        
        # # Add subject nuisance parameters
        # for i in range(len(subjects)-1):  # n-1 subject parameters to avoid collinearity
        #     params.add(f's{i}', value=0)
        
        # # Fit group RSA
        # results, model_fit = nonlinear_rsa_items_group(
        #     info, params, group_fmri_dsm, group_model_dsms,
        #     group_abs_value, trial_type_inds, btwn_day_inds
        # )
        
        # Save results
        # save_dir = os.path.join(bundle_path, 'mvpa', 'analyses', 'group')
        # os.makedirs(save_dir, exist_ok=True)
        
        # with open(os.path.join(save_dir, f"group_rsa_results_{mask_name}.json"), 'w') as f:
        #     model_fit.params.dump(f)

    save_results(bundle_path, results_dict)