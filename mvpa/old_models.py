#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:02:13 2024

@author: ryanwebb
"""

# def nonlinear_rsa_regression2(temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds):

#     # Define the nonlinear model function
#     def nonlinear_model_sigma(x, a0, a1, a2, a3, a4, sigma, abs_value, partial_dsms):
#         norm_values = np.zeros(len(abs_value))
        
#         # Divisively normalize values
#         for trial_inds in trial_type_inds:
#             avg_value = np.mean(abs_value[trial_inds])
#             norm_values[trial_inds] = abs_value[trial_inds] / (sigma + avg_value)
        
#         ds_value = norm_values.reshape(-1, 1)
#         value_dsm = pdist(ds_value, metric='euclidean')
        
#         if ranked:
#             value_dsm = rankdata(value_dsm)
            
#         if remove_within_day and btwn_day_inds is not None:
#             value_dsm = value_dsm[btwn_day_inds]
        
#         # transpose partial_dsms
#         partial_dsms = partial_dsms.T
#         # concatenate partial_dsms with value_dsm by column
#         x = np.column_stack((partial_dsms, value_dsm))
        
#         return a0 + a1 * x[:, 0] + a2 * x[:, 1] + a3 * x[:, 2] + a4 * x[:, 3] 
    
#     # Create the LMfit model
#     lmfit_model = Model(nonlinear_model_sigma, independent_vars=['x', 'abs_value', 'partial_dsms'])
    
#     # Set up parameters with initial guesses
#     params = Parameters()
#     params.add('a0', value=0)
#     params.add('a1', value=1)
#     params.add('a2', value=1)
#     params.add('a3', value=1)
#     params.add('a4', value=1)
#     params.add('sigma', value=0, min=0)  # sigma should be non-negative
    
#     # Fit the model
#     result = lmfit_model.fit(temp_fmri, params, x=np.zeros(len(temp_fmri)), 
#                              abs_value=abs_value, partial_dsms=partial_dsms)
       
#     # Calculate statistics
#     bic = result.bic
    
#     # Calculate adjusted R-squared
#     nobs = len(temp_fmri)
#     num_params = len([p for p in result.params.values() if not p.vary])
#     df_modelwc = len(result.params) - num_params # degrees of freedom
#     ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
#     ss_residual = np.sum(result.residual**2)
#     r_squared = 1 - (ss_residual / ss_total)
#     adj_r2 = 1 - (1 - r_squared) * ((nobs - 1) / (nobs - df_modelwc - 1))

#     fit_dict = {'r2': r_squared,'bic': bic, 'adj_r2': adj_r2, 
#                     'a0': result.params['a0'].value, 'a1': result.params['a1'].value, 
#                     'a2': result.params['a2'].value, 'a3': result.params['a3'].value,
#                     'a4': result.params['a4'].value, 'sigma': result.params['sigma'].value}
#     print('Divisive by VCat Average\n')
#     print(result.fit_report())
#     print('\n')
#     return fit_dict





# #AverageByType and V with Parameters in Denom
# def nonlinear_rsa_regression2_plus_vt_denom(temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds):

#     # Define the nonlinear model function
#     def nonlinear_model_sigma(x, a0, a1, a2, a3, b1, w2, abs_value, partial_dsms):
#         norm_values = np.zeros(len(abs_value))

#         # Divisively normalize values
#         for trial_inds in trial_type_inds:
#             avg_value = np.mean(abs_value[trial_inds])
#             norm_values[trial_inds] = abs_value[trial_inds] / (avg_value + w2*abs_value[trial_inds])
        
#         ds_value = norm_values.reshape(-1, 1)
#         value_dsm = pdist(ds_value, metric='euclidean')
        
#         if ranked:
#             value_dsm = rankdata(value_dsm)
            
#         if remove_within_day and btwn_day_inds is not None:
#             value_dsm = value_dsm[btwn_day_inds]
        
#         # transpose partial_dsms
#         partial_dsms = partial_dsms.T
#         # concatenate partial_dsms with value_dsm by column
#         x = np.column_stack((partial_dsms, value_dsm))
        
#         return a0 + a1 * x[:, 0] + a2 * x[:, 1] + a3 * x[:, 2] + b1 * x[:, 3] 
    
#     # Create the LMfit model
#     lmfit_model = Model(nonlinear_model_sigma, independent_vars=['x', 'abs_value', 'partial_dsms'])
    
#     # Set up parameters with initial guesses
#     params = Parameters()
#     params.add('a0', value=0)
#     params.add('a1', value=1)
#     params.add('a2', value=1)
#     params.add('a3', value=1)
#     params.add('b1', value=1, min=0)
#     params.add('w2', value=1, min=0)
#    # params.add('sigma', value=0, min=0)  # sigma should be non-negative
    
#     # Fit the model
#     result = lmfit_model.fit(temp_fmri, params, x=np.zeros(len(temp_fmri)), 
#                              abs_value=abs_value, partial_dsms=partial_dsms) # method='nelder'

#     # Calculate statistics
#     bic = result.bic
    
#     # Calculate adjusted R-squared
#     nobs = len(temp_fmri)
#     num_params = len([p for p in result.params.values() if not p.vary])
#     df_modelwc = len(result.params) - num_params # degrees of freedom
#     ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
#     ss_residual = np.sum(result.residual**2)
#     r_squared = 1 - (ss_residual / ss_total)
#     adj_r2 = 1 - (1 - r_squared) * ((nobs - 1) / (nobs - df_modelwc - 1))

#     fit_dict = {'r2': r_squared, 'bic': bic, 'adj_r2': adj_r2, 
#                     'a0': result.params['a0'].value, 'a1': result.params['a1'].value, 
#                     'a2': result.params['a2'].value, 'a3': result.params['a3'].value,
#                     'b1': result.params['b1'].value, 'w2': result.params['w2'].value, 
#                     #'sigma': result.params['sigma'].value
#                     }
#     print('Divisive by Cat Average and V Denominator\n')
#     print(result.fit_report())
#     print('\n')   
#     return fit_dict


# def nonlinear_rsa_regression3(temp_fmri, partial_dsms, abs_value, btwn_day_inds, item1_value, item2_value):

#     # Define the nonlinear model function
#     def nonlinear_model_sigma(x, a0, a1, a2, a3, sigma, w, item1_value, item2_value, partial_dsms):

#         # Divisively normalize values
#         #norm_values = abs_value / (sigma + abs_value*w)
#         norm_values = (item1_value + item2_value) / (sigma + w*(item1_value + item2_value))
        
#         ds_value = norm_values.reshape(-1, 1)
#         value_dsm = pdist(ds_value, metric='euclidean')
        
#         if ranked:
#             value_dsm = rankdata(value_dsm)
            
#         if remove_within_day and btwn_day_inds is not None:
#             value_dsm = value_dsm[btwn_day_inds]
        
#         # transpose partial_dsms
#         partial_dsms = partial_dsms.T
#         # concatenate partial_dsms with value_dsm by column
#         x = np.column_stack((partial_dsms, value_dsm))
        
#         return a0 + a1 * x[:, 0] + a2 * x[:, 1] + a3 * x[:, 2] + x[:, 3]  
    
#     # Create the LMfit model
#     lmfit_model = Model(nonlinear_model_sigma, independent_vars=['x', 'item1_value', 'item2_value', 'partial_dsms'])
    
#     # Set up parameters with initial guesses
#     params = Parameters()
#     params.add('a0', value=0)
#     params.add('a1', value=1)
#     params.add('a2', value=1)
#     params.add('a3', value=1)
#     params.add('sigma', value=5, min=0)  # sigma should be non-negative
#     params.add('w', value=1, min=0)  # w should be non-negative
    
#     # Fit the model
#     result = lmfit_model.fit(temp_fmri, params, x=np.zeros(len(temp_fmri)), 
#                              item1_value=item1_value, item2_value=item2_value, partial_dsms=partial_dsms)
    
#     # Calculate statistics
#     bic = result.bic
    
#     # Calculate adjusted R-squared
#     nobs = len(temp_fmri)
#     num_params = len([p for p in result.params.values() if not p.vary])
#     df_modelwc = len(result.params) - num_params # degrees of freedom
#     ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
#     ss_residual = np.sum(result.residual**2)
#     r_squared = 1 - (ss_residual / ss_total)
#     adj_r2 = 1 - (1 - r_squared) * ((nobs - 1) / (nobs - df_modelwc - 1))

#     fit_dict = {'bic': bic, 'adj_r2': adj_r2,
#                     'a0': result.params['a0'].value, 'a1': result.params['a1'].value, 
#                     'a2': result.params['a2'].value, 'a3': result.params['a3'].value,
#                     'sigma': result.params['sigma'].value, 'w': result.params['w'].value}
    
#     return fit_dict

# def nonlinear_rsa_regression2_denom(temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds):
#     name = 'Divisive by Cat Denom'
#     # Define the nonlinear model function
#     def nonlinear_model_sigma(x, a0, a1, a2, a3, sigma, w1, abs_value, partial_dsms):
#         norm_values = np.zeros(len(abs_value))
        
#         # Divisively normalize values
#         for trial_inds in trial_type_inds:
#             avg_value = np.mean(abs_value[trial_inds])
#             norm_values[trial_inds] = abs_value[trial_inds] / (sigma + w1 * trial_categ[trial_inds])
#             #norm_values[trial_inds] = a4*abs_value[trial_inds] + a5*abs_value[trial_inds]*trial_categ[trial_inds]
            
#         ds_value = norm_values.reshape(-1, 1)
#         value_dsm = pdist(ds_value, metric='euclidean')
        
#         if ranked:
#             value_dsm = rankdata(value_dsm)
            
#         if remove_within_day and btwn_day_inds is not None:
#             value_dsm = value_dsm[btwn_day_inds]
        
#         # transpose partial_dsms
#         partial_dsms = partial_dsms.T
#         # concatenate partial_dsms with value_dsm by column
#         x = np.column_stack((partial_dsms, value_dsm))

#         return a0 + a1 * x[:, 0] + a2 * x[:, 1] + a3 * x[:, 2] + x[:, 3] 
    
#     # Create the LMfit model
#     lmfit_model = Model(nonlinear_model_sigma, independent_vars=['x', 'abs_value', 'partial_dsms'])
    
#     # Set up parameters with initial guesses
#     params = Parameters()
#     params.add('a0', value=0)
#     params.add('a1', value=1)
#     params.add('a2', value=1)
#     params.add('a3', value=1)
#     params.add('sigma', value=1, min=0)
#     params.add('w1', value=0)
#     #params.add('sigma', value=0, min=0)  # sigma should be non-negative
#     breakpoint()
#     # Fit the model
#     result = lmfit_model.fit(temp_fmri, params, x=np.zeros(len(temp_fmri)), 
#                              abs_value=abs_value, partial_dsms=partial_dsms)
       
#     # Calculate statistics
#     bic = result.bic
    
#     # Calculate adjusted R-squared
#     nobs = len(temp_fmri)
#     num_params = len([p for p in result.params.values() if not p.vary])
#     df_modelwc = len(result.params) - num_params # degrees of freedom
#     ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
#     ss_residual = np.sum(result.residual**2)
#     r_squared = 1 - (ss_residual / ss_total)
#     adj_r2 = 1 - (1 - r_squared) * ((nobs - 1) / (nobs - df_modelwc - 1))

#     fit_dict = {'r2': r_squared,'bic': bic, 'adj_r2': adj_r2, 
#                     'a0': result.params['a0'].value, 'a1': result.params['a1'].value, 
#                     'a2': result.params['a2'].value, 'a3': result.params['a3'].value,
#                     #'w1': result.params['w1'].value, 'sigma': result.params['sigma'].value
#                     'sigma': result.params['sigma'].value, 'w1': result.params['w1'].value
#                     }
#     print(name,'\n')
#     print(result.fit_report())
#     print('\n')
    
#     return fit_dict


# def abs_value_rsa_regression(temp_fmri, partial_dsms, abs_value, btwn_day_inds, item1_value, item2_value):

#     # Define the nonlinear model function
#     def nonlinear_model_sigma(x, a0, a1, a2, a3, sigma, abs_value, partial_dsms):

#         # Divisively normalize values
#         norm_values = abs_value / sigma 
        
#         ds_value = norm_values.reshape(-1, 1)
#         value_dsm = pdist(ds_value, metric='euclidean')
        
#         if ranked:
#             value_dsm = rankdata(value_dsm)
            
#         if remove_within_day and btwn_day_inds is not None:
#             value_dsm = value_dsm[btwn_day_inds]
        
#         # transpose partial_dsms
#         partial_dsms = partial_dsms.T
#         # concatenate partial_dsms with value_dsm by column
#         x = np.column_stack((partial_dsms, value_dsm))
        
#         return a0 + a1 * x[:, 0] + a2 * x[:, 1] + a3 * x[:, 2] + x[:, 3] 
    
#     # Create the LMfit model
#     lmfit_model = Model(nonlinear_model_sigma, independent_vars=['x', 'abs_value', 'partial_dsms'])
    
#     # Set up parameters with initial guesses
#     params = Parameters()
#     params.add('a0', value=0)
#     params.add('a1', value=1)
#     params.add('a2', value=1)
#     params.add('a3', value=1)
#     params.add('sigma', value=5, min=0)  # sigma should be non-negative
    
#     # Fit the model
#     result = lmfit_model.fit(temp_fmri, params, x=np.zeros(len(temp_fmri)), 
#                              abs_value=abs_value, partial_dsms=partial_dsms)
    
#     # Calculate statistics
#     bic = result.bic
    
#     # Calculate adjusted R-squared
#     nobs = len(temp_fmri)
#     num_params = len([p for p in result.params.values() if not p.vary])
#     df_modelwc = len(result.params) - num_params # degrees of freedom
#     ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
#     ss_residual = np.sum(result.residual**2)
#     r_squared = 1 - (ss_residual / ss_total)
#     adj_r2 = 1 - (1 - r_squared) * ((nobs - 1) / (nobs - df_modelwc - 1))
    
#     fit_dict = {'bic': bic, 'adj_r2': adj_r2,
#                     'a0': result.params['a0'].value, 'a1': result.params['a1'].value, 
#                     'a2': result.params['a2'].value, 'a3': result.params['a3'].value,
#                     'sigma': result.params['sigma'].value}
    
#     return fit_dict


#Interact Value and Category
# def nonlinear_rsa_regression2_interact(temp_fmri, partial_dsms, abs_value, trial_type_inds, btwn_day_inds):

#     # Define the nonlinear model function
#     def nonlinear_model_sigma(x, a0, a1, a2, a3, a4, w1, abs_value, partial_dsms):
#         norm_values = np.zeros(len(abs_value))
        
#         # Interact values
#         for trial_inds in trial_type_inds:
#             avg_value = np.mean(abs_value[trial_inds])
#             norm_values[trial_inds] = a4*abs_value[trial_inds] + w1*abs_value[trial_inds]*trial_categ[trial_inds]
            
#         ds_value = norm_values.reshape(-1, 1)
#         value_dsm = pdist(ds_value, metric='euclidean')
        
#         if ranked:
#             value_dsm = rankdata(value_dsm)
            
#         if remove_within_day and btwn_day_inds is not None:
#             value_dsm = value_dsm[btwn_day_inds]
        
#         # transpose partial_dsms
#         partial_dsms = partial_dsms.T
#         # concatenate partial_dsms with value_dsm by column
#         x = np.column_stack((partial_dsms, value_dsm))

#         return a0 + a1 * x[:, 0] + a2 * x[:, 1] + a3 * x[:, 2] + x[:, 3] 
    
#     # Create the LMfit model
#     lmfit_model = Model(nonlinear_model_sigma, independent_vars=['x', 'abs_value', 'partial_dsms'])
    
#     # Set up parameters with initial guesses
#     params = Parameters()
#     params.add('a0', value=0)
#     params.add('a1', value=1)
#     params.add('a2', value=1)
#     params.add('a3', value=1)
#     params.add('a4', value=1)
#     params.add('w1', value=1)
    
#     # Fit the model
#     result = lmfit_model.fit(temp_fmri, params, x=np.zeros(len(temp_fmri)), 
#                              abs_value=abs_value, partial_dsms=partial_dsms)
       
#     # Calculate statistics
#     bic = result.bic
    
#     # Calculate adjusted R-squared
#     nobs = len(temp_fmri)
#     num_params = len([p for p in result.params.values() if not p.vary])
#     df_modelwc = len(result.params) - num_params # degrees of freedom
#     ss_total = np.sum((temp_fmri - np.mean(temp_fmri))**2)
#     ss_residual = np.sum(result.residual**2)
#     r_squared = 1 - (ss_residual / ss_total)
#     adj_r2 = 1 - (1 - r_squared) * ((nobs - 1) / (nobs - df_modelwc - 1))
    
#     fit_dict = result.params.create_uvars(result.covar)
#     fit_dict['r2'] = r_squared
#     fit_dict['bic'] = result.bic
#     fit_dict['adj_r2'] = adj_r2
    
#     print('\n')
#     print(info['subj'])
#     print(info['mask'])
#     print(info['model'])
#     print(result.fit_report()) 
#     print('\n')

#     subj_dir = os.path.join(bundle_path, 'mvpa', 'analyses', 'sub'+str(info['subj']))
    
#     # Save parameter fits and statistics
#     with open(os.path.join(subj_dir, "".join(['rsa_norm_results_11_07_24-', info['model'], info['mask'],'.json'])), 'w') as f:
#         result.params.dump(f)
        
#     # Save model fit and statistics
#     #save_modelresult(result, os.path.join(subj_dir, "".join(['rsa_norm_results_11_07_24-', info['model'], info['mask'],'.sav'])))
#     return fit_dict