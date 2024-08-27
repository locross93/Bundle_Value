#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:04:07 2022

@author: logancross
"""

divnorm_value = np.zeros([len(abs_value)])
beta = 1000
sigma = 0
divnorm_value[sitem_inds] = compute_divisive_norm(abs_value[sitem_inds], beta, sigma) 
divnorm_value[bundle_inds] = compute_divisive_norm(abs_value[bundle_inds], beta, sigma)

plt.scatter(abs_value[sitem_inds], divnorm_value[sitem_inds], c='r', label='Single Item')
plt.scatter(abs_value[bundle_inds], divnorm_value[bundle_inds], c='b', label='Bundle')
plt.legend(bbox_to_anchor=(1.0, 1))
plt.show()

def mean(X):
    """
    returns mean of vector X.
    """
    return(float(sum(X))/ len(X))

def svar(X, xbar = None):
    """
    returns the sample variance of vector X.
    xbar is sample mean of X.
    """ 
    if xbar is None: #fools had mean instead of xbar
       xbar = mean(X)
    S = sum([(x - xbar)**2 for x in X])
    return S / (len(X) - 1)

def corr(X,Y, xbar= None, xvar = None, ybar = None, yvar= None):
    """
    Computes correlation coefficient between X and Y.
    returns None on error.
    """
    n = len(X)
    if n != len(Y):
       return 'size mismatch X/Y:',len(X),len(Y)
    if xbar is None: xbar = mean(X)
    if ybar is None: ybar = mean(Y)
    if xvar is None: xvar = svar(X)
    if yvar is None: yvar = svar(Y)
 
    S = sum([(X[i] - xbar)* (Y[i] - ybar) for i in range(len(X))])
    return S/((n-1)* np.sqrt(xvar* yvar))

def pcf3(X,Y,Z):
    """
    Returns a dict of the partial correlation coefficients
    r_XY|z , r_XZ|y, r_YZ|x 
    """
    
    xbar = mean(X)
    ybar = mean(Y)
    zbar = mean(Z)
    xvar = svar(X)
    yvar = svar(Y)
    zvar = svar(Z)
    # computes pairwise simple correlations.
    rxy  = corr(X,Y, xbar=xbar, xvar= xvar, ybar = ybar, yvar = yvar)
    rxz  = corr(X,Z, xbar=xbar, xvar= xvar, ybar = zbar, yvar = zvar)
    ryz  = corr(Y,Z, xbar=ybar, xvar= yvar, ybar = zbar, yvar = zvar)
    rxy_z = (rxy - (rxz*ryz)) / np.sqrt((1 -rxz**2)*(1-ryz**2))
    rxz_y = (rxz - (rxy*ryz)) / np.sqrt((1-rxy**2) *(1-ryz**2))
    ryz_x = (ryz - (rxy*rxz)) / np.sqrt((1-rxy**2) *(1-rxz**2))
    
    return rxy_z


model_dsms = [target_dsms[model_dsm][btwn_day_inds] for model_dsm in model_dsm_names]
partial_dsm = np.column_stack((model_dsms))
target_dsm = res_value[btwn_day_inds]
rp = pcf3(temp_fmri, target_dsm, target_dsms['item_or_bundle'][btwn_day_inds])
res_partial = np.array([rp['rxy_z']])