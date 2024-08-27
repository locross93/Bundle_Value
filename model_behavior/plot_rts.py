#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:01:50 2021

@author: logancross
"""

import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import matplotlib.pyplot as plt

single_item_rts = pd.read_csv('/Users/logancross/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/rt_data_wtp_single_item.csv')

sns.barplot(x="Subject", y="RT", data=single_item_rts, ci=68)
plt.title('Average RTs - WTP Single Item Trials')

bundle_rts = pd.read_csv('/Users/logancross/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/rt_data_wtp_bundle.csv')

sns.barplot(x="Subject", y="RT", data=bundle_rts, ci=68)
plt.title('Average RTs - WTP Bundle Trials')