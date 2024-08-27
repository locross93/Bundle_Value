# -*- coding: utf-8 -*-
"""
Created on Thu May  2 19:13:40 2024

@author: locro
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate data for wide range and narrow range trials
n_trials = 100
wide_range_values = np.random.uniform(0, 10, n_trials)
narrow_range_values = np.random.uniform(4, 6, n_trials)

# Apply divisive normalization
wide_range_div_norm = wide_range_values / np.mean(wide_range_values)
narrow_range_div_norm = narrow_range_values / np.mean(narrow_range_values)

# Apply range normalization
wide_range_min, wide_range_max = np.min(wide_range_values), np.max(wide_range_values)
narrow_range_min, narrow_range_max = np.min(narrow_range_values), np.max(narrow_range_values)

wide_range_range_norm = (wide_range_values - wide_range_min) / (wide_range_max - wide_range_min)
narrow_range_range_norm = (narrow_range_values - narrow_range_min) / (narrow_range_max - narrow_range_min)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

sns.scatterplot(x=wide_range_values, y=wide_range_div_norm, ax=axes[0, 0])
axes[0, 0].set_title("Wide Range - Divisive Normalization")
axes[0, 0].set_xlabel("Absolute Value")
axes[0, 0].set_ylabel("Normalized Value")

sns.scatterplot(x=wide_range_values, y=wide_range_range_norm, ax=axes[0, 1])
axes[0, 1].set_title("Wide Range - Range Normalization")
axes[0, 1].set_xlabel("Absolute Value")
axes[0, 1].set_ylabel("Normalized Value")

sns.scatterplot(x=narrow_range_values, y=narrow_range_div_norm, ax=axes[1, 0])
axes[1, 0].set_title("Narrow Range - Divisive Normalization")
axes[1, 0].set_xlabel("Absolute Value")
axes[1, 0].set_ylabel("Normalized Value")

sns.scatterplot(x=narrow_range_values, y=narrow_range_range_norm, ax=axes[1, 1])
axes[1, 1].set_title("Narrow Range - Range Normalization")
axes[1, 1].set_xlabel("Absolute Value")
axes[1, 1].set_ylabel("Normalized Value")

plt.tight_layout()
plt.show()