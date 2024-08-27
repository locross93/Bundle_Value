# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:38:15 2024

@author: locro
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

best_behavior_model = list(best_model)
best_neural_model = ['Div Norm', 'MC Norm', 'Div Norm', 'RW Norm', 'MC Norm', 'MC Norm', 'RW Norm', 'RW Norm', 'Div Norm', 'RW Norm', 'Div Norm', 'RW Norm', 'RW Norm', 'RW Norm'] 

# Assuming best_neural_model and best_behavior_model are already defined
data = pd.DataFrame({'Best Neural Model': best_neural_model,
                     'Best Behavioral Model': best_behavior_model})

# Define the order of the models
model_order = ['Abs Value', 'Div Norm', 'MC Norm', 'RW Norm']

# Create a contingency table with all models
contingency_table = pd.crosstab(data['Best Neural Model'], data['Best Behavioral Model'])
contingency_table = contingency_table.reindex(index=model_order, columns=model_order, fill_value=0)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, cmap='YlGnBu', fmt='d', cbar_kws={'label': 'Count'})
plt.xlabel('Best Behavioral Model')
plt.ylabel('Best Neural Model')
plt.title('Heatmap of Best Neural Model vs Best Behavioral Model')
plt.show()
