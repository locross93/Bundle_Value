import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the data
subj_model_fits = pandas.read_csv('results/bayesian_model_comparison.csv')

# find the best model for each subject, and count how many times each model was the best
# take the argmax across columns to get the best model per subject (row)
best_model = subj_model_fits.idxmax(axis=1)

# make a new dataframe taking the different models in subj_model_fits as a column, and the counts of each model as the best in other column
# get unique model names from the columns of subj_model_fits
models = subj_model_fits.columns
# count how many times each model was the best
best_model_counts = best_model.value_counts()
# make a new dataframe with the best model counts
best_model_counts = best_model_counts.to_frame().reset_index()
best_model_counts.columns = ['model', 'count']
# add a row for each model that wasn't the best for any subject
best_model_counts = best_model_counts.append(
    [{'model': model, 'count': 0} for model in models if model not in best_model_counts['model'].values],
    ignore_index=True
)

# make a list exceedance probabilities for each model with the elements: 0.5135 0.2765 0.0021 0.2080
exc_probs = [0.5135, 0.2765, 0.0021, 0.2080]

sn_palette='tab10'
#sn_palette='Set2'

# plot the exceedance probabilities
plt.figure(figsize=(8, 6))
sns.barplot(x=models, y=exc_probs, palette=sn_palette)
sns.despine(top=True, right=True)
plt.title('Exceedance probabilities', fontsize=14)
plt.ylabel('Exceedance probability', fontsize=16)
plt.yticks(fontsize=12)
plt.xlabel('Model', fontsize=16)
plt.tight_layout()
# if save_file is not None:
#     plt.savefig(save_file, dpi=300, bbox_inches='tight')
plt.show()

sns.barplot(x='model', y='count', data=best_model_counts, palette=sn_palette)
sns.despine(top=True, right=True)
#plt.xticks(np.arange(len(labels)), labels, fontsize=18)
plt.title('How many participants were\n best explained by each model', fontsize=14)
plt.ylabel('Number of participants', fontsize=16)
plt.yticks(fontsize=12)
plt.xlabel('Model', fontsize=16)
plt.tight_layout()
# if save_file is not None:
#     plt.savefig(save_file, dpi=300, bbox_inches='tight')
plt.show()

save_file = '/Users/locro/Documents/Bundle_Value/figures/bayesian_model_comparison'

# Create a figure with 1 row and 2 columns of subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plotting Exceedance probabilities
sns.barplot(x=models, y=exc_probs, palette=sn_palette, ax=axs[0])
sns.despine(top=True, right=True, ax=axs[0])
axs[0].set_title('Exceedance probabilities', fontsize=20)
axs[0].set_ylabel('Exceedance probability', fontsize=16)
axs[0].tick_params(axis='y', labelsize=16)
axs[0].tick_params(axis='x', labelsize=16)
axs[0].set_xlabel('Model', fontsize=20)
# Plotting How many participants were best explained by each model
sns.barplot(x='model', y='count', data=best_model_counts, palette=sn_palette, ax=axs[1], order=models)
sns.despine(top=True, right=True, ax=axs[1])
axs[1].set_title('How many participants were\n best explained by each model', fontsize=20)
axs[1].set_ylabel('Number of participants', fontsize=16)
axs[1].tick_params(axis='y', labelsize=16)
axs[1].tick_params(axis='x', labelsize=16)
axs[1].set_xlabel('Model', fontsize=20)
# Adjust layout
plt.tight_layout()
# Add overall title
fig.suptitle('Bayesian Model Comparison', fontsize=24, y=1.05)
# Optional: save the figure if save_file is defined
if save_file is not None:
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
# Show the plots
plt.show()



