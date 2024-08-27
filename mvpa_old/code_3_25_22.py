#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:54:22 2022

@author: logancross
"""

df = df[cols]

vmpfc_df = tidy_df[tidy_df['ROI'] == 'vmPFC']

avg_corr_by_model = []
for model in models:
    temp_avg_corr = np.mean(vmpfc_df[vmpfc_df['variable'] == model]['value'].values)
    avg_corr_by_model.append(temp_avg_corr)
    
sort_inds = np.argsort(-np.array(avg_corr_by_model))

x_order = [models[i] for i in sort_inds]

model_combos = list(itertools.combinations(x_order , 2))

pairs = []
for combo in model_combos:
    #temp_pair = (('vmPFC', combo[0]),('vmPFC', combo[1]))
    temp_pair = ((combo[0]),(combo[1]))
    pairs.append(temp_pair)

#ax = sns.barplot(x='ROI', y='value', hue='variable', data=vmpfc_df, ci=68)
ax = sns.barplot(x='variable', y='value', order=x_order, data=vmpfc_df, ci=68)
sns.despine()

#annot = Annotator(ax, pairs, data=vmpfc_df, x='ROI', y='value', hue='variable')
annot = Annotator(ax, pairs, data=vmpfc_df, x='variable', y='value')
annot.configure(test='t-test_paired', verbose=2)
#annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', verbose=2)
annot.apply_test()
annot.annotate()

plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
plt.title('RSA Normalized Codes', fontsize=18)
plt.ylabel('RSA Correlation', fontsize=18)
plt.xlabel('ROI', fontsize=18)
plt.show()




vmpfc_df2 = scores_df[scores_df['ROI'] == 'vmPFC']
#vmpfc_df2 = vmpfc_df2.rename(columns={"Relative": "Z-Score"})

avg_corr_by_model = []
for model in models:
    temp_avg_corr = np.mean(vmpfc_df2[model].values)
    avg_corr_by_model.append(temp_avg_corr)
    
sort_inds = np.argsort(-np.array(avg_corr_by_model))

x_order = [models[i] for i in sort_inds]

vmpfc_df2 = vmpfc_df2[x_order]

vmpfc_df = vmpfc_df2.melt(value_vars=x_order)

model_combos = list(itertools.combinations(x_order , 2))

pairs = []
for combo in model_combos:
    #temp_pair = (('vmPFC', combo[0]),('vmPFC', combo[1]))
    temp_pair = ((combo[0]),(combo[1]))
    pairs.append(temp_pair)
    
pairs = pairs[:7]

def_palette = sns.color_palette()

custom_palette = def_palette
custom_palette[0] = sns.color_palette()[2]
custom_palette[2] = sns.color_palette()[0]

#ax = sns.barplot(x='ROI', y='value', hue='variable', data=vmpfc_df, ci=68)
ax = sns.barplot(x='variable', y='value', order=x_order, data=vmpfc_df, ci=68, palette=custom_palette)
sns.despine()

#annot = Annotator(ax, pairs, data=vmpfc_df, x='ROI', y='value', hue='variable')
annot = Annotator(ax, pairs, data=vmpfc_df, x='variable', y='value')
#annot.configure(test='Wilcoxon', verbose=2)
annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', text_format='star', verbose=2)
annot.apply_test()
annot.annotate()

plt.xticks(rotation=90)
#plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
plt.title('RSA Normalized Codes in vmPFC', fontsize=18)
plt.ylabel('RSA Correlation (r)', fontsize=18)
plt.xlabel('Model', fontsize=16)
plt.xticks(rotation=90)
model_labels = ['Divisive','Z-Score','Absolute','Subtractive','WTP - Ref']
plt.xticks([0, 1, 2, 3, 4], model_labels)
plt.show()


mask_names = ['vmPFC','OFCmed','dmPFC']

#reorder inds/rows based on mask_names order
reorder_inds = []
for mask in mask_names:
    mask_inds = np.where(tidy_df['ROI'] == mask)[0].tolist()
    reorder_inds = reorder_inds + mask_inds
    
df_plot = tidy_df.loc[reorder_inds,:]

model_combos = list(itertools.combinations(models, 2))

pairs = []
for mask in mask_names:
    for combo in model_combos:
        temp_pair = ((mask, combo[0]),(mask, combo[1]))
        pairs.append(temp_pair)

#ax = sns.barplot(x='ROI', y='value', hue='variable', data=vmpfc_df, ci=68)
ax = sns.barplot(x='ROI', y='value', data=df_plot, hue='variable', ci=68)
sns.despine()

#annot = Annotator(ax, pairs, data=vmpfc_df, x='ROI', y='value', hue='variable')
annot = Annotator(ax, pairs, data=df_plot, x='ROI', y='value', hue='variable')
#annot.configure(test='Wilcoxon', verbose=2)
annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', text_format='star', verbose=2)
annot.apply_test()
annot.annotate()

plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
plt.title('RSA Normalized Codes', fontsize=18)
plt.ylabel('RSA Correlation (r)', fontsize=18)
plt.xlabel('Model', fontsize=16)
plt.xticks(rotation=0)
plt.show()

