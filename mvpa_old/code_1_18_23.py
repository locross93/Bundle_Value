# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:31:26 2023

@author: locro
"""


vmpfc_df = tidy_df[tidy_df['ROI'] == 'vmPFC']

mask_names = ['vmPFC']

pairs = []
for mask in mask_names:
    for combo in model_combos:
        temp_pair = ((mask, combo[0]),(mask, combo[1]))
        pairs.append(temp_pair)

ax = sns.barplot(x='ROI', y='value', hue='variable', data=vmpfc_df, ci=68)
sns.despine()
plt.legend(bbox_to_anchor=(0.95, 1))

annot = Annotator(ax, pairs, data=vmpfc_df, x='ROI', y='value', hue='variable')
annot.configure(test='Wilcoxon', verbose=2)
#annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', verbose=2)
#annot.configure(test='t-test_paired', verbose=2)
annot.apply_test()
annot.annotate()

plt.xticks(rotation=90)
#plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
plt.title('RSA Normalized Codes', fontsize=18)
plt.ylabel('Adj R2', fontsize=18)
plt.xlabel('ROI', fontsize=18)
plt.show()



pairs = []
for combo in model_combos:
    #temp_pair = (('vmPFC', combo[0]),('vmPFC', combo[1]))
    temp_pair = ((combo[0]),(combo[1]))
    pairs.append(temp_pair)

ax = sns.barplot(x='variable', y='value', order=x_order, data=vmpfc_df, ci=68)
sns.despine()

#annot = Annotator(ax, pairs, data=vmpfc_df, x='ROI', y='value', hue='variable')
annot = Annotator(ax, pairs, data=vmpfc_df, x='variable', y='value')
#annot.configure(test='t-test_paired', verbose=2)
annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', verbose=2)
annot.apply_test()
annot.annotate()

plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
plt.title('RSA Normalized Codes in vmPFC', fontsize=18)
plt.ylabel('RSA Correlation', fontsize=18)
plt.xlabel('ROI', fontsize=18)
plt.show()