# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:42:22 2023

@author: locro
"""



bundle_path = '/Users/locro/Documents/Bundle_Value/'

def plot_results(subj_df, subj, save=False):
    # train on single item trails
    subj_df_strain = subj_df[(subj_df['Decoding Type'] == 'S2Abs') + (subj_df['Decoding Type'] == 'S2Rel')]
    
    # train on bundle trails
    subj_df_btrain = subj_df[(subj_df['Decoding Type'] == 'B2Abs') + (subj_df['Decoding Type'] == 'B2Rel')]
    
    # Create a (1, 2) subplot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    
    # Define the item_or_bundle list for plotting
    item_or_bundle_list = ['item', 'bundle']
    
    # Iterate over the axes and the item_or_bundle_list
    for ax, item_or_bundle in zip(axes, item_or_bundle_list):
        # Prepare the data for the current item_or_bundle
        if item_or_bundle == 'item':
            df_plot = subj_df_strain # DataFrame for 'item'
            title = 'Sub'+subj+' Cross Decoding Train on Single Item'
            d_pfx = 'S' 
            i = 0
        elif item_or_bundle == 'bundle':
            df_plot = subj_df_btrain # DataFrame for 'bundle'
            title = 'Sub'+subj+' Cross Decoding Train on Bundles'
            d_pfx = 'B'
            i = 1
        
        ax_temp = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot, ci=68, ax=axes[i])
        if i == 0:
            ax.get_legend().remove()
        if i == 1:
            l = plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
            l.get_texts()[0].set_text('Absolute Value')
            l.get_texts()[1].set_text('Relative Value')
        ax.set_ylabel('Accuracy (Pearson r)', fontsize=14)
        ax.set_xlabel('ROI', fontsize=16)
        ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90) 
        
        # statistical annotation
        # test with nonparametric stats - wilcoxon signed rank
        cat_diff_list = []
        uncorr_pvals = []
        for mask_num,mask in enumerate(mask_names):
            mask_abs_scores = df_plot.loc[(df_plot['Mask'] == mask) & (df_plot['Decoding Type'] == d_pfx+'2Abs')]['Accuracy'].values
            mask_rel_scores = df_plot.loc[(df_plot['Mask'] == mask) & (df_plot['Decoding Type'] == d_pfx+'2Rel')]['Accuracy'].values
            cat_diff = np.mean(mask_abs_scores - mask_rel_scores)
            wilcoxon_p = stats.wilcoxon(mask_abs_scores, mask_rel_scores)[1]
            cat_diff_list.append(cat_diff)
            uncorr_pvals.append(wilcoxon_p)
            
        # correct for multiple comparisons
        sig_bools, corr_pvals, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(uncorr_pvals, alpha=0.05, method='fdr_bh')
        for mask_num,sig in enumerate(sig_bools):
            mask = mask_names[mask_num]

            fdr_p = corr_pvals[mask_num]   
            if sig:
                print mask,' Significant S Train',cat_diff_list[mask_num],fdr_p
                annotate_stats2(ax.patches[mask_num],  ax.patches[12+mask_num], ax)
            else:
                print mask,' Not Significant S Train',cat_diff_list[mask_num],fdr_p
    if save:
        save_file = 'sub'+subj+'_xdecode_abs_rel_scores'
        plt.savefig(bundle_path+'figures/'+save_file, dpi=500, bbox_inches='tight')
    plt.show()

subj = '103'

save_path = bundle_path+'mvpa/analyses/sub'+str(subj)
subj_df = pd.read_csv(save_path+'/xdecode_abs_rel_scores_allcvs')

# plot individual subject results
save = False
plot_results(subj_df, subj, save=save)