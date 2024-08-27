# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:36:00 2023

@author: locro
"""
from matplotlib.text import Text

df_plot['Subj'] = df_plot['Subj'].replace({101: '001', 102: '002', 103: '003', 
       104: '004', 105: '005', 106: '006', 107: '007', 108: '008', 109: '009', 
       110: '010', 111: '011', 112: '012', 113: '013', 114: '014'})

#####################################
# Train on Single Items plot
# Train on Single Items plot
df_plot_strain = df_plot[(df_plot['Decoding Type'] == 'S2S') + (df_plot['Decoding Type'] == 'S2B')]
df_plot_strain = df_plot_strain.reset_index(drop=True)

color={'S2S': 'tomato','S2B': 'dodgerblue'}
xtick_labels = [Text(0, 0, 'rACC'), Text(1, 0, 'dACC'), Text(2, 0, 'vlPFC'), Text(3, 0, 'vmPFC'), Text(4, 0, 'OFCant'), Text(5, 0, 'OFClat'), Text(6, 0, 'OFCmed'), Text(7, 0, 'OFCpost'), Text(8, 0, 'dmPFC'), Text(9, 0, 'dlPFC'), Text(10, 0, 'MFG'), Text(11, 0, 'IFG')]

g = sns.catplot(x="Mask", y="Accuracy", hue="Decoding Type", col="Subj", data=df_plot_strain,
kind="bar", ci=68, palette=color, legend=False, col_wrap=5, aspect=1.5)

g = sns.catplot(x="Mask", y="Accuracy", hue="Decoding Type", col="Subj", data=df_plot_strain,
kind="bar", ci=None, palette=color, legend=False, col_wrap=5, aspect=1.5)

# add a global title
g.fig.suptitle('Cross Decoding Value - Train on Single Items', fontsize=34)

# adjust the spacing between the title and the subplots
g.fig.subplots_adjust(top=0.9)

for ax in g.axes:
  ax.set_title(ax.get_title(), fontdict={'size': 20})
  ax.tick_params(labelbottom=True)
  #ax.set_xticklabels(labels=mask_names)
  #plt.setp(ax.get_xticklabels(), visible=True)
  #plt.setp(xtick_labels, visible=True)
  #plt.setp(ax.get_title(), fontsize=20)
  
lg = plt.legend(['Single Item', 'Bundle'], prop={'size':26}, 
           bbox_to_anchor=(1.04, 1), loc="upper left")
lg.set_title('Test On', prop={'size': 30})
#g.set_titles('Subject {col_name}', fontdict={'size': 40})
#g.set_xticklabels(labels=mask_names)
g.set_ylabels('Accuracy (Pearson r)', fontdict={'size': 20})
g.set_xlabels('ROI', fontdict={'size': 20})


h, l = ax.get_legend_handles_labels()
labels=['Single Item','Bundle']
#g.legend(h, labels, title='Test On', bbox_to_anchor=(1.04,1), loc="upper left")
plt.ylabel('Accuracy (Pearson r)', fontsize=14)
plt.xlabel('ROI', fontsize=16)
plt.savefig('/Users/locro/Documents/Bundle_Value/figures/xdecode_roi_individual_subj_strain.png', dpi=300)


#####################################
# Train on Bundles plot
df_plot_btrain = df_plot[(df_plot['Decoding Type'] == 'B2S') + (df_plot['Decoding Type'] == 'B2B')]
df_plot_btrain = df_plot_btrain.reset_index(drop=True)

color={'B2S': 'tomato','B2B': 'dodgerblue'}

g = sns.catplot(x="Mask", y="Accuracy", hue="Decoding Type", col="Subj", data=df_plot_btrain,
kind="bar", ci=None, palette=color, legend=False, col_wrap=5, aspect=1.5)

# add a global title
g.fig.suptitle('Cross Decoding Value - Train on Bundles', fontsize=34)

# adjust the spacing between the title and the subplots
g.fig.subplots_adjust(top=0.9)

for ax in g.axes:
  ax.set_title(ax.get_title(), fontdict={'size': 20})
  ax.tick_params(labelbottom=True)
  #ax.set_xticklabels(labels=mask_names)
  #plt.setp(ax.get_xticklabels(), visible=True)
  #plt.setp(xtick_labels, visible=True)
  #plt.setp(ax.get_title(), fontsize=20)
  
lg = plt.legend(['Single Item', 'Bundle'], prop={'size':26}, 
           bbox_to_anchor=(1.04, 1), loc="upper left")
lg.set_title('Test On', prop={'size': 30})
#g.set_titles('Subject {col_name}', fontdict={'size': 40})
#g.set_xticklabels(labels=mask_names)
g.set_ylabels('Accuracy (Pearson r)', fontdict={'size': 20})
g.set_xlabels('ROI', fontdict={'size': 20})


h, l = ax.get_legend_handles_labels()
labels=['Single Item','Bundle']
#g.legend(h, labels, title='Test On', bbox_to_anchor=(1.04,1), loc="upper left")
plt.ylabel('Accuracy (Pearson r)', fontsize=14)
plt.xlabel('ROI', fontsize=16)
plt.savefig('/Users/locro/Documents/Bundle_Value/figures/xdecode_roi_individual_subj_btrain.png', dpi=300)