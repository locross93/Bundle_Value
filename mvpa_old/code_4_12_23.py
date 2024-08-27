# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:31:56 2023

@author: locro
"""

color={'S2S': 'tomato','S2B': 'dodgerblue'}

ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_strain, palette=color, ci=None)    

# Plot the points for each subject
ax2 = sns.stripplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_strain, 
                   dodge=True, jitter=True, marker='o', edgecolor='black', linewidth=1, alpha=0.7)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
#ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot, ci=None)
plt.xticks(rotation=90)

# Remove the default legend
ax.get_legend().remove()

# Create custom legend for bars
legend_elements = [plt.Line2D([0], [0], color='tomato', lw=4, label='Single Item'),
                   plt.Line2D([0], [0], color='dodgerblue', lw=4, label='Bundle')]

legend = Legend(ax, legend_elements, ['Single Item', 'Bundle'], title='Test On', bbox_to_anchor=(1.04, 1), loc="upper left")
ax.add_artist(legend)

# Create custom legend for subjects
unique_subjects = df_plot_strain['Subj'].unique()
legend_elements2 = [plt.Line2D([0], [0], marker='o', color='w', label=subj,
                               markerfacecolor=ax2.get_children()[i].get_facecolor()[0]) for i, subj in enumerate(unique_subjects)]

legend2 = Legend(ax, legend_elements2, unique_subjects, title='Subjects', bbox_to_anchor=(1.04, 0.7), loc="upper left")
ax.add_artist(legend2)

plt.ylabel('Accuracy (Pearson r)', fontsize=14)
plt.xlabel('ROI', fontsize=16)
plt.title('Cross Decoding Value - Train on Single Items')




# Define a color palette with 14 colors for the 14 subjects
color_palette = sns.color_palette("hls", 14)

# Create a dictionary that maps each subject to a color
color_dict = dict(zip(df_plot_strain['Subj'].unique(), color_palette))

# Plot the barplot as before
color={'S2S': 'tomato','S2B': 'dodgerblue'}
ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_strain, palette=color, ci=None)

# Plot the stripplot with the color_dict as the hue parameter
ax2 = sns.stripplot(x="Mask", y="Accuracy", hue="Subj", data=df_plot_strain, dodge=True, jitter=True, marker='o', edgecolor='black', linewidth=1, alpha=0.7, palette=color_dict)

plt.xticks(rotation=90)

# Remove the default legend
ax.get_legend().remove()

# Create custom legend for bars
legend_elements = [plt.Line2D([0], [0], color='tomato', lw=4, label='Single Item'),
                   plt.Line2D([0], [0], color='dodgerblue', lw=4, label='Bundle')]
legend = Legend(ax, legend_elements, ['Single Item', 'Bundle'], title='Test On', bbox_to_anchor=(1.04, 1), loc="upper left")
ax.add_artist(legend)

# Create custom legend for subjects using the color_dict
legend_elements2 = [plt.Line2D([0], [0], marker='o', color='w', label=subj,
                               markerfacecolor=color_dict[subj]) for subj in df_plot_strain['Subj'].unique()]
legend2 = Legend(ax, legend_elements2, df_plot_strain['Subj'].unique(), title='Subjects', bbox_to_anchor=(1.04, 0.7), loc="upper left")
ax.add_artist(legend2)

plt.ylabel('Accuracy (Pearson r)', fontsize=14)
plt.xlabel('ROI', fontsize=16)
plt.title('Cross Decoding Value - Train on Single Items')







# Define a color palette with 14 colors for the 14 subjects
color_palette = sns.color_palette("hls", 14)

# Create a dictionary that maps each subject to a color
color_dict = dict(zip(df_plot_strain['Subj'].unique(), color_palette))

# Plot the barplot as before
color={'S2S': 'tomato','S2B': 'dodgerblue'}
ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_strain, palette=color, ci=None)

# Get the x and y positions of the bars
x_pos = ax.get_xticks()
y_pos = df_plot_strain.groupby(['Mask', 'Decoding Type'])['Accuracy'].mean().values

# Get the width and height of the bars
width = ax.patches[0].get_width()
height = ax.patches[0].get_height()

# Get the dodge parameter for the stripplot
dodge = sns.utils.categorical._GroupPlotter('Mask', 'Accuracy', 'Decoding Type', df_plot_strain).estimate_dodge(width)

# Loop over the rows of the data frame and plot each point using plt.scatter
for i, row in df_plot_strain.iterrows():
    # Get the x and y coordinates of the point
    x = x_pos[row['Mask']] + dodge[row['Decoding Type']] - width / 2
    y = row['Accuracy']
    
    # Get the color of the point from the color_dict
    c = color_dict[row['Subj']]
    
    # Plot the point using plt.scatter with some jitter and other parameters
    plt.scatter(x + np.random.uniform(-0.1, 0.1), y, marker='o', edgecolor='black', linewidth=1, alpha=0.7, color=c)

plt.xticks(rotation=90)

# Remove the default legend
ax.get_legend().remove()

# Create custom legend for bars
legend_elements = [plt.Line2D([0], [0], color='tomato', lw=4, label='Single Item'),
                   plt.Line2D([0], [0], color='dodgerblue', lw=4, label='Bundle')]
legend = Legend(ax, legend_elements, ['Single Item', 'Bundle'], title='Test On', bbox_to_anchor=(1.04, 1), loc="upper left")
ax.add_artist(legend)

# Create custom legend for subjects using the color_dict
legend_elements2 = [plt.Line2D([0], [0], marker='o', color='w', label=subj,
                               markerfacecolor=color_dict[subj]) for subj in df_plot_strain['Subj'].unique()]
legend2 = Legend(ax, legend_elements2, df_plot_strain['Subj'].unique(), title='Subjects', bbox_to_anchor=(1.04, 0.7), loc="upper left")
ax.add_artist(legend2)

plt.ylabel('Accuracy (Pearson r)', fontsize=14)
plt.xlabel('ROI', fontsize=16)
plt.title('Cross Decoding Value - Train on Single Items')