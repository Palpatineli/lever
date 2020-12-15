##
from pathlib import Path 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mplplot import Figure
from scipy.stats import mannwhitneyu
proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe/")
table = pd.read_csv(proj_folder.joinpath("data", "analysis", "related.csv"))
colors = ["#e0bad7", "#92dbe6", "#70ee9c", "#fb9f89", "#5688c7"]

##
table.loc[table['group'] == "cno", "group"] = 'dredd'
table['related'] = table['p'] < 0.05
means = table.groupby(["id", "group"]).mean()
##
plt.boxplot(table['related'])


gb = means.groupby('group')    
groups = {x: gb.get_group(x)['related'].values for x in gb.groups}
plt.boxplot(groups.values(), labels=groups.keys())

mannwhitneyu(groups['wt'], groups['glt1'])
np.median(groups['dredd'])
with Figure(proj_folder.joinpath("report", "fig", "motion_related.svg"), (9, 6)) as axes:
    ax = axes[0]
    sns.swarmplot(x='group', y='related', data=means.reset_index(), ax=ax)
    sns.boxplot(x='group', y='related', data=means.reset_index(), boxprops={"alpha": .3}, ax=ax)
