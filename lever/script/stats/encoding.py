##
import numpy as np
import pandas as pd
from lever.script.steps.encoding import proj_folder

data_folder = proj_folder.joinpath("data", "analysis")

## Calculate significance
## two-way ranked ANOVA
from scipy.stats import mannwhitneyu
from rpy2 import robjects as robj
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
artool = importr("ARTool")
pandas2ri.activate()

def art_2by2(df: pd.DataFrame, feature: str, group: str):
    feature_wide = df[["Unnamed: 0", feature, "all", "group"]]
    feature_long = feature_wide.melt(id_vars=["Unnamed: 0", "group"], value_vars=[feature, "all"])
    feature_long = feature_long.query(f"group in ('wt', '{group}')")
    feature_long.loc[feature_long["value"] < 0, "value"] = 0
    feature_long["group"] = feature_long["group"].astype("category")
    feature_long["variable"] = feature_long["variable"].astype("category")
    r_df = pandas2ri.py2rpy(feature_long)
    model = artool.art(robj.Formula("value ~ group * variable"), data=r_df)
    anova = robj.r["anova"]
    return anova(model)

def wilcox(df: pd.DataFrame, feature):
    res = {group: np.clip(df.query(f"group == '{group}'")[feature], 0, None)
           for group in ("wt", "dredd", "glt1")}
    print(feature)
    print([(key, np.median(x)) for key, x in res.items()])
    print(" < ".join([list(res.keys())[y] for y in np.argsort([np.median(x) for x in res.values()])]))
    print("wt vs. dreadd: ", mannwhitneyu(res["wt"], res["dredd"]))
    print("wt vs. glt1: ", mannwhitneyu(res["wt"], res["glt1"]))
##
data = pd.read_csv(data_folder.joinpath("encoding_minimal.csv"))
[wilcox(data, x) for x in ("start", "reward", "isMoving", "trajectory")]
[wilcox(data, x) for x in ("all", "hit", "delay", "speed")]
##
data = pd.read_csv(data_folder.joinpath("encoding_minimal.csv"))
[art_2by2(data, x, 'glt1') for x in ("hit", "delay")]
[art_2by2(data, x, 'dredd') for x in ("hit", "delay")]
[art_2by2(data, x, 'glt1') for x in ("start", "reward", "isMoving", "trajectory", "speed")]
[art_2by2(data, x, 'dredd') for x in ("start", "reward", "isMoving", "trajectory", "speed")]
##
