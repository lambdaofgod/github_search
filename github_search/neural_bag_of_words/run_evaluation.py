# %%
from github_search.neural_bag_of_words.evaluation import *


# %% tags=["parameters"]
product = None
upstream = ["nbow.train-*-*"]
params = None

# %%
yaml_paths = get_yaml_paths(upstream["nbow.train-*-*"])
metrics_df = get_metrics_df(yaml_paths)

# %%
## NBOW evaluation metrics sorted by accuracy@10
### Highlight top metrics
metrics_df.sort_values("accuracy@10", ascending=False).style.highlight_max()

# %%
metrics_df.to_csv(product["metrics"])
