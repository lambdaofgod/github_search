# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: generation_metrics_experiment
#     language: python
#     name: generation_metrics_experiment
# ---

# %%
import pandas as pd
import pickle
from pathlib import Path
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

# %%
DATASET_PATH = Path("~/Projects/pwc_github_search/").expanduser()
assert DATASET_PATH.exists()

# %%
#with open(DATASET_PATH / "dagster_dfs" / "generation_metrics_df", "rb") as f:
#    generation_metrics_df = pickle.load(f)
generation_metrics_df = pd.read_json("generation_metrics.json")

# %%
per_task_ir_metrics_df = pd.read_csv(DATASET_PATH / "metrics" / "per_query_ir_results.csv")


# %%
per_task_ir_metrics_df.shape

# %%
per_task_ir_metrics_df = per_task_ir_metrics_df.fillna(0)

# %%
per_task_ir_metrics_df = per_task_ir_metrics_df[per_task_ir_metrics_df["retriever"] != "word2vec"]

# %%
task_generation_metrics_df = generation_metrics_df.explode("task").drop(columns=["repo"]).groupby("task").agg("mean", numeric_only=True)

# %%
task_generation_metrics_df

# %%
per_task_ir_metrics_df.columns

# %%
task_generation_metrics_df.columns


# %%
def find_lowest_percentile_by(df, colname, percentile=0.2):
    threshold = df[colname].quantile(percentile)
    return df[df[colname] <= threshold]

def find_highest_percentile_by(df, colname, percentile=0.2):
    threshold = df[colname].quantile(percentile)
    return df[df[colname] > threshold]

def permtest_by_groups(df1, df2, split_by, test_column):
    """
    Run KS test between two dataframes for each group defined by split_by columns.
    
    Args:
        df1: First dataframe (typically the full dataset)
        df2: Second dataframe (typically a subset)
        split_by: Column name(s) to group by (str or list)
        test_column: Column name to run KS test on
    
    Returns:
        DataFrame with KS test results for each group
    """
    if isinstance(split_by, str):
        split_by = [split_by]
    
    results = []
    
    # Group both dataframes by the same columns
    for group_key1, group1 in df1.groupby(split_by):
        for group_key2, group2 in df2.groupby(split_by):
            # Only compare groups with matching keys
            if group_key1 == group_key2:
                values1 = group1[test_column].dropna()
                values2 = group2[test_column].dropna()
                
                if len(values1) > 0 and len(values2) > 0:
                    res = scipy.stats.permutation_test((values1.values, values2.values), alternative="greater", statistic=lambda x,y : np.mean(x) - np.mean(y))
                    
                    result = {
                        'statistic': res.statistic,
                        'p_value': res.pvalue,
                        'n_df1': len(values1),
                        'n_df2': len(values2)
                    }
                    
                    # Add group key columns
                    if len(split_by) == 1:
                        result[split_by[0]] = group_key1
                    else:
                        for i, col in enumerate(split_by):
                            result[col] = group_key1[i]
                    
                    results.append(result)
                break
    
    return pd.DataFrame(results)


# %%
for c in per_task_ir_metrics_df["corpus"].unique():
    print(c.replace("_", " "))


# %%
def colorcode_corpus(corpus_name):
    corpus_name = corpus_name.replace("_", " ")
    if corpus_name  in ["code2doc files summary", "code2doc generated readme"]:
        return "{\\color{green}" + corpus_name + "}"
    elif corpus_name  in ["dependency signature", "generated tasks", "repository signature"]:
        return "{\\color{red}" + corpus_name + "}"
    elif corpus_name  in ["repomap code2doc files summary", "repomap code2doc generated readme"]:
        return "{\\color{cyan}" + corpus_name + "}"
    #elif corpus_name  in ["selected code", "repomap]:
    #    return "\\color{orange" + corpus_name + "}"
    else:
        return corpus_name


# %%
def print_latex_display_df(df, float_format="%.3f"):
    df = df.copy()
    if "corpus" not in df.columns:
        df = df.reset_index()
    df["corpus"] = df["corpus"].apply(colorcode_corpus)
    raw_latex = df.reset_index(drop=True).to_latex(float_format=float_format, index=False)
    
    raw_latex = raw_latex.replace("\\toprule", "\\toprule\n\hline")
    raw_latex = raw_latex.replace("\\\\", "\\\\\n\hline")
    print(raw_latex)


# %%

# %%
def find_lowest_percentile_by(df, colname, percentile=0.2):
    threshold = df[colname].quantile(percentile)
    return df[df[colname] <= threshold]

def find_highest_percentile_by(df, colname, percentile=0.2):
    threshold = df[colname].quantile(percentile)
    return df[df[colname] > threshold]

def kstest_by_groups(df1, df2, split_by, test_column):
    """
    Run KS test between two dataframes for each group defined by split_by columns.
    
    Args:
        df1: First dataframe (typically the full dataset)
        df2: Second dataframe (typically a subset)
        split_by: Column name(s) to group by (str or list)
        test_column: Column name to run KS test on
    
    Returns:
        DataFrame with KS test results for each group
    """
    if isinstance(split_by, str):
        split_by = [split_by]
    
    results = []
    
    # Group both dataframes by the same columns
    for group_key1, group1 in df1.groupby(split_by):
        for group_key2, group2 in df2.groupby(split_by):
            # Only compare groups with matching keys
            if group_key1 == group_key2:
                values1 = group1[test_column].dropna()
                values2 = group2[test_column].dropna()
                
                if len(values1) > 0 and len(values2) > 0:
                    res = scipy.stats.ks_2samp(values1.values, values2.values, alternative="two-sided")
                    
                    result = {
                        'statistic': res.statistic,
                        'p_value': res.pvalue,
                        'n_df1': len(values1),
                        'n_df2': len(values2)
                    }
                    
                    # Add group key columns
                    if len(split_by) == 1:
                        result[split_by[0]] = group_key1
                    else:
                        for i, col in enumerate(split_by):
                            result[col] = group_key1[i]
                    
                    results.append(result)
                break
    
    return pd.DataFrame(results)


# %%
task_generation_metrics_df

# %% [markdown]
# ## fill missing bm25 tasks

# %%
per_task_ir_metrics_df

# %%
task_all_metrics_df = task_generation_metrics_df.merge(per_task_ir_metrics_df, left_on="task", right_on="query", how="right").fillna(0)

# %%
task_all_metrics_df["retriever"].value_counts()

# %%
task_all_metrics_df["corpus"].value_counts()

# %%
import itertools


def split_groupby_lowest_pctiles():
    percentile = 0.
    
    lowest_rouge_metrics_dfs = []
    highest_rouge_metrics_dfs = []
    
    for corpus, retriever in itertools.product(task_all_metrics_df["corpus"].unique(), task_all_metrics_df["retriever"].unique()):
        sel_df = task_all_metrics_df[(task_all_metrics_df["corpus"] == corpus) & (task_all_metrics_df["retriever"] == retriever)]
        sel_lowest_rouge_metrics_df = find_lowest_percentile_by(sel_df, "cross_encoder_score", percentile)
        sel_highest_rouge_metrics_df = find_highest_percentile_by(sel_df, "cross_encoder_score", percentile)
        lowest_rouge_metrics_dfs.append(sel_lowest_rouge_metrics_df)
        highest_rouge_metrics_dfs.append(sel_highest_rouge_metrics_df)
    
    lowest_rouge_metrics_df = pd.concat(lowest_rouge_metrics_dfs)
    highest_rouge_metrics_df = pd.concat(highest_rouge_metrics_dfs)


# %% [markdown]
# ## Correlations

# %%

# %%
tables = dict()
percentile = 0.025
sel_dfs = dict()
corrs = []

col = "Precision@10"

for corpus, retriever in itertools.product(task_all_metrics_df["corpus"].unique(), task_all_metrics_df["retriever"].unique()):
    sel_df = task_all_metrics_df[(task_all_metrics_df["corpus"] == corpus)].copy() #& (task_all_metrics_df["retriever"] == retriever)]
    sel_dfs[(corpus, retriever)] = sel_df
    vertical_var = sel_df["cross_encoder_score"] > sel_df["cross_encoder_score"].quantile(percentile)
    sel_dfs[(corpus, retriever)]["cross_encoder_score>10% percentile?"] = vertical_var
    horizontal_var = sel_df[col]
    tables[(corpus, retriever)] = pd.crosstab(horizontal_var, vertical_var)
    corr = scipy.stats.kendalltau(sel_df["cross_encoder_score"], sel_df[col])
    corrs.append({"corpus": corpus, "retriever": retriever, "\\tau": corr.statistic, "pvalue": corr.pvalue})

corr_df = pd.DataFrame.from_records(corrs)

# %%
corr_df.sort_values("\\tau").drop(columns=["pvalue"]).reset_index(drop=True)

# %%
print_latex_display_df(corr_df.sort_values("\\tau").drop(columns=["pvalue"]).reset_index(drop=True))

# %%

# %% [markdown]
# THIS IS IT:
# - README relationshiop is not stat. significant
# - for everything else is

# %%
chi2df = pd.DataFrame.from_records(
    [
    {"corpus": k[0], "retriever": k[1], "pvalue": scipy.stats.chi2_contingency(tables[k]).pvalue}
        for k in tables.keys()
    ]
).sort_values("pvalue")
chi2df

# %%
chi2_latex = chi2df.to_latex(index=False, float_format="%.2e")

# %%
print_latex_display_df(chi2df, float_format="%.2e")

# %%
sns.histplot(sel_dfs[("generated_tasks", "sentence-transformers/all-mpnet-base-v2")], x="Precision@10", hue="cross_encoder_score>10% percentile?")
plt.savefig("/Users/kuba/Downloads/readme_crossencoder_histograms_st.png")

# %%
sns.histplot(sel_dfs[("generated_tasks", "sentence-transformers/all-mpnet-base-v2")], x="Precision@10", hue="cross_encoder_score>10% percentile?")
plt.savefig("/Users/kuba/Downloads/readme_crossencoder_histograms_st.png")

# %%
sns.histplot(sel_dfs[("code2doc_generated_readme", "bm25")], x="Precision@10", hue="cross_encoder_score>10% percentile?")
plt.savefig("/Users/kuba/Downloads/code2doc_crossencoder_histograms.png")

# %%
sns.histplot(sel_dfs[("readme", "bm25")], x="Precision@10", hue="cross_encoder_score>10% percentile?")
plt.savefig("/Users/kuba/Downloads/readme_crossencoder_histograms.png")

# %%
sns.histplot(sel_dfs[("selected_code", "sentence-transformers/all-mpnet-base-v2")], x="Precision@10", hue="cross_encoder_score>10% percentile?")
plt.savefig("/Users/kuba/Downloads/code2doc_crossencoder_histograms_st.png")

# %%

sns.histplot(tables[("readme", "bm25")][False])

# %%
task_all_metrics_df.groupby(["corpus", "retriever"]).agg("mean", numeric_only=True)[["Accuracy@10", "Precision@10"]].sort_values("Accuracy@10")

# %%
sns.pairplot(
    task_all_metrics_df[(task_all_metrics_df["corpus"] == "readme") & (task_all_metrics_df["retriever"] == "bm25")][["cross_encoder_score", "Precision@10"]],
    plot_kws={'alpha': 0.1}
)
plt.savefig("/Users/kuba/Downloads/readme_precision_vs_cross_encoder.png")

# %%
sns.pairplot(
    task_all_metrics_df[(task_all_metrics_df["corpus"] == "code2doc_generated_readme") & (task_all_metrics_df["retriever"] == "bm25")][["cross_encoder_score", "Precision@10"]],
    plot_kws={'alpha': 0.1}
)
plt.savefig("/Users/kuba/Downloads/generated_readme_precision_vs_cross_encoder.png")

# %%
task_all_metrics_df.columns

# %%
task_all_metrics_df.groupby("query").agg("mean",numeric_only=True)["cross_encoder_score"].plot.hist()

# %%
sns.boxplot(task_all_metrics_df[(task_all_metrics_df["corpus"] == "readme") & (task_all_metrics_df["retriever"] == "bm25")], y="cross_encoder_score", x="Precision@10")
plt.savefig("/Users/kuba/Downloads/readme_precision_vs_cross_encoder.png")

# %%
sns.boxplot(task_all_metrics_df[(task_all_metrics_df["corpus"] == "code2doc_generated_readme") & (task_all_metrics_df["retriever"] == "bm25")], y="cross_encoder_score", x="Precision@10")
plt.savefig("/Users/kuba/Downloads/code2doc_precision_vs_cross_encoder.png")

# %%

# %%
sns.boxplot(task_all_metrics_df[(task_all_metrics_df["corpus"] == "code2doc_generated_readme") & (task_all_metrics_df["retriever"] == "bm25")], y="cross_encoder_score", x="Precision@10")

# %%
ir_metrics_full_results_df = task_all_metrics_df.groupby(["corpus", "retriever"]).agg("mean", numeric_only=True)[["Accuracy@10", "Precision@10"]].sort_values("Accuracy@10")

# %%
ir_metrics_full_results_df

# %%
print_latex_display_df(ir_metrics_full_results_df)

# %%

# %%
ir_metrics_max_results_df = task_all_metrics_df.groupby(["corpus", "retriever"]).agg("mean", numeric_only=True)[["Accuracy@10", "Precision@10"]].reset_index().sort_values("Accuracy@10", ascending=False).drop_duplicates(subset=["corpus"], keep="first")

# %%
ir_metrics_max_results_df

# %%
print_latex_display_df(ir_metrics_max_results_df)

# %%
ir_metrics_mean_results_df = task_all_metrics_df.groupby(["corpus"]).agg("mean", numeric_only=True)[["Accuracy@10", "Precision@10"]].reset_index().sort_values("Accuracy@10", ascending=False)

# %%
ir_metrics_mean_results_df

# %%
print_latex_display_df(ir_metrics_mean_results_df)

# %%

# %%
print(corpus, retriever)
sns.boxplot(sel_df, x="Accuracy@10", y="cross_encoder_score")

# %%

# %%

# %% [markdown]
# ## Basic corpora metrics

# %%

# %%
## Basic corpora metrics

# %%
basic_corpora = ["readme", "selected_code", "repomap", "dependency_signature"]
ir_metrics_max_results_df[ir_metrics_max_results_df["corpus"].isin(basic_corpora)]

# %%
print_latex_display_df(ir_metrics_max_results_df[ir_metrics_max_results_df["corpus"].isin(basic_corpora)])

# %%
## Signature corpora metrics

# %%
signature_corpora = ["readme", "repository_signature", "generated_tasks", "dependency_signature"]
ir_metrics_max_results_df[ir_metrics_max_results_df["corpus"].isin(signature_corpora)]

# %%
print_latex_display_df(ir_metrics_max_results_df[ir_metrics_max_results_df["corpus"].isin(signature_corpora)])

# %%
code2doc_corpora = ["readme"] + [corpus for corpus in ir_metrics_max_results_df["corpus"].unique() if "code2doc" in corpus]
ir_metrics_max_results_df[ir_metrics_max_results_df["corpus"].isin(code2doc_corpora)]


# %%
print_latex_display_df(ir_metrics_max_results_df[ir_metrics_max_results_df["corpus"].isin(code2doc_corpora)])

# %%
# Best results for each corpus

# %%
print_latex_display_df(ir_metrics_max_results_df)

# %%
print_latex_display_df(ir_metrics_mean_results_df)


# %%

# %%

# %%

# %%
def split_by_qtile_select(df, col, qtile, selected_col):
    df_hi, df_lo = split_by_qtile(df, col, qtile)
    return df_hi[selected_col], df_lo[selected_col]


# %%
task_all_metrics_df.groupby(["corpus", "retriever"]).apply(
    lambda df: pd.Series(scipy.stats.mannwhitneyu(*split_by_qtile_select(df, "cross_encoder_score", 0.1, "Precision@10")), index=["statistic", "pvalue"])
).sort_values("pvalue")

# %%
task_all_metrics_df.groupby(["corpus", "retriever"]).apply(
    lambda df: pd.Series(scipy.stats.ks_2samp(*split_by_qtile_select(df, "cross_encoder_score", 0.1, "Precision@10")), index=["statistic", "pvalue"])
).sort_values("pvalue")

# %%
Comparing precision/accuracy of overall vs 


# %%
def compare_aggregates(df1, df2):
    return df1.describe() - df2.describe()


# %%
def compare_aggregates_rel(df1, df2):
    return (df1.describe() - df2.describe()) / df1.describe()


# %%
task_all_metrics_df.groupby(["corpus", "retriever"]).apply(
    lambda df: compare_aggregates(*split_by_qtile_select(df, "cross_encoder_score", 0.1, "Precision@10"))
).sort_values("mean").reset_index()

# %%
task_all_metrics_df.groupby(["corpus", "retriever"]).apply(
    lambda df: compare_aggregates_rel(*split_by_qtile_select(df, "cross_encoder_score", 0.1, "Precision@10"))
).sort_values("mean").reset_index()


# %%
def split_lo_vs_overall(df, colname, qtile):
    lo_df = find_lowest_percentile_by(df, colname, qtile)
    return df, lo_df


def split_lo_vs_overall_select(df, colname, qtile, sel_colname):
    lo_df = find_lowest_percentile_by(df, colname, qtile)
    return df[sel_colname], lo_df[sel_colname]


# %%
ir_metrics_col = "Precision@10"

precision_df = task_all_metrics_df.groupby(["corpus", "retriever"]).apply(
    lambda df: df[ir_metrics_col].describe()
)["mean"].rename(ir_metrics_col)

# %%
lo_precision_df = task_all_metrics_df.groupby(["corpus", "retriever"]).apply(
    lambda df: find_lowest_percentile_by(df, "cross_encoder_score", 0.1)[ir_metrics_col].describe()
)["mean"].rename(f"hard tasks {ir_metrics_col}")

# %%
((precision_df - lo_precision_df) / precision_df).sort_values()

# %%

hard_tasks_precision_difference_df = pd.DataFrame({
    ir_metrics_col: precision_df,
    f"{ir_metrics_col} (lowest cross encoder score tasks)": lo_precision_df,
    "difference": (precision_df - lo_precision_df), "relative difference": (precision_df - lo_precision_df) / precision_df * 100})
hard_tasks_precision_difference_df = hard_tasks_precision_difference_df.reset_index()#.sort_values(ir_metrics_col, ascending=False).drop_duplicates(subset=["corpus"])
hard_tasks_precision_difference_df = hard_tasks_precision_difference_df.sort_values("relative difference")

# %%
hard_tasks_precision_difference_df

# %%
hard_tasks_precision_difference_df["relative difference"] = hard_tasks_precision_difference_df["relative difference"].round(1).apply(lambda v: str(v) + "%")

# %%
print_latex_display_df(hard_tasks_precision_difference_df)

# %%
print_latex_display_df(hard_tasks_precision_difference_df)

# %%

# %%
precision_differences_df = task_all_metrics_df.groupby(["corpus", "retriever"]).apply(
    lambda df: compare_aggregates(*split_lo_vs_overall_select(df, "cross_encoder_score", 0.1, "Precision@10"))
)["mean"].rename("Precision@10 difference")

# %%
rel_precision_differences_df = task_all_metrics_df.groupby(["corpus", "retriever"]).apply(
    lambda df: compare_aggregates_rel(*split_lo_vs_overall_select(df, "cross_encoder_score", 0.1, "Precision@10"))
)["mean"].rename("relative Precision@10 difference")

# %%
pd.concat([precision_differences_df, rel_precision_differences_df], axis=1).sort_values("relative Precision@10 difference")

# %%
