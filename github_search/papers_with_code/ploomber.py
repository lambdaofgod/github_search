import pandas as pd
from github_search import utils
from github_search.papers_with_code.paperswithcode_tasks import (
    PapersWithCodeMetadataAggregator, clean_task_name, get_paperswithcode_dfs)


def prepare_paperswithcode_df(paperswithcode_filename, papers_filename, product):
    paper_link_df, paper_df = get_paperswithcode_dfs(
        paperswithcode_filename, papers_filename
    )
    paperswithcode_df = paper_link_df.merge(paper_df, on="paper_url")
    paperswithcode_df["tasks"] = paperswithcode_df["tasks"].apply(
        lambda tasks: [clean_task_name(t) for t in tasks]
    )
    paperswithcode_df = paperswithcode_df[paperswithcode_df["tasks"].apply(len) > 0]
    paperswithcode_df.to_csv(str(product["paperswithcode_path"]), index=False)


def prepare_filtered_paperswithcode_df(upstream, min_task_count, product):
    raw_paperswithcode_df = utils.load_paperswithcode_df(
        str(upstream["prepare_paperswithcode_df"]["paperswithcode_path"])
    )
    deduplicated_paperswithcode_df = PapersWithCodeMetadataAggregator.merge_repos(
        raw_paperswithcode_df
    )
    task_counts = deduplicated_paperswithcode_df["tasks"].explode().value_counts()
    filtered_paperswithcode_df = PapersWithCodeMetadataAggregator.filter_small_tasks(
        deduplicated_paperswithcode_df, task_counts, min_task_count
    )
    filtered_paperswithcode_df.to_csv(str(product["paperswithcode_path"]), index=False)
    task_counts = pd.DataFrame(task_counts).reset_index()
    task_counts.columns = ["task", "task_count"]
    task_counts.to_csv(str(product["task_counts_path"]))
