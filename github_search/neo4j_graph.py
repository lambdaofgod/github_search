import pandas as pd
import ast


def prepare_neo4j_dependency_records(
    upstream, id_col, rel_col, graph_dependencies_path, product
):
    raw_dependencies_df = pd.read_json(
        graph_dependencies_path, orient="records", lines=True
    )
    dependencies_df = DependencyExtractor.with_dependency_signatures(
        raw_dependencies_df, id_col, rel_col
    )
    repos_train_df = pd.read_json(
        upstream["prepare_repo_train_test_split"]["train"])
    repos_test_df = pd.read_json(
        upstream["prepare_repo_train_test_split"]["test"])

    _prepare_split(dependencies_df, repos_train_df, product["train"])
    _prepare_split(dependencies_df, repos_test_df, product["test"])


def _merge_split(dependencies_df, repo_split_df):
    df = dependencies_df.merge(repo_split_df, on="repo")
    df["tasks"] = df["tasks"].apply(_ensure_task_list)
    return df


def _ensure_task_list(tasks):
    if type(tasks) is str:
        return ast.literal_eval(tasks)
    else:
        return tasks


def _prepare_split(dependencies_df, repo_split_df, output_path):
    _merge_split(dependencies_df, repo_split_df).to_json(
        output_path, lines=True, orient="records"
    )


class DependencyExtractor:
    selection_cols = ["nodes_HAS_FILE",
                      "nodes_HAS_FUNCTION", "nodes_CALLS_FUNCTION"]

    @classmethod
    def with_dependency_signatures(cls, df, id_col, rel_col):
        df = cls.stack_relations(df, id_col, rel_col)
        return cls.add_dependency_signatures(df)

    @classmethod
    def stack_relations(cls, df, id_col, rel_col):
        rel_types = df[rel_col].unique()
        rel_type = rel_types[0]
        out_df = df[df[rel_col] == rel_type]
        out_df.columns = cls.rename_join_cols(
            df.columns, [id_col, rel_col], rel_type)
        for rel_type in rel_types[1:]:
            merged_df = df[df[rel_col] == rel_type]
            merged_df.columns = cls.rename_join_cols(
                df.columns, [id_col, rel_col], rel_type
            )
            out_df = out_df.merge(merged_df, on=id_col)
        return out_df

    @classmethod
    def add_dependency_signatures(cls, df, max_deps=15):
        df["dependency_signature"] = df.apply(
            lambda row: cls.select_elems(row, cols=cls.selection_cols), axis=1
        )
        return df

    @classmethod
    def select_elems(cls, row, cols, k=15, per_col=5):
        elem_lists = [row[col][:per_col] for col in cols]
        return sum(elem_lists, [])[:k]

    @classmethod
    def rename_join_cols(cls, cols, unrenamed_cols, suffix):
        return unrenamed_cols + [
            col + "_" + suffix for col in cols if not col in unrenamed_cols
        ]
