import pandas as pd


def normalize_df(df, dict_col):
    normalized_col_df = pd.json_normalize(df[dict_col])
    other_df = df.drop(dict_col, axis=1)
    return pd.concat([normalized_col_df, other_df], axis=1)
