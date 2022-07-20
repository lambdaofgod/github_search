def clean_datasets_df(df):
    valid_description_index = ~df["description"].str.contains(
        "Click to add a brief description of the dataset"
    )
    valid_tasks_index = df["labels"].apply(len) > 0
    return df[valid_description_index & valid_tasks_index].dropna(subset="description")


def normalize_datasets_df(df):
    df = df.copy()
    df["labels"] = df["tasks"].apply(
        lambda task_list: [t.get("task").lower() for t in task_list]
    )
    return df[["name", "full_name", "labels", "description"]]


def clean_methods_df(df):
    valid_description_index = ~df["description"].str.contains(
        "Please enter a description about the method here"
    )
    return df[valid_description_index]


def normalize_methods_df(df):
    df["labels"] = df["collections"].apply(
        lambda task_list: [t.get("collection").lower() for t in task_list]
    )
    return df[["name", "full_name", "labels", "description"]].dropna(
        subset="description"
    )
