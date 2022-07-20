from mlutil import semantic_web_utils


def get_ml_related_dbpedia_concepts_df():
    entity_names = [
        "Machine_learning",
        "Deep_learning",
        "Statistical_learning_theory",
        "Natural_language_processing",
        "Computer_vision",
        "Time_series",
        "Reinforcement_learning",
        "Neural_networks",
        "Feature_engineering",
        "Supervised_learning",
        "Unsupervised_learning",
        "Pattern_recognition",
        "Learning_algorithm",
    ]

    entities = [pref + name for pref in ["dbc:", "dbr:"] for name in entity_names]

    return (
        semantic_web_utils.make_dataframe_from_results(
            semantic_web_utils.get_related_concepts_results(entities)
        )
    )

def normalize_dbpedia_df(df):
    df = df.copy()
    df["full_name"] = df["label"]
    df["name"] = df["child"]
    df["labels"] = [[]] * len(df)
    df["description"] = df["abstract"]
    df["type"] = "dbpedia"
    return df[["name", "full_name", "labels", "description"]]
