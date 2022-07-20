import os
import pandas as pd
import subprocess
from github_search.sentence_embeddings import dbpedia, paperswithcode


def prepare_paperswithcode_data(product):
    url_template = "https://production-media.paperswithcode.com/about/{}.json.gz"
    for prod_name in ["datasets", "methods"]:
        subprocess.Popen(
            ["wget", url_template.format(prod_name), "-O", str(product[prod_name])]
        )


def prepare_dbpedia_machine_learning_data(product):
    dbpedia.get_ml_related_dbpedia_concepts_df().to_csv(str(product), index=None)


def prepare_data(upstream, product):
    dbpedia_df = pd.read_csv(upstream["sentence_embeddings.prepare_dbpedia_data"])
    datasets_df = pd.read_json(upstream["sentence_embeddings.prepare_paperswithcode_data"]["datasets"])
    methods_df = pd.read_json(upstream["sentence_embeddings.prepare_paperswithcode_data"]["methods"])
    dbpedia_df = dbpedia.normalize_dbpedia_df(dbpedia_df)
    datasets_df = paperswithcode.clean_datasets_df(paperswithcode.normalize_datasets_df(datasets_df))
    methods_df = paperswithcode.clean_methods_df(paperswithcode.normalize_methods_df(methods_df))
    pd.concat([dbpedia_df, datasets_df, methods_df]).to_csv(str(product), index=None)
