from dagster import (
    asset,
    multi_asset,
    Definitions,
    Output,
    AssetOut,
    ConfigurableResource,
    load_assets_from_modules,
)
from github_search.pipelines.dagster import code2doc_assets, corpora_assets


def get_assets():
    return load_assets_from_modules(
        [code2doc_assets], group_name="code2doc"
    ) + load_assets_from_modules([corpora_assets], group_name="corpus")


defs = Definitions(
    assets=get_assets(),
    resources={
        "code2doc_config": code2doc_assets.Code2DocConfig(),
        "corpus_config": corpora_assets.CorpusConfig(),
        "phoenix": code2doc_assets.PhoenixTracker(),
    },
)
