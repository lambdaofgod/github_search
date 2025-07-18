from dagster import (
    asset,
    multi_asset,
    Definitions,
    Output,
    AssetOut,
    ConfigurableResource,
    load_assets_from_modules,
)
from github_search.pipelines.dagster import (
    code2doc_assets,
    corpora_assets,
    document_expansion,
    call_graph_assets,
    input_assets,
)


def get_assets():
    return (
        load_assets_from_modules([code2doc_assets], group_name="code2doc")
        + load_assets_from_modules([corpora_assets], group_name="corpus")
        + load_assets_from_modules([document_expansion], "librarian")
        + load_assets_from_modules([call_graph_assets], "call_graph")
        + load_assets_from_modules([input_assets], "inputs")
    )


defs = Definitions(
    assets=get_assets(),
    resources={
        "phoenix": code2doc_assets.PhoenixTracker(),
    },
)
