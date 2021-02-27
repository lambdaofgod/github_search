import neptune


def init_neptune_project():
    neptune_token = open('.neptune_token', 'r').read()
    return neptune.init('lambdaofgod/github-search', api_token=neptune_token)


def init_neptune_experiment(experiment_name, params, **kwargs):
    init_neptune_project()
    return neptune.create_experiment(
        experiment_name,
        params=params,
        upload_source_files=["**/*.ipynb"]
    )
