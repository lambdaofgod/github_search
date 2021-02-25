import neptune


def init_neptune(experiment_name, params, **kwargs):
    neptune_token = open('.neptune_token', 'r').read()
    neptune.init('lambdaofgod/github-search', api_token=neptune_token)
    neptune.create_experiment(
        experiment_name,
        params=params,
        upload_source_files=["**/*.ipynb"]
    )
