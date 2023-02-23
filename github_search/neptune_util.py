import neptune.new as neptune
import json


def init_neptune(tags, neptune_config_path="neptune_stuff.json"):
    neptune_args = json.loads(open(neptune_config_path, "r").read())
    return neptune.init(tags=tags, **neptune_args)


def init_neptune_experiment(experiment_name, params, **kwargs):
    init_neptune_project()
    return neptune.create_experiment(
        experiment_name, params=params, upload_source_files=["**/*.ipynb"]
    )

def get_neptune_callback(param_names, tags=[]):
    run = init_neptune(tags=tags)
    def callback(param_values):
        for param_name in param_names:
            run[param_name].log(param_values[param_name])
    return callback
