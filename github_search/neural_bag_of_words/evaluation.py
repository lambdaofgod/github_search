import pathlib
import yaml
import pandas as pd


def get_metrics_dict_with_name(path, get_name=lambda p: p.parent.name):
    name = get_name(path)
    with open(path, "r") as f:
        metrics = yaml.safe_load(f)
    return (name, metrics)


def merge_at_key(pattern_str, fill_value, placeholder="k"):
    return pattern_str.replace(placeholder, str(fill_value))


def get_flattened_keys(key, values, merge_keys=merge_at_key):
    if type(values) is dict:
        return [
            (merge_keys(key, subk), subvalue) for (subk, subvalue) in values.items()
        ]
    else:
        return [(key, values)]


def get_single_metrics_df(name, metrics_dict):
    df = pd.DataFrame(
        [
            val
            for k in metrics_dict.keys()
            for val in get_flattened_keys(k, metrics_dict[k])
        ]
    ).set_index(0)
    name_df = pd.DataFrame.from_records([("name", name)]).set_index(0)
    return pd.concat([name_df, df]).T


def get_metrics_df_from_dicts(metrics_dicts, similarity="cos_sim"):
    return pd.concat(
        [
            get_single_metrics_df(name, metrics_dicts[name]["cos_sim"])
            for name in metrics_dicts.keys()
        ]
    ).set_index("name")


def get_yaml_paths(upstream_dict):
    upstream_partial_paths = [
        pathlib.Path(str(upstream_dict[k]["document_nbow"])).parent
        for k in upstream_dict.keys()
    ]
    return [list(p.glob("*yaml"))[0] for p in upstream_partial_paths]


def get_metrics_df(yaml_paths, get_name=lambda p: p.parent.name):
    metrics_dicts = dict((get_metrics_dict_with_name(p, get_name) for p in yaml_paths))
    return get_metrics_df_from_dicts(metrics_dicts)
