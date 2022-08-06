import pandas as pd


def find_file_import_lines(file_contents_series: pd.Series):
    return file_contents_series.str.split("\n").apply(
        lambda lines: " ".join([line for line in lines if "import" in line])
    )


def prepare_file_imports(product, python_files_path, upsteam=None):
    files_df = pd.read_feather(str(python_files_path))
    imports = find_file_import_lines(files_df["content"])
    files_df["imports"] = imports
    files_df[["repo_name", "path", "imports"]].to_feather(str(product))
