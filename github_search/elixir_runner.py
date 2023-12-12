import subprocess
import fire
import logging


def download_readmes(input_data_path, output_path):
    elixir_program = f'GithubSearch.ReadmeDownloader.from_repos_csv("{input_data_path}", "{output_path}")'
    subprocess.run(['mix', 'run', '-e', elixir_program], cwd="ghs_ex")


def download_readmes_pb(upstream, product):
    input_data_path = upstream["pwc_data.prepare_final_paperswithcode_df"]["paperswithcode_path"]
    output_path = str(product)
    logging.info(f"Downloading readmes from {input_data_path}")
    logging.info(f"to {output_path}")
    download_readmes(input_data_path, output_path)


if __name__ == '__main__':
    fire.Fire(download_readmes)
