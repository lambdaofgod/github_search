import os
import shutil
from pathlib import Path

print(Path.cwd())  # /home/skovorodkin/stack

input_file_path = 'nlp_github_repos.csv'

os.mkdir('data')

with open(input_file_path, 'r') as src, open(os.path.join('data', input_file_path), 'w') as dst:
    dst.write(src.read())
