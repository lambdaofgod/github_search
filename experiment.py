import os
import pandas as pd

DATA_DIR = 'data'
pd.read_csv(os.path.join(DATA_DIR, 'nlp_github_repos.csv'))