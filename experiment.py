import os
import pandas as pd

DATA_DIR = 'data'
df = pd.read_csv(os.path.join(DATA_DIR, 'file.csv'))
print('shape:', df.shape)
