import pandas as pd
import os
from operator import itemgetter
from github_search import text_preprocessing

input_file_path = 'github_repos.json'


raw_df = pd.read_json(os.path.join('data', input_file_path), lines=True)
raw_df.head()


raw_df['languages'] = raw_df['language'].apply(lambda ds: [d['name'] for d in ds])


selected_langs = ['python', 'r', 'matlab', 'julia', 'c++', 'java', 'scala']
df = raw_df[raw_df['languages'].apply(lambda langs: any([lang.lower() in selected_langs for lang in langs]))]
df = df.drop(['language'], axis=1)
df = df[(~df['content'].isna())]
df = df[df['content'].str.split().apply(len) > 25]
df = df[(df['content'].apply(itemgetter(0)) != '<') & (df['content'].apply(itemgetter(-1)) != '>')]


n_examples = 10000
print('selected_n_examples: {}'.format(n_examples))
lm_df = df[['repo_name', 'languages', 'content']][:n_examples]

lm_df = lm_df.dropna()


lm_df[['content']].to_csv('github_repos_lm_text_small.csv')

lm_df.index = pd.RangeIndex(len(lm_df))


import tqdm

extracted_content = pd.Series([text_preprocessing.tokenize_markdown(md_string) for md_string in tqdm.tqdm(lm_df['content'])])
lm_df['text'] = extracted_content.apply(' '.join)
lm_df = lm_df[(~lm_df['text'].isna()) & (lm_df['text'].apply(len) > 0)]
print('filtered_n_examples: {}'.format(lm_df.shape[0]))
out_file = 'github_repos_lm_text.csv'
print('saving results to: {}'.format(out_file))
lm_df[['text']].to_csv(out_file)

# ### Load to FastAI api


bs = 64


#import fastai
#import nltk



#markdown_tokenizer = fastai.text.Tokenizer(
#    tok_func=tok_fn,
#    pre_rules=[],
#    post_rules=[],
#    special_cases=[],
#    n_cpus=12)

#def tok_fn(lang):
#    tok = fastai.text.transform.BaseTokenizer('none')
#    tok.tokenizer = tokenize_markdown
#    return tok
#
#
#data_lm = load_data('', 'data_lm_export.pkl', bs=bs, bptt=50)

