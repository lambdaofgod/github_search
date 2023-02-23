import tqdm
from findkit import index


def get_file_embeddings_by_name(query, files_df, embeddings):
    selected_files_df = files_df[files_df["path"].str.contains(query)]
    return embeddings[selected_files_df.iloc[::10][:10].index].mean(axis=0)


def get_files_similar_to_embedding(emb, indexes, n_files=2):
    similar_files = {
        repo: repo_index.find_similar(emb, n_files)
        for (repo, repo_index) in indexes.items()
    }
    return similar_files


def get_indexes(file_df, embeddings):
    repo_indexes = {}
    for repo in tqdm.tqdm(file_df["repo_name"].unique()):
        repo_df = file_df[file_df["repo_name"] == repo]
        repo_indexes[repo] = index.NMSLIBIndex.build(embeddings[repo_df.index], repo_df)
    return repo_indexes
