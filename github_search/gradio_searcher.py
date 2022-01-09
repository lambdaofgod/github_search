import gradio as gr
import numpy as np
import pandas as pd

import attr
import pickle

from sklearn import metrics

from github_search.matching_zsl import *
from github_search import github_readmes


@attr.s
class Retriever:

    input_embedder = attr.ib()
    query_embedder = attr.ib()
    zs_learner = attr.ib()
    embeddings_calculated = attr.ib(default=False)

    def set_embeddings(self, X_names, X, X_descriptions=None):
        self.X_embeddings = self.input_embedder.transform(X)
        self.X_df = pd.DataFrame({"input": X})
        if not X_descriptions is None:
            self.X_df["description"] = X_descriptions
        self.X_df.index = X_names
        self.embeddings_calculated = True

    def retrieve_query_results(
        self, query, k=25, similarity=metrics.pairwise.cosine_similarity
    ):
        if not self.embeddings_calculated:
            raise Exception("embeddings not calculated")
        input_embeddings = self.X_embeddings
        y_embeddings = self.query_embedder.transform([query])
        predictions = self.zs_learner.predict_raw(input_embeddings)
        input_target_similarities = similarity(predictions, y_embeddings)
        top_idxs = np.argsort(-input_target_similarities[:, 0])[:k]
        top_similarities = input_target_similarities[top_idxs, 0]
        results_df = self.X_df.iloc[top_idxs]
        results_df["similarity"] = top_similarities
        return results_df.drop(columns=["input"])

    def from_retriever_learner(learner):
        return Retriever(learner.input_embedder, learner.y_embedder, learner.zs_learner)

print("loading data")
readme_data_test = pickle.load(open("output/readme_data_test.pkl", "rb"))
readme_learner = pickle.load(open("output/readme_learner.pkl", "rb"))

print("setting up retriever")
readme_retriever = Retriever.from_retriever_learner(readme_learner)
readme_retriever.set_embeddings(
    readme_data_test.repos, readme_data_test.X, readme_data_test.X
)

def search_f(query):
    results = readme_retriever.retrieve_query_results(query)
    #results['repo'] = results.index
    results['link'] = "https://github.com/" + results.index
    results['description'] = results['description'].apply(lambda s: " ".join(s.split()[:20]))
    return results.reset_index()

iface = gr.Interface(fn=search_f, inputs=["text"], outputs=[gr.outputs.Dataframe(max_rows=50, overflow_row_behaviour="show_ends")]).launch()
