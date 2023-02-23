#!/usr/bin/env python
# coding: utf-8

# In[1]:


# default_exp retrieval_results


# In[ ]:


# In[2]:


import pickle

import attr

# export
import numpy as np
import pandas as pd
from sklearn import metrics

from github_search.matching_zsl import *

# In[3]:


get_ipython().run_line_magic("cd", "..")


# In[48]:


# export


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


# In[49]:


readme_data_test = pickle.load(open("output/readme_data_test.pkl", "rb"))


# In[50]:


readme_learner = pickle.load(open("output/readme_learner.pkl", "rb"))


# In[51]:


readme_retriever = Retriever.from_retriever_learner(readme_learner)
readme_retriever.set_embeddings(
    readme_data_test.repos, readme_data_test.X, readme_data_test.X
)


# In[52]:


distance_learning_results = readme_retriever.retrieve_query_results(
    "similarity learning"
)


# In[53]:


metric_learning_results = readme_retriever.retrieve_query_results("metric learning")


# In[56]:


metric_learning_results


# In[57]:


word_embedding_results = readme_retriever.retrieve_query_results("word embeddings")


# In[58]:


word_embedding_results


# In[13]:


image_generation_results = readme_retriever.retrieve_query_results("image generation")


# In[14]:


image_generation_results


# In[19]:


from github_search import github_readmes

# In[ ]:


# In[26]:


readmes_df = pd.DataFrame(
    {"repo": readme_data_test.repos, "readme": readme_data_test.X.copy()}
)


# In[30]:


image_generation_results.merge(
    readmes_df, left_on="matched_record", right_on="repo"
).drop(columns=["matched_record"]).iloc[:, [1, 2, 0]]


# In[ ]:
