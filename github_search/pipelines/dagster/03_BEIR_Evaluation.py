#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json

from beir.retrieval.search.lexical import BM25Search as BM25


from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models import SPLADE, SentenceBERT, UniCOIL
from beir.retrieval.search.sparse import SparseSearch


from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from github_search.evaluation.beir_evaluation import EvaluateRetrievalCustom as EvaluateRetrieval, CorpusDataLoader
from beir.retrieval.search.lexical import BM25Search as BM25

import sentence_transformers


# In[2]:


import pickle 

with open("/home/kuba/Projects/github_search/.dagster/storage/corpus_information", "rb") as f:
    corpora = json.loads(pickle.load(f))

[(cname, len(corpora[cname].keys())) for cname in corpora.keys()]


# In[3]:


list(corpora["repository_signature"].items())[:10]


# In[4]:


corpora.keys()


# In[5]:


def get_repos_for_query(query, repos_df):
    return repos_df[repos_df["tasks"].apply(lambda ts: query in ts)]


def get_queries(repos_df, min_query_count):
    all_queries = repos_df["query_tasks"].explode()
    qcounts = all_queries.value_counts()
    return qcounts[qcounts >= min_query_count].index.to_list()

def prepare_query_data(repos_df, min_query_count=5):
    task_queries = {str(i): query for (i, query) in enumerate(get_queries(repos_df, min_query_count=min_query_count))}

    task_qrels = {
        qid: {str(corpus_id): 1 for corpus_id in get_repos_for_query(task_queries[qid], repos_df).index}
        for qid in task_queries.keys()
    }
    return task_queries, task_qrels



pd.Series(map(len, task_qrels.values())).describe()



import elasticsearch

es_client = elasticsearch.Elasticsearch()
def retrieve_repos_with_es(query, k=50, index="readme", es_client=es_client):
    es_result = es_client.search(index=index, body={"query": {"match": {"txt": query}}}, size=k)
    return [
        hit["_source"]["title"]
        for hit in es_result["hits"]["hits"]
    ]



def get_elasticsearch_results():
    retrieved_repo_tasks = {}

    qcounts = sampled_repos_df["tasks"].explode().value_counts()
    used_queries = [
        query
        for query in sampled_repos_df["tasks"].explode().drop_duplicates()
        if qcounts.loc[query] > 5
    ]
    # [task_queries[qid] for qid in task_queries.keys()]
    
    index="selected_code"
    for query in used_queries:
        retrieved_tasks = sampled_repos_df[sampled_repos_df["repo"].isin(retrieve_repos_with_es(query, index=index))]["tasks"].to_list()
        retrieved_repo_tasks[query] = retrieved_tasks
    
    k = 10
    query_hits = pd.Series({
        query: sum([query in tasks for tasks in retrieved_repo_tasks[query][:k]])
        for query in retrieved_repo_tasks.keys()
    })

def show_elasticsearch_results(qid='10'):
    query = task_queries[qid]
    
    print(query)
    print(query_hits[query], "hits")
    
    for hit in es_client.search(index=index, body={"query": {"match": {"txt": task_queries[qid]}}}, size=k)["hits"]["hits"]:
        print("#" * 100)
        print("#" * 100)
        repo_name = hit["_source"]["title"]
        repo_record = sampled_repos_df[sampled_repos_df["repo"] == repo_name].iloc[0]
        is_hit = query in repo_record["tasks"]
        print(repo_name, "HIT" if is_hit else "NO HIT")
        
        if is_hit:
            print("#" * 100)
            print("#" * 100)
            print(hit['_source']['txt'])


# ## Evaluating with BEIR

# In[13]:


def load_w2v_sentence_transformer(w2v_model_path):
    w2v_layer = sentence_transformers.models.WordEmbeddings.load(w2v_model_path)
    return sentence_transformers.SentenceTransformer(modules=[w2v_layer, sentence_transformers.models.Pooling(200)])



def load_sentence_bert(model_name):
    st_model = SentenceBERT("sentence-transformers/all-mpnet-base-v2")
    st_model.doc_model = sentence_transformers.SentenceTransformer(model_name, trust_remote_code=True)
    st_model.q_model = st_model.doc_model
    return st_model

def get_w2v_retriever(w2v_model_path="../models/rnn_abstract_readme_w2v/0_WordEmbeddings"):
    w2v_model = load_w2v_sentence_transformer(w2v_model_path)
    st_model = SentenceBERT("sentence-transformers/all-mpnet-base-v2")
    st_model.q_model = w2v_model
    st_model.doc_model = w2v_model
    return EvaluateRetrieval(DRES(st_model), score_function="cos_sim")

def get_splade_retriever(splade_model_path = "splade/weights/distilsplade_max", batch_size=128):
    splade_model = DRES(SPLADE(splade_model_path), batch_size=128)
    return EvaluateRetrieval(splade_model, score_function="dot")


def get_bm25_retrievers(corpora):
    def sanitize_index_name(index_name):
        if type(index_name) is str:
            return index_name
        else:
            return "".join(map(str, index_name))
    
    bm25_retrievers = {}
    for corpus_name, corpus in corpora.items():
        model = BM25(index_name=sanitize_index_name(corpus_name))
        retriever = EvaluateRetrieval(model)
        bm25_retrievers[corpus_name] = retriever
    return bm25_retrievers


sentence_transformer_model_names = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    #"nomic-ai/modernbert-embed-base",
    
    #"estrogen/ModernBERT-base-nli-v3"
    #"BAAI/bge-large-en-v1.5",
    #"mixedbread-ai/mxbai-embed-large-v1"
]

def get_sentence_transformer_retriever(model_name="sentence-transformers/all-mpnet-base-v2", batch_size=8):
    model = DRES(load_sentence_bert(model_name), batch_size=batch_size)
    return EvaluateRetrieval(model, score_function="cos_sim")

def get_unicoil_retriever(model_name="castorini/unicoil-msmarco-passage"):
    """
    THERE IS A BUG WITH BEIR THAT MAKES THIS UNUSABLE
    """
    model = SparseSearch(UniCOIL(model_path=model_name), batch_size=32)
    return EvaluateRetrieval(model, score_function="dot")



def get_repomaps_df(repo_names, repomap_path="../output/aider/selected_repo_maps_1024.json"):
    with open(repomap_path) as f:
        repomaps = json.load(f)

    records = []
    for repo in repo_names:
        records.append({"repo_name": repo, "text": repomaps[repo], "representation": "repomap"})
    return pd.DataFrame.from_records(records)




import sentence_transformers

w2v_retriever = get_w2v_retriever()
# change sentence-transformers to 2.7?
sentence_transformer_retrievers = {
    model_name: get_sentence_transformer_retriever(model_name)
    for model_name in sentence_transformer_model_names
}


# In[32]:


bm25_retrievers = get_bm25_retrievers(corpora)


# ## Per query results

# In[33]:


from pydantic import BaseModel
from typing import Dict

class RetrieverInput(BaseModel):
    corpus: Dict[str, dict]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]


class RetrievalEvaluationResults(BaseModel):
    retrieval_results: Dict[str, Dict[str, float]]
    metrics: dict
    model_type: str

    @classmethod
    def from_retriever(cls, retriever, retriever_input, metric_names=["accuracy@k", "hits@k", "r_cap@k", "mrr@k"]):
        retrieval_results = retriever.retrieve(retriever_input.corpus, retriever_input.queries)
        custom_metrics = retriever.evaluate_custom_multi(retriever_input.qrels, retrieval_results, retriever.k_values, metrics=metric_names)
        other_metrics = retriever.evaluate(retriever_input.qrels, retrieval_results, retriever.k_values, ignore_identical_ids=False)
        metrics = custom_metrics | cls.tuple_to_dict(other_metrics)
        try:
            model_type = str(retriever.retriever.model)
        except:
            model_type = "bm25"
        return RetrievalEvaluationResults(metrics=metrics, model_type=model_type, retrieval_results=retrieval_results)


    @classmethod
    def tuple_to_dict(cls, dicts):
        merged_dict = {}
        for d in dicts:
            merged_dict = d | merged_dict
        return merged_dict


# In[34]:


retriever_inputs = {
    corpus_name: RetrieverInput(corpus=corpus, queries=task_queries, qrels=task_qrels)
    for (corpus_name, corpus) in corpora.items()
}


# In[35]:


from github_search.evaluation.beir_evaluation import PerQueryIREvaluator


# In[36]:


per_query_evaluator = PerQueryIREvaluator(k_values=[1, 5, 10, 25])


# In[37]:


retriever_inputs = {
    corpus_name: RetrieverInput(corpus=corpus, queries=task_queries, qrels=task_qrels)
    for (corpus_name, corpus) in corpora.items()
}


# In[38]:


# In[39]:


named_retrievers = {
    corpus_name: [
        ("bm25", bm25_retrievers[corpus_name]),
        ("word2vec", w2v_retriever),
    ] + list(sentence_transformer_retrievers.items())
    for corpus_name in retriever_inputs.keys()
}


# In[40]:


retriever_inputs.keys()


# In[ ]:

per_query_results = {
    (corpus_name, retriever_name): per_query_evaluator.get_scores(retriever=retriever, ir_data=retriever_inputs[corpus_name])
    for corpus_name in retriever_inputs.keys()
    for (retriever_name, retriever) in named_retrievers[corpus_name]
}


# In[ ]:


raw_per_query_results_df = pd.concat([
    df.assign(retriever=[retriever_name]*len(df)).assign(corpus=[corpus_name]*len(df))
    for ((corpus_name, retriever_name), df) in per_query_results.items()
])


# In[ ]:


per_query_results_df = raw_per_query_results_df.assign(
    corpus=raw_per_query_results_df["corpus"].apply(lambda cname: cname if type(cname) is str else cname[0]),
    generation=raw_per_query_results_df["corpus"].apply(lambda cname: 0 if type(cname) is str else cname[1])
)


# In[44]:


per_query_results_df = (
    per_query_results_df
        .groupby(["query", "retriever", "corpus"]).agg("mean").drop(columns=["generation"])
        .reset_index()
)


# In[45]:


per_query_results_df


# In[46]:


per_query_results_df.to_csv("../results/per_query_ir_results.csv", index=False)


# In[47]:


(per_query_results_df
    .drop(columns=["query"])
    .groupby(["corpus", "retriever"])
    .agg("mean").reset_index(drop=False)
    .sort_values("Accuracy@10")
)[["corpus", "retriever", "Hits@10", "Accuracy@10"]]


# ## Aggregated results

# In[48]:


for corpus_name in corpora.keys():
    try:
        RetrievalEvaluationResults.from_retriever(bm25_retrievers[corpus_name], retriever_inputs[corpus_name])
    except:
        print(corpus_name)


# In[52]:


bm25_results = {
    corpus_name: RetrievalEvaluationResults.from_retriever(bm25_retrievers[corpus_name], retriever_inputs[corpus_name])
    for corpus_name in corpora.keys()
}


# splade_results = {
#     corpus_name: RetrievalEvaluationResults.from_retriever(splade_retriever, retriever_inputs[corpus_name])
#     for corpus_name in corpora.keys()
# }

# In[53]:


word2vec_results = {
    corpus_name: RetrievalEvaluationResults.from_retriever(w2v_retriever, retriever_inputs[corpus_name])
    for corpus_name in corpora.keys()
}


# In[54]:


sentence_transformer_results = {
    (corpus_name, model_name.split("/")[1]): RetrievalEvaluationResults.from_retriever(sentence_transformer_retrievers[model_name], retriever_inputs[corpus_name])
    for corpus_name in corpora.keys()
    for model_name in sentence_transformer_model_names
}


# In[55]:


bm25_metrics = [
    {"corpus": corpus_name, "retriever": "bm25", **bm25_results[corpus_name].metrics}
    for corpus_name in corpora.keys()
]


# In[56]:


word2vec_metrics = [
    {"corpus": corpus_name, "retriever": "Python code word2vec", **word2vec_results[corpus_name].metrics}
    for corpus_name in corpora.keys()
]


# In[57]:


#splade_metrics = [
#    {"corpus": corpus_name, "retriever": "splade", **splade_results[corpus_name].metrics}
#     for corpus_name in corpora.keys()
#]
 
sentence_transformer_metrics = [
    {"corpus": corpus_name, "retriever": f"{model_name} (sentence_transformer)", **sentence_transformer_results[(corpus_name, model_name)].metrics}
    for (corpus_name, model_name) in sentence_transformer_results.keys()
]

all_metrics_df = pd.DataFrame.from_records(bm25_metrics + word2vec_metrics +  sentence_transformer_metrics).sort_values("Hits@10", ascending=False)


# In[58]:


all_metrics_df.shape


# In[59]:


all_metrics_df[["corpus", "retriever", "Accuracy@10"]]


# In[63]:


all_metrics_df.columns


# In[60]:


model_name = "qwen2.5:7b-instruct"


# In[61]:


all_metrics_df.to_csv(f"../output/code2doc/beir_results_{model_name}.csv", index=False)


# In[62]:


#all_metrics_df.to_csv(f"../output/code2doc/beir_results_with_modernbert_{model_name}.csv", index=False)


# ## Results
# 
# By default we will use min_task_count=10 (as we used originally)
# 
# We can switch to smaller task counts like 5 to incorporate the fact that we use sample of repos

# In[ ]:


metric_df_cols = ["corpus", "retriever", "Accuracy@10", "Hits@10", "R_cap@10", "P@1", "P@5", "P@10"]


# In[ ]:


all_metrics_df[metric_df_cols]


# In[ ]:


all_metrics_df[metric_df_cols].sort_values("Accuracy@10", ascending=False)


# In[ ]:


all_metrics_df.groupby("corpus").apply(lambda df: df.sort_values("Accuracy@10", ascending=False).iloc[0])[metric_df_cols].sort_values("Accuracy@10", ascending=False)


# In[ ]:


all_metrics_df.groupby("retriever").apply(lambda df: df.sort_values("Accuracy@10", ascending=False).iloc[0])[metric_df_cols].sort_values("Accuracy@10", ascending=False)


# In[ ]:


all_metrics_df[all_metrics_df["retriever"] == "bm25"][metric_df_cols]


# In[ ]:


len(task_queries)


# In[ ]:


# task count = 5


# In[ ]:


all_metrics_df[["corpus", "retriever", "Accuracy@10"]].sort_values("Accuracy@10", ascending=False)


# In[ ]:


# task count = 10


# In[ ]:


all_metrics_df[["corpus", "retriever", "Accuracy@10"]].sort_values("Accuracy@10", ascending=False)


# In[ ]:


all_metrics_df.groupby("retriever")["Accuracy@10"].agg("mean").sort_values()


# In[ ]:


all_metrics_df.groupby("retriever")["Accuracy@10"].agg("mean").sort_values()


# In[ ]:


all_metrics_df.groupby("retriever")["Accuracy@10"].agg("mean").sort_values()


# In[ ]:


all_metrics_df.groupby("corpus")["Accuracy@10"].agg("mean").sort_values()


# In[ ]:


sampled_repos_df["tasks"].explode().value_counts().loc[list(task_queries.values())]


# In[ ]:


all_metrics_df[["corpus", "retriever", "Accuracy@10"]].sort_values("Accuracy@10", ascending=False)


# ## Does combining rationale with generated readme help?
# 
# It seems that the best sentence transformer retrievers can only get worse when using any other information!

# In[ ]:


sentence_transformer_results.keys()


# In[ ]:


st_generated_readme_results= sentence_transformer_results[('generated_readme', 'all-mpnet-base-v2')].retrieval_results
st_rationale_results = sentence_transformer_results[('generated_rationale', 'all-mpnet-base-v2')].retrieval_results
bm25_generated_readme_results = bm25_results["generated_readme"].retrieval_results
st_context_results = sentence_transformer_results[('generation_context', 'all-mpnet-base-v2')].retrieval_results


# In[ ]:


len(list(bm25_generated_readme_results.keys()))


# In[ ]:


len(list(st_generated_readme_results.keys()))


# In[ ]:


def merge_qrels(qrels1, qrels2):
    merged_qrels = {}
    for k in qrels1.keys():
        tmp_rel = dict()
        for rel_k in set(qrels1[k].keys()).union(qrels2[k]):
            tmp_rel[rel_k] = qrels1[k].get(rel_k, 0) +  qrels2[k].get(rel_k, 0)
        merged_qrels[k] = tmp_rel
    return merged_qrels


# In[ ]:


st_generation_results = merge_qrels(bm25_generated_readme_results, st_generated_readme_results)


# In[ ]:


EvaluateRetrieval().evaluate_custom(task_qrels, st_generation_results, metric="acc", k_values=[1,5,10])


# In[ ]:


EvaluateRetrieval().evaluate_custom(task_qrels, st_generated_readme_results, metric="acc", k_values=[1,5,10])


# In[ ]:


EvaluateRetrieval().evaluate_custom(task_qrels, st_rationale_results, metric="acc", k_values=[1,5,10])


# In[ ]:


all_metrics_df[all_metrics_df["retriever"] == "bm25"][["corpus", "retriever", "Accuracy@10"]].sort_values("Accuracy@10")


# In[ ]:


Splitting does not make much sense as the most of generated data is under the sentence-transformer context length (384 tokens)


# In[ ]:


def split_corpus_by_lengths(corpus, chunk_length):
    splitted_corpora = [dict() for _ in range(n_splits)]
    for c_id in corpus.keys():
        text = corpus[c_id]["text"]
        chunk_length =  len(text) // n_splits
        for i in range(0, n_splits):
            splitted_corpora[i] = text[i*chunk_length:(i+1)*chunk_length]
        


# In[ ]:


class MultiTextEvaluator(BaseModel):
    """
    Evaluate a dataframe that has multiple texts for each query (multiple generation experiments)
    iteration_col says which experiment it was
    """
    iteration_col: str
    text_cols: List[str]
    k_values: List[int] = [1,5,10,25]

    def get_ir_datas(self, df):
        for iter in df[self.iteration_col].unique():
            ir_data = load_ir_data(df[df[self.iteration_col] == iter], self.text_cols)
            yield (iter, ir_data)

    def evaluate(self, df, retriever):
        ir_datas = dict(self.get_ir_datas(df))
        dfs = []
        for iter, ir_data in ir_datas.items():
            per_query_evaluator = PerQueryIREvaluator(k_values=self.k_values)
            df = per_query_evaluator.get_scores(ir_data, retriever)
            df[self.iteration_col] = iter
            dfs.append(df)
        metrics_df = pd.concat(dfs)
        metrics_df["query"] = metrics_df.index
        return metrics_df

