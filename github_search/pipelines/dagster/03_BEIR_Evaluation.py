#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pytrec_eval
import pandas as pd


# In[ ]:


import torch


# In[ ]:


pd.set_option('max_colwidth', 128)

from pathlib import Path
import json


# In[ ]:


from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from github_search.beir_evaluation import EvaluateRetrievalCustom as EvaluateRetrieval, CorpusDataLoader
from beir.retrieval.search.lexical import BM25Search as BM25


from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models import SPLADE, SentenceBERT, UniCOIL
from beir.retrieval.search.sparse import SparseSearch

#from github_search.ir.evaluate_bm25 import load_ir_data, load_generation_metrics_df, RetrievalConfig, get_retriever
#from github_search.pipelines.get_zenml_results import ArtifactLoader


# In[ ]:


## Dependency and librarian signatures


# In[ ]:


librarian_signatures_df = pd.read_parquet("/home/kuba/Projects/uhackathons/fastrag_util/data/librarian_signatures.parquet")


# In[7]:


## load import2vec


# In[9]:


import sentence_transformers


# In[10]:


from typing import Union
import ast
from pydantic import BaseModel


data_path = Path("../output").expanduser()

small_sample_loader = CorpusDataLoader(
    repos_df_path= data_path / "code2doc/sample2k/sampled_repos.jsonl",
    generated_readmes_df_path=Path("~/Projects").expanduser() / "torch_example/phoenix/sample_2k/trace_dataset-353a22a7-b529-4d9d-a4ec-75f442aa3eb7.parquet",
    code_df_path=data_path / "code" / "python_files_with_selected_code.feather"
)


# In[11]:


class ExperimentParams:
    sampled_repos_per_task = 20
    min_repos_per_task = 10


# In[12]:


#sampled_repos_df, sampled_generated_readmes_df, sample_python_code_df = small_sample_loader.load_corpus_dfs(librarian_signatures_df["repo"])


# In[13]:


def filter_dfs_by_cols_in(dfs, col_values, colnames=["repo", "repo_name"]):
    out_dfs = []
    for df in dfs:
        df_cols = [c for c in colnames if c in df.columns]
        col = df_cols[0]
        filtered_df = df[df[col].isin(col_values)]
        out_dfs.append(filtered_df)
    return out_dfs


def align_dfs(dfs, colname="repo"):
    df0 = dfs[0].reset_index()
    df_index = df0[colname]
    new_dfs = [
        df.set_index(colname).loc[df_index].reset_index()
        for df in dfs[1:]
    ]
    return [df0] + new_dfs


# In[ ]:





# In[14]:


bigger_sample_path = f"../output/code2doc/sample_per_task_5_repos/sampled_repos{ExperimentParams.sampled_repos_per_task}.jsonl"
sample_path = bigger_sample_path#"../output/code2doc/sample_small/sampled_repos_min10.jsonl"
sampled_repos_df = pd.read_json(sample_path, orient="records", lines=True)
sample_python_code_df = pd.read_feather(Path(data_path) / "code" / "python_files_with_selected_code.feather")


# In[15]:


sampled_repos_df.shape


# In[16]:


repos_with_all_data = (
    set(sampled_repos_df["repo"]) &
    set(librarian_signatures_df["repo"]) &
    set(sample_python_code_df["repo_name"])
)


# In[17]:


len(repos_with_all_data)


# In[18]:


#librarian_signatures_df = librarian_signatures_df[librarian_signatures_df["generation"] == 0]


# Select only repos with signatures that were in sample

# In[19]:


sampled_repos_df, sample_python_code_df, sampled_librarian_signatures_df = filter_dfs_by_cols_in([sampled_repos_df, sample_python_code_df, librarian_signatures_df], repos_with_all_data)
sampled_repos_df, sampled_librarian_signatures_df = align_dfs([sampled_repos_df, sampled_librarian_signatures_df])


# ## Sample with generated READMEs

# In[20]:


model_name = "codellama_repomaps"
sample_prefix = "sample_per_task_5_repos"

sample_loader = CorpusDataLoader(
    repos_df_path= data_path / f"code2doc/{sample_prefix}/sampled_repos5.jsonl",
    generated_readmes_df_path=data_path / f"code2doc/{sample_prefix}/{model_name}_generated_readmes5.jsonl",
    code_df_path=data_path / "code" / "python_files_with_selected_code.feather"
)


# In[21]:


sample_loader.generated_readmes_df_path


# In[22]:


sampled_repos_df, sampled_generated_readmes_df, sample_python_code_df = sample_loader.load_corpus_dfs(librarian_signatures_df["repo"])


# In[23]:


repos_with_all_data = set(sampled_repos_df["repo"]).intersection(librarian_signatures_df["repo"])


# In[24]:


sampled_repos_df, sample_python_code_df, sampled_librarian_signatures_df = filter_dfs_by_cols_in([sampled_repos_df, sample_python_code_df, librarian_signatures_df], repos_with_all_data)
sampled_repos_df, sampled_librarian_signatures_df = align_dfs([sampled_repos_df, sampled_librarian_signatures_df])#[sampled_librarian_signatures_df["generation"] == 0]])


# ## Example BEIR dataset

# In[25]:


import os
import pathlib

dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path("..").parent.absolute(), "datasets")
beir_data_path = util.download_and_unzip(url, out_dir)


# In[26]:


_corpus, _queries, _qrels = GenericDataLoader(beir_data_path).load(split="test")


# In[27]:


print(_corpus['4983'].keys())
_corpus['4983']


# ## Data preparation

# In[28]:


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


def prepare_readme_corpus(repos_df):
    return {str(i): {"text": row["readme"], "title": row["repo"], 'tasks': row['tasks']} for (i, row) in repos_df.iterrows()}


def prepare_generated_readme_corpus(repos_df, generated_readmes_df, columns=["answer"]):
    generated_readmes_df = generated_readmes_df.set_index("repo_name").loc[repos_df["repo"]].reset_index()
    return {str(i): {"text": "\n".join(row[columns]), "title": row["repo_name"]} for (i, row) in generated_readmes_df.iterrows()}

    
def prepare_code_corpus(repos_df, selected_python_code_df):
    per_repo_code_df = selected_python_code_df.groupby("repo_name").apply(lambda df: "\n\n".join(df["selected_code"].fillna("")))
    per_repo_code_df = per_repo_code_df.loc[repos_df["repo"]].reset_index()
    return {str(i): {"text": row[0], "title": row["repo_name"]} for (i, row) in per_repo_code_df.iterrows()}


# THIS IS FOR ONE GENERATION ONLY NOW
def prepare_librarian_corpora(repos_df, sampled_librarian_signatures_df):
    columns = ["dependency_signature", "repository_signature", "generated_tasks"]
    sampled_librarian_signatures_df = sampled_librarian_signatures_df.set_index("repo").loc[repos_df["repo"]].reset_index()
    return {
        (column, g): {
            str(i): {
                "text": row[column], "title": row["repo"]
            } for (i, row) in sampled_librarian_signatures_df[sampled_librarian_signatures_df["generation"] == g].reset_index(drop=True)[["repo", column]].iterrows()
        } 
        for column in columns
        for g in sampled_librarian_signatures_df["generation"].unique()
    }


def prepare_basic_corpora(repos_df, selected_python_code_df):
    readme_corpus = prepare_readme_corpus(repos_df)
    selected_python_code_corpus = prepare_code_corpus(repos_df, selected_python_code_df)
    return {"readme": readme_corpus, "selected_code": selected_python_code_corpus}


def prepare_corpora(repos_df, generated_readmes_df, selected_python_code_df):
    basic_corpora = prepare_basic_corpora(repos_df, selected_python_code_df)
    readme_corpus = basic_corpora["readme"]
    selected_python_code_corpus = basic_corpora["selected_code"]
    generated_readme_corpus = prepare_generated_readme_corpus(repos_df, sampled_generated_readmes_df)
    generated_rationale_corpus = prepare_generated_readme_corpus(repos_df, sampled_generated_readmes_df, columns=["rationale"])
    generated_readme_rationale_corpus = prepare_generated_readme_corpus(repos_df, sampled_generated_readmes_df, columns=["answer", "rationale"])
    generated_readme_context_corpus = prepare_generated_readme_corpus(repos_df, sampled_generated_readmes_df, columns=["context_history"])
    

    assert len(readme_corpus) == len(generated_readme_corpus)
    assert len(selected_python_code_corpus) == len(readme_corpus)
    
    for k in readme_corpus.keys():
        assert readme_corpus[k]['title'] == generated_readme_corpus[k]['title'], str((readme_corpus[k]['title'], generated_readme_corpus[k]['title']))
        assert readme_corpus[k]['title'] == selected_python_code_corpus[k]['title']
    return {
        "readme": readme_corpus,
        "generated_readme": generated_readme_corpus,
        "selected_code": selected_python_code_corpus,
        "generated_rationale": generated_rationale_corpus,
        "generation_context": generated_readme_context_corpus,
    }


# In[29]:


sampled_repos_df


# In[30]:


task_queries, task_qrels = prepare_query_data(sampled_repos_df, min_query_count=ExperimentParams.min_repos_per_task)


# In[31]:


pd.Series(task_qrels).apply(len).describe()


# In[32]:


pd.Series([len(qrl) for qrl in task_qrels.values()]).describe()


# In[33]:


#corpora = prepare_basic_corpora(sampled_repos_df, sample_python_code_df) |  #
corpora =  prepare_corpora(sampled_repos_df, sampled_generated_readmes_df, sample_python_code_df) | prepare_librarian_corpora(sampled_repos_df, sampled_librarian_signatures_df)


# In[34]:


[len(corpora[cname].keys()) for cname in corpora.keys()]


# In[35]:


for cid in corpora["readme"].keys():
    assert corpora["readme"][cid]["title"] == corpora["readme"][cid]["title"], f"no match at {cid}"
    #assert corpora["readme"][cid]["title"] == corpora[("dependency_signature", 0)][cid]["title"], f"no match at {cid}"


# In[36]:


## Checking elasticsearch


# In[37]:


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

# In[38]:


def load_w2v_sentence_transformer(w2v_model_path):
    w2v_layer = sentence_transformers.models.WordEmbeddings.load(w2v_model_path)
    return sentence_transformers.SentenceTransformer(modules=[w2v_layer, sentence_transformers.models.Pooling(200)])

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
    "flax-sentence-embeddings/st-codesearch-distilroberta-base"
]

def get_sentence_transformer_retriever(model_name="sentence-transformers/all-mpnet-base-v2", batch_size=256):
    model = DRES(SentenceBERT(model_name), batch_size=batch_size)
    return EvaluateRetrieval(model, score_function="cos_sim")

def get_unicoil_retriever(model_name="castorini/unicoil-msmarco-passage"):
    """
    THERE IS A BUG WITH BEIR THAT MAKES THIS UNUSABLE
    """
    model = SparseSearch(UniCOIL(model_path=model_name), batch_size=32)
    return EvaluateRetrieval(model, score_function="dot")


# In[39]:


corpora.keys()


# In[43]:


bm25_retrievers = get_bm25_retrievers(corpora)


# In[44]:


#splade_retriever = get_splade_retriever()
sentence_transformer_retrievers = {
    model_name: get_sentence_transformer_retriever(model_name)
    for model_name in sentence_transformer_model_names
}


# In[45]:


for dres in sentence_transformer_retrievers.values():
    print(dres.retriever.model.doc_model.device)# = dres.retriever.model.doc_model.to("cuda")


# In[46]:


sentence_transformer_retrievers['sentence-transformers/all-mpnet-base-v2'].retriever.model.doc_model.device


# In[47]:


w2v_retriever = get_w2v_retriever()


# ## Per query results

# In[48]:


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
    def from_retriever(cls, retriever, retriever_input, metric_names=["accuracy@k", "hits@k", "r_cap@k"]):
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


# In[49]:


retriever_inputs = {
    corpus_name: RetrieverInput(corpus=corpus, queries=task_queries, qrels=task_qrels)
    for (corpus_name, corpus) in corpora.items()
}


# In[50]:


from github_search.ir.beir_utils import PerQueryIREvaluator


# In[51]:


per_query_evaluator = PerQueryIREvaluator(k_values=[1, 5, 10, 25])


# In[52]:


retriever_inputs = {
    corpus_name: RetrieverInput(corpus=corpus, queries=task_queries, qrels=task_qrels)
    for (corpus_name, corpus) in corpora.items()
}


# In[53]:


retriever_inputs.keys()


# In[54]:


named_retrievers = {
    corpus_name: [
        ("bm25", bm25_retrievers[corpus_name]),
        ("word2vec", w2v_retriever),
    ] + list(sentence_transformer_retrievers.items())
    for corpus_name in retriever_inputs.keys()
}


# In[63]:


get_ipython().run_cell_magic('time', '', 'per_query_results = {\n    (corpus_name, retriever_name): per_query_evaluator.get_scores(retriever=retriever, ir_data=retriever_inputs[corpus_name])\n    for corpus_name in retriever_inputs.keys()\n    for (retriever_name, retriever) in named_retrievers[corpus_name]\n}\n')


# In[64]:


per_query_results_df = pd.concat([
    df.assign(retriever=[retriever_name]*len(df)).assign(corpus=[corpus_name]*len(df))
    for ((corpus_name, retriever_name), df) in per_query_results.items()
])


# In[65]:


per_query_results_df = per_query_results_df.assign(
    corpus=per_query_results_df["corpus"].apply(lambda cname: cname if type(cname) is str else cname[0]),
    generation=per_query_results_df["corpus"].apply(lambda cname: 0 if type(cname) is str else cname[1])
)


# In[66]:


per_query_results_df = (
    per_query_results_df
        .groupby(["query", "retriever", "corpus"]).agg("mean").drop(columns=["generation"])
        .reset_index()
)


# In[67]:


per_query_results_df.to_csv("../results/per_query_ir_results.csv", index=False)


# In[68]:


(per_query_results_df
    .drop(columns=["query"])
    .groupby(["corpus", "retriever"])
    .agg("mean").reset_index(drop=False)
    .sort_values("Accuracy@10")
)[["corpus", "retriever", "Hits@10", "Accuracy@10"]]


# ## Aggregated results

# In[69]:


bm25_results = {
    corpus_name: RetrievalEvaluationResults.from_retriever(bm25_retrievers[corpus_name], retriever_inputs[corpus_name])
    for corpus_name in corpora.keys()
}


# In[62]:


splade_results = {
    corpus_name: RetrievalEvaluationResults.from_retriever(splade_retriever, retriever_inputs[corpus_name])
    for corpus_name in corpora.keys()
}


# In[70]:


word2vec_results = {
    corpus_name: RetrievalEvaluationResults.from_retriever(w2v_retriever, retriever_inputs[corpus_name])
    for corpus_name in corpora.keys()
}


# In[71]:


sentence_transformer_results = {
    (corpus_name, model_name.split("/")[1]): RetrievalEvaluationResults.from_retriever(sentence_transformer_retrievers[model_name], retriever_inputs[corpus_name])
    for corpus_name in corpora.keys()
    for model_name in sentence_transformer_model_names
}


# In[72]:


bm25_metrics = [
    {"corpus": corpus_name, "retriever": "bm25", **bm25_results[corpus_name].metrics}
    for corpus_name in corpora.keys()
]


# In[73]:


word2vec_metrics = [
    {"corpus": corpus_name, "retriever": "Python code word2vec", **word2vec_results[corpus_name].metrics}
    for corpus_name in corpora.keys()
]


# In[76]:


#splade_metrics = [
#    {"corpus": corpus_name, "retriever": "splade", **splade_results[corpus_name].metrics}
#     for corpus_name in corpora.keys()
#]
 
sentence_transformer_metrics = [
    {"corpus": corpus_name, "retriever": f"{model_name} (sentence_transformer)", **sentence_transformer_results[(corpus_name, model_name)].metrics}
    for (corpus_name, model_name) in sentence_transformer_results.keys()
]

all_metrics_df = pd.DataFrame.from_records(bm25_metrics + word2vec_metrics +  sentence_transformer_metrics).sort_values("Hits@10", ascending=False)


# In[77]:


f"../output/code2doc/{sample_prefix}/beir_results_{model_name}.csv"


# In[78]:


all_metrics_df.to_csv(f"../output/code2doc/{sample_prefix}/beir_results_{model_name}.csv", index=False)


# ## Results
# 
# By default we will use min_task_count=10 (as we used originally)
# 
# We can switch to smaller task counts like 5 to incorporate the fact that we use sample of repos

# In[ ]:


metric_df_cols = ["corpus", "retriever", "Accuracy@10", "Hits@10", "R_cap@10", "NDCG@10"]


# In[ ]:


all_metrics_df[metric_df_cols]


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


st_generation_results['0']


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

