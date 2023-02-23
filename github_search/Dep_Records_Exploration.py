#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from github_search import utils
import sentence_transformers


# In[2]:


get_ipython().run_cell_magic('time', '', 'dependency_df = pd.read_feather("../../output/dependency_records.feather")\n')


# In[3]:


dependency_df = dependency_df[dependency_df['source'] != "<ROOT>"]


# In[4]:


dependency_without_leaf_functions_df = dependency_df[dependency_df['edge_type'] != "function-function"]


# In[5]:


dependency_df.shape


# In[6]:


dependency_without_leaf_functions_df.shape


# In[7]:


repos = dependency_without_leaf_functions_df['repo'].drop_duplicates()


# In[8]:


from operator import itemgetter


# In[9]:


repo = repos.iloc[0]


# In[10]:


def get_repo_dependencies(dependency_df, repo):
    repo_df = dependency_df[dependency_df['repo'] == repo]

    return repo_df


# In[11]:


dependency_df[dependency_df['repo'] == repo]


# In[12]:


repo_df = get_repo_dependencies(dependency_df, repos.iloc[10])


# In[13]:


paperswithcode_df = utils.load_paperswithcode_df("../../data/paperswithcode_with_tasks.csv")


# In[14]:


import igraph


# In[15]:


def make_repo_df_graph(repo_df):
    graph = igraph.Graph()
    vertices = pd.concat([repo_df['source'], repo_df['destination']]).unique()
    graph.add_vertices(vertices, attributes={"text": vertices})
    graph.add_edges(repo_df[['source', 'destination']].to_records(index=False))
    return graph


# In[16]:


from github_search.graphs import data_preparation
import tqdm


# In[17]:


repo_task_mapping = {
    repo: tasks
    for (repo, tasks) in tqdm.notebook.tqdm(
        paperswithcode_df[paperswithcode_df['repo'].isin(set(repos))][['repo', 'tasks']].to_records(index=False)
    )
}


# In[18]:


import tqdm


# In[19]:


repo_keyed_dependency_df = dependency_without_leaf_functions_df.groupby("repo")


# In[20]:


def get_repo_df(dependency_gb, repo):
    return dependency_gb.get_group(repo)


# In[21]:


repo = repos.iloc[10]


# In[22]:


get_ipython().run_cell_magic('time', '', 'get_repo_df(repo_keyed_dependency_df, repo)\n')


# In[23]:


make_repo_df_graph(get_repo_df(repo_keyed_dependency_df, repo))


# In[24]:


get_repo_df(repo_keyed_dependency_df, repo)


# In[25]:


task_areas_df = pd.read_csv("../../data/paperswithcode_tasks.csv")


# In[26]:


task_areas_df


# In[27]:


task_metadata_records_ = paperswithcode_df[['repo', 'least_common_task']].merge(task_areas_df, left_on="least_common_task", right_on="task").drop_duplicates(subset="repo").to_dict(orient="records")
task_metadata_records = {}
for rec in task_metadata_records_:
    repo = rec["repo"]
    task_metadata_records[repo] = {k: rec[k] for k in ['area', 'least_common_task']} 


# In[28]:


len([repo for repo in repo_task_mapping.keys() if repo in task_metadata_records.keys()])


# In[29]:


repos_with_metadata = [repo for repo in repo_task_mapping.keys() if repo in task_metadata_records.keys()]


# In[30]:


r = repos_with_metadata[0]


# In[31]:


from findkit.feature_extractor import FastAITextFeatureExtractor 
from fastai.text import all as text


# In[32]:


fastai_learner = text.load_learner("learn.pkl")


# In[34]:


fastai_extractor = FastAITextFeatureExtractor.build_from_learner(fastai_learner, max_length=48)


# In[35]:


texts = dependency_df.iloc[:100000]["source"]


# In[36]:


texts


# In[39]:


features = fastai_extractor.extract_features(texts.to_list())


# In[40]:


features.shape


# In[41]:


type(fastai_learner.model)


# In[42]:


fastai_learner.dls.tfms


# In[43]:


def make_graph_record(repo, encoding_fn):
    metadata = {"tasks": repo_task_mapping[repo], **task_metadata_records.get(repo)}
    return data_preparation.get_graph_data(make_repo_df_graph(get_repo_df(repo_keyed_dependency_df, repo)), repo, metadata, encoding_fn)


# In[47]:


import torch
import toolz


# In[45]:


graph_list = []

from github_search import utils

def get_graph_data_generator(repos, encoding_fn):
    pbar = tqdm.notebook.tqdm(repos)
    for repo in pbar:
        try:
            graph_data = make_graph_record(repo, encoding_fn)
            yield graph_data
        except RuntimeError:
            print(f"failed for repo {repo}")


# In[104]:


import h5py


# In[109]:


repo_keyed_dependency_df.get_group('000Justin000/gnn-residual-correlation')


# In[ ]:





# In[ ]:


get_graph_data_generator(repos_with_metadata, toolz.partial(fastai_extractor.extract_features, show_progbar=False))


# In[50]:


from typing import List
import torch_geometric

def add_graph_to_hdf_groups(g: torch_geometric.data.Data, x_group: h5py.Group, edge_index_group: h5py.Group):
    if not g.graph_name in x_group.keys():
        d = x_group.create_dataset(g.graph_name, data=g.x.numpy())
        d.attrs["tasks"] = g.tasks
        d.attrs["least_common_task"] = g.least_common_task
        d.attrs["area"] = g.area
        edge_index_group.create_dataset(g.graph_name, data=g.edge_index.numpy())


def add_graph_data_list_to_file(h5py_file: h5py.File, graph_data_list: List[torch_geometric.data.Data]):
    x_gp = h5py_file.create_group("x")
    edge_index_gp = h5py_file.create_group("edge_index")
    for g in tqdm.tqdm(graph_data_list):
        add_graph_to_hdf_groups(g, x_gp, edge_index_gp)


# In[51]:


encoding_fn = toolz.partial(fastai_extractor.extract_features, show_progbar=False)

with h5py.File("graph_records.h5", "w") as f:
    add_graph_data_list_to_file(f,  get_graph_data_generator(repos_with_metadata, encoding_fn))


# In[13]:


import h5py


# In[64]:


f = h5py.File("graph_records.h5", "r") 


# In[63]:


f.close()


# In[23]:


gp = f['x']


# In[24]:


keys = [
    (key, subkey)
    for key in gp.keys()
    for subkey in gp[key].keys()
]


# In[30]:


k = '/'.join(keys[0])


# In[31]:


gp[k]


# In[ ]:





# In[6]:


gp = f.get("x")["008karan"]


# In[9]:


list(f["edge_index"]["008karan"]["SincNet_demo"])[0]


# In[5]:


list(f["x"]["008karan"]["SincNet_demo"])


# In[12]:


import torch_geometric.data
import h5py 


# In[ ]:


h


# In[148]:


import torch_geometric.data as ptg_data
import h5py
import torch


class HDF5Dataset(ptg_data.Dataset):
    
    def __init__(self, h5file, meta_keys, hierarchical_keys=True):
        self.h5file = h5file
        self.hierarchical_keys = hierarchical_keys
        self.keys = self._get_keys(h5file, hierarchical_keys)
        self.meta_keys = meta_keys 
        self.transform = None
        
    def len(self):
        return len(self.keys)
            
    def indices(self):
        return self.keys
        
    def get(self, idx):
        graph_record_data = self._get_graph_record_data(idx)
        return self._make_graph_record(graph_record_data)
    
    def _get_graph_record_data(self, key):
        graph_record_data = {
            "x": torch.Tensor(self.h5file["x"][key][()]),
            "edge_index": torch.LongTensor(self.h5file["edge_index"][key][()]),
        }
        for meta_key in self.meta_keys:
            graph_record_data[meta_key] = self.h5file["x"][key].attrs[meta_key]
        return graph_record_data
    
    def _make_graph_record(self, data_dict):
        return ptg_data.Data(**data_dict)
    
    @classmethod
    def _get_keys(cls, h5file, hierarchical_keys):
        if hierarchical_keys:
            return [
                "/".join([key, subkey])
                for key in h5file["x"].keys()
                for subkey in h5file["x"][key].keys()
            ]
        else:
            return list(h5file["x"].keys())


# In[1]:


import h5py


# In[2]:


f = h5py.File("graph_records.h5", "r") 


# In[3]:


from github_search.graphs import datasets


# In[4]:


keys = datasets.HDF5Dataset._get_keys(f, True)


# In[5]:


list(f["x"][keys[0]].attrs)


# In[10]:


dset = datasets.HDF5Dataset(f, ["area", "tasks", "least_common_task"])


# In[7]:


f["x"][keys[0]]


# In[150]:


k = HDF5Dataset._get_keys(f, False)[0]


# In[151]:


f["x"][k].keys()


# In[152]:


ds = HDF5Dataset(f, "area")


# In[154]:


ds[0]


# In[155]:


ds = f["x"][ds.keys[0]]


# In[127]:


ds.value


# In[ ]:





# In[51]:


ds = HDF5Dataset(f, "area")


# In[57]:


ds.get(5)


# In[ ]:





# In[ ]:


class GraphHDF5Dataset()


# In[ ]:


len(x_grp)


# In[ ]:


with h5py.File("/tmp/graph_data.h5", "w") as f:
    add_graph_data_list_to_file(f, graph_list)


# In[ ]:


data_preparation.get_graph_data(graph, repo, repo_task_mapping, rnn.encode)


# In[ ]:


graph.vs[2]


# In[ ]:


set(repo_df['source'])


# In[ ]:


repos.isin(paperswithcode_df['repo']).mean()


# In[ ]:


paperswithcode_df


# In[ ]:


pd.concat([file_dependencies, function_dependencies, function_function_dependencies]).shape


# In[ ]:


function_function_dependencies


# In[ ]:


function_dependencies


# In[ ]:


function_function_dependencies


# In[ ]:





# In[ ]:




