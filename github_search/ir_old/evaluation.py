from typing import Generic, List, Optional, Protocol, TypeVar

import sentence_transformers
import torch
from github_search import utils
from github_search.ir import evaluator, ir_utils

CorpusType = TypeVar("CorpusType")
QueryType = TypeVar("QueryType")


