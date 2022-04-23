import os

import pandas as pd
import tqdm
from mlutil.sentence_rnn import SentenceRNN
from sentence_transformers import InputExample, SentenceTransformer, evaluation, models

RNN_MODEL_TYPES = ["sru", "lstm"]


def get_sbert_ir_dicts(input_df, query_col="tasks", doc_col="readme"):
    df_copy = input_df.copy()
    queries = df_copy[query_col].explode().drop_duplicates()
    queries = pd.DataFrame(
        {"query": queries, "query_id": [str(s) for s in queries.index]}
    )
    queries.index = queries["query_id"]
    corpus = df_copy[doc_col]
    corpus.index = [str(i) for i in corpus.index]
    df_copy["doc_id"] = corpus.index
    relevant_docs_str = df_copy[["doc_id", "tasks", doc_col]].explode(column="tasks")
    relevant_docs = (
        relevant_docs_str.merge(queries, left_on="tasks", right_on="query")[
            ["doc_id", "query_id"]
        ]
        .groupby("query_id")
        .apply(lambda df: set(df["doc_id"]))
        .to_dict()
    )
    return queries["query"].to_dict(), corpus.to_dict(), relevant_docs


def get_ir_metrics(path):
    metrics_df = pd.read_csv(
        os.path.join(path, "Information-Retrieval_evaluation_results.csv")
    )
    return metrics_df[[col for col in metrics_df if "cos" in col]]


def get_ir_evaluator(paperswithcode_df, query_col="tasks", doc_col="readme"):
    queries, corpus, relevant_docs = get_sbert_ir_dicts(
        paperswithcode_df.dropna(subset=[query_col, doc_col]), query_col, doc_col
    )
    ir_evaluator = evaluation.InformationRetrievalEvaluator(
        queries,
        corpus,
        relevant_docs,
        main_score_function="cos_sim",
        map_at_k=[10],
        corpus_chunk_size=5000,
    )
    return ir_evaluator


def get_input_examples(df, target_col, source_cols, max_seq_length, model):
    sources = [df[src_col] for src_col in source_cols]
    src_lengths_masks = [
        pd.Series(
            src.str.split().apply(len) > 0
        )
        for src in sources
    ]
    target_source_pairs = [
        (target_text, source_text)
        for (mask, source) in zip(src_lengths_masks, sources)
        for (target_text, source_text) in zip(df[target_col][mask], source[mask])
    ]
    return [
        InputExample(
            texts=[
                " ".join(target.split()[:max_seq_length]),
                " ".join(source.split()[:max_seq_length]),
            ]
        )
        for (target, source) in tqdm.tqdm(target_source_pairs)
    ]


def make_rnn_model(
    word_embedding_model,
    model_type,
    pooling_mode_mean_tokens,
    pooling_mode_cls_token,
    pooling_mode_max_tokens,
    num_layers=2,
    n_hidden=256,
    dropout=0.25,
    max_seq_length=256,
):
    rnn_model_types = ["lstm", "sru"]
    maybe_chosen_model_type = [tpe for tpe in rnn_model_types if tpe in model_type]
    assert len(maybe_chosen_model_type) == 1
    chosen_model_type = maybe_chosen_model_type[0]

    rnn = SentenceRNN(
        word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
        dropout=dropout,
        hidden_dim=n_hidden,
        num_layers=num_layers,
        rnn_class_type=chosen_model_type,
    )

    pooling_model = models.Pooling(
        rnn.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=pooling_mode_mean_tokens,
        pooling_mode_cls_token=pooling_mode_cls_token,
        pooling_mode_max_tokens=pooling_mode_max_tokens,
    )
    model = SentenceTransformer(modules=[word_embedding_model, rnn, pooling_model])
    return model


def make_model(
    model_type,
    w2v_file,
    num_layers,
    n_hidden,
    dropout,
    max_seq_length,
    pooling_mode_mean_tokens,
    pooling_mode_cls_token,
    pooling_mode_max_tokens,
):
    if any(rnn_type in model_type for rnn_type in RNN_MODEL_TYPES):
        word_embedding_model = models.WordEmbeddings.from_text_file(w2v_file)
        model = make_rnn_model(
            word_embedding_model=word_embedding_model,
            model_type=model_type,
            num_layers=num_layers,
            n_hidden=n_hidden,
            dropout=dropout,
            max_seq_length=max_seq_length,
            pooling_mode_mean_tokens=pooling_mode_mean_tokens,
            pooling_mode_cls_token=pooling_mode_cls_token,
            pooling_mode_max_tokens=pooling_mode_max_tokens,
        )
    else:
        transformer = models.Transformer(model_type, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=pooling_mode_mean_tokens,
            pooling_mode_cls_token=pooling_mode_cls_token,
            pooling_mode_max_tokens=pooling_mode_max_tokens,
        )
        model = SentenceTransformer(modules=[transformer, pooling_model])
    return model
