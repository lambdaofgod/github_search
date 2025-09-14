colorcodings = {
    "generated_tasks": "red",
    "repository_signature": "red",
    "dependency_signature": "red",
    "selected_code": "orange",
    "code2doc_generated_readme": "green",
    "code2doc_generation_context": "green",
    "code2doc_reasoning": "green",
    "generated_readme (repomap)": "cyan",
    "generation_context (repomap)": "cyan",
    "generated_rationale (repomap)": "cyan",
}


def rename_source(corpus_name):
    for source in ["dep_sig", "repomap"]:
        if source in corpus_name:
            corpus_name = corpus_name.replace(source + "_", "")
            return corpus_name + f" ({source})"

    if corpus_name == "code2doc_generated_readme":
        return "code2doc_generated_readme (selected code)"
    elif corpus_name == "code2doc_files_summary":
        return "code2doc_files_summary(selected code)"
    else:
        return corpus_name


def colorcode(corpus_name):
    if "signature" in corpus_name:
        return "red"
    if "dep_sig" in corpus_name:
        return "blue"
    elif "repomap" in corpus_name:
        return "cyan"
    elif "code2doc" in corpus_name:
        return "green"
    else:
        return None


def format_corpus_for_latex(corpus_name, intensity=25):
    maybe_color = colorcode(corpus_name)
    if maybe_color is None:
        return corpus_name
    else:
        corpus_name = corpus_name.replace("generation_context", "code summary")

        corpus_name = rename_source(corpus_name)
        corpus_name = corpus_name.replace("dep_sig", "dependency signature")

        return f"{{\color{{{maybe_color}}} {corpus_name}}}"


def format_retriever_for_latex(retriever_name, intensity=25):
    if retriever_name == "bm25":
        retriever_name = "BM25"
    elif "all-mpnet-base-v2" in retriever_name:
        retriever_name = "MPNet"
    elif "all-MiniLM-L12-v2" in retriever_name:
        retriever_name = "MiniLM"
    elif "embeddinggemma" in retriever_name:
        retriever_name = "gemma embedder"
    elif "static-retrieval" in retriever_name:
        retriever_name = "static embedder"
    elif retriever_name == "Word2Vec":
        retriever_name = retriever_name
    else:
        pass
    return retriever_name.replace("_", " ")
