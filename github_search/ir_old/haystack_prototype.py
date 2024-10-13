from haystack import Pipeline, Document


if __name__ == "__main__":

    docs = [{"name": "sample_document", "content": "This is a sample document."}]

    indexing_pipeline = Pipeline.load_from_yaml(
        "haystack_pipeline.yml", pipeline_name="indexing"
    )
    retrieval_pipeline = Pipeline.load_from_yaml(
        "haystack_pipeline.yml", pipeline_name="query"
    )

    indexing_pipeline.run(
        documents=[Document(d["content"], id=d["name"]) for d in docs]
    )

    retrieval_pipeline.run(query="sample query")
