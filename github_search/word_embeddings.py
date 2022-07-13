import sentence_transformers


def build_word_embeddings_sentence_transformer_model(
    word_embeddings_file: str,
    pooling_mode_mean_tokens: bool = True,
    pooling_mode_max_tokens: bool = False,
) -> sentence_transformers.SentenceTransformer:
    """
    build from word embeddings from word_embeddings_file
    word_embeddings_file is assumed to be a word2vec .txt format file
    """
    word_embeddings = sentence_transformers.models.WordEmbeddings.from_text_file(
        word_embeddings_file
    )

    pooling_model = sentence_transformers.models.Pooling(
        word_embeddings.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=pooling_mode_mean_tokens,
        pooling_mode_max_tokens=pooling_mode_max_tokens,
    )
    model = sentence_transformers.SentenceTransformer(
        modules=[word_embeddings, pooling_model]
    )
    return model
