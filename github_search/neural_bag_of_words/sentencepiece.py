from torchtext.data.functional import (
    sentencepiece_tokenizer,
    generate_sp_model,
    sentencepiece_tokenizer,
)


def create_sentencepiece_model(upstream, product, vocab_size):
    sp_model_prefix = os.path.splitext(product)[0]
    sp_model = generate_sp_model(
        upstream["nbow.prepare_dependency_bow_data"]["raw_text"],
        model_prefix=sp_model_prefix,
        vocab_size=vocab_size,
    )


def load_sentencepiece_model():
    pass
