import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd


def batched_generate(
    model, tokenizer, texts, batch_size=32, max_length=64, num_beams=5, **kwargs
):
    """
    generate text in batches using Seq2Seq model
    """
    generated_sequences = []
    for i in range(int(np.ceil(len(texts) / batch_size))):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        text_chunk = texts[batch_start:batch_end]
        input_ids = tokenizer.batch_encode_plus(
            text_chunk,
            max_length=320,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(
            **input_ids,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=1
        )

        batch_generated_sequences = tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        generated_sequences += batch_generated_sequences
    return generated_sequences


def add_summaries(transformer_model_name, df, summarized_col, summary_col_name):
    tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(transformer_model_name)
    model = model.cuda().half()
    df[summary_col_name] = batched_generate(
        model, tokenizer, df[summarized_col].to_list()
    )
    return df


def prepare_function_df_with_summarized_code(transformer_model_name, upstream, product):
    df = pd.read_feather(str(upstream["prepare_function_code_df"]))
    df = add_summaries(
        transformer_model_name, df, "function_code", "function_description"
    )
    df.to_feather(str(product))
