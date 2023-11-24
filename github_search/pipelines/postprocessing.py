from operator import itemgetter

import pandas as pd
from github_search.pipelines.metrics_comparison import *
import re


class GenerationPostprocessor:
    @classmethod
    def _clean_parens(cls, s):
        s_without_enclosing_parens = re.sub(r"\]\s*\[", ", ", s)
        s_without_enclosing_parens = re.sub(
            r"\],\s*\[", ", ", s_without_enclosing_parens
        )
        return re.sub(r",\s*,", ", ", s_without_enclosing_parens).replace("]", "")

    @classmethod
    def _sanitize_generated_text(cls, raw_generated_text, input_text):
        raw_generated_text = raw_generated_text[0].replace(input_text, "")
        generated_text = raw_generated_text.split("##")[0].strip()
        return cls._clean_parens(generated_text)

    @classmethod
    def run(cls, generated_records):
        prompt_info_dicts = generated_records["prompt_info"]
        for d, generated_text, input_text in zip(
            generated_records["prompt_info"],
            generated_records["generated_text"],
            generated_records["input_text"],
        ):
            d["true_text"] = d["true_text"][0]
            d["generated_text"] = cls._sanitize_generated_text(
                generated_text, input_text
            )
        return pd.DataFrame(
            dict(
                repo=prompt_info_dicts.apply(itemgetter("name")),
                tasks=generated_records["generated_text"],
                true_tasks=prompt_info_dicts.apply(itemgetter("true_text")),
                generated_text=prompt_info_dicts.apply(itemgetter("generated_text")),
                prompt_info=prompt_info_dicts,
                generation=generated_records["generation"],
                input_text=generated_records["input_text"],
            )
        )

    @classmethod
    def convert_cols_to_dict(cls, df, cols):
        for c in cols:
            if type(df[c].iloc[0]) is list:
                df[c] = df[c].apply(lambda l: [dict(item) for item in l])
            else:
                df[c] = df[c].apply(dict)
        return df
