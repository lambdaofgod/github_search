from operator import itemgetter

import pandas as pd
import re


class GenerationPostprocessor:
    @classmethod
    def _clean_parens(cls, s):
        s_without_enclosing_parens = re.sub(r"\]", ", ", s)
        s_semi_cleaned = re.sub(r",\s*,", ", ", s_without_enclosing_parens).replace(
            "\n", " "
        )
        return re.sub(",\W+", ", ", s_semi_cleaned)

    @classmethod
    def _sanitize_generated_text(cls, raw_generated_text, input_text):
        raw_generated_text = raw_generated_text.replace(input_text, "")
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
            d["generated_text"] = generated_text[0]
            d["tasks"] = cls._sanitize_generated_text(generated_text[0], input_text)
        return pd.DataFrame(
            dict(
                repo=prompt_info_dicts.apply(itemgetter("name")),
                tasks=prompt_info_dicts.apply(itemgetter("tasks")),
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
