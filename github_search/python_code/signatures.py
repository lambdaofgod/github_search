from typing import List, Tuple, Callable
import re
import itertools
import polars as pl


Lines = List[str]

LineSplitter = Callable[[Lines], List[Tuple[int, int]]]

SignatureExtractor = Callable[[Lines], List[str]]


def _tag_line(line):
    first_strip_part = line.lstrip()
    if first_strip_part.startswith("def "):
        return "f"
    elif first_strip_part.startswith("@"):
        return "d"
    elif first_strip_part.startswith("class "):
        return "c"
    elif first_strip_part == "":
        return "e"
    else:
        return "o"


def get_split(lines, split_index):
    return lines[split_index[0] : split_index[1]]


def get_function_signature(def_lines):
    acc = []
    for line in def_lines:
        if line.endswith(":"):
            acc.append(line)
            break
        acc.append(line)
    return "\n".join(acc)


def _get_regex_function_matches(tags):
    return re.finditer("(d[o]*)*f[oe]+(?!fc)", tags)


def _get_regex_class_matches(tags):
    return re.finditer("(d[o]*)*c[oe]+(?!fc)", tags)


def regex_based_segment_splitter(lines: List[str]) -> List[Tuple[int, int]]:
    line_reps = "".join([_tag_line(l) for l in lines])
    line_matches = itertools.chain.from_iterable(
        [_get_regex_function_matches(line_reps), _get_regex_class_matches(line_reps)]
    )
    return ((m.start(), m.end()) for m in line_matches)


def get_signatures(lines, line_splitter: LineSplitter = regex_based_segment_splitter):
    return [
        get_function_signature(get_split(lines, split_idx))
        for split_idx in line_splitter(lines)
    ]


def get_function_name_from_signature(signature):
    m = re.search(r"def (.*)\(", signature)
    if not m is None:
        return m.group(1)
    else:
        return signature


def get_name_from_signature(signature):
    return get_function_name_from_signature(signature)


class SignatureSelector:
    @staticmethod
    def get_rarest_signatures_pl(signatures_path, n_rarest):
        signatures_pldf = pl.scan_parquet(signatures_path).unique(
            subset=["function_signature", "repo_name"]
        )
        function_counts_pldf = signatures_pldf.groupby("function_name").agg(
            [pl.count()]
        )
        signatures_with_counts_pldf = signatures_pldf.join(
            function_counts_pldf, on="function_name"
        )
        return signatures_with_counts_pldf.groupby("repo_name").apply(
            lambda df: df.sort("count").head(n_rarest),
            schema=signatures_with_counts_pldf.schema,
        )

    @staticmethod
    def prepare_rarest_signatures_corpus_pldf(signatures_path, n_rarest):
        rarest_signatures_pldf = SignatureSelector.get_rarest_signatures_pl(
            signatures_path, n_rarest
        ).collect()
        agg_signatures_pl = rarest_signatures_pldf.groupby(["repo_name"]).agg_list()
        return agg_signatures_pl.with_columns(
            [
                agg_signatures_pl["path"].apply(" ".join),
                agg_signatures_pl["function_name"].apply(" ".join),
                agg_signatures_pl["function_signature"].apply("\npass\n".join),
            ]
        )
