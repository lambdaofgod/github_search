from collections import Counter
from operator import itemgetter

excluded_langs = ["JavaScript", "HTML", "CSS"]


def get_languages_counter(languages):
    languages_counter = Counter(lang for langs in languages for lang in langs)
    return sorted(languages_counter.items(), key=itemgetter(1), reverse=True)


def get_top_langs(langs, n_top_langs, excluded_langs=excluded_langs):
    languages_counter = get_languages_counter(langs)
    top_langs = list(
        map(itemgetter(0), languages_counter[: n_top_langs + len(excluded_langs)])
    )
    return [lang for lang in top_langs if not lang in excluded_langs]


def filter_non_top_langs(languages, use_top_langs=20, excluded_langs=excluded_langs):
    top_langs = get_top_langs(languages, use_top_langs, excluded_langs)
    return [[lang for lang in langs if lang in top_langs] for langs in languages]
