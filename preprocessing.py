#import fastai.text



#from fastai.text import transform


import re

import nltk
from bs4 import BeautifulSoup
from markdown import markdown


class Rules:

    url_replacement = (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'xxurl ')
    code_replacement_basic = (r'<pre>(.*?)</pre>', ' ')
    code_replacement = (r'<code>(.*?)</code >', ' ')
    code_block_replacement = (r'```(.*?)```', ' ')
    hashtag_replacement = (r'#+', 'xxhashtag ')
    star_replacement = (r'\*+', 'xxstar')
    equality_replacement = (r'=+', 'xxequals')
    number_replacement = (r'[0-9]+', 'xxnumber')

    all_rules = [
        url_replacement,
        code_replacement_basic,
        code_replacement,
        code_block_replacement,
        hashtag_replacement,
        star_replacement,
        equality_replacement,
        number_replacement
    ]


def replace_with_rules(s, rules):
    substituted = s
    for pattern, replacement in rules:
        substituted = re.sub(pattern, replacement, substituted)

    return substituted


def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    html = replace_with_rules(html, Rules.all_rules)

    return BeautifulSoup(html).get_text()


def tokenize_markdown(markdown_string, n_tokens=250):
    text = markdown_to_text(markdown_string)
    return nltk.tokenize.wordpunct_tokenize(text)[:n_tokens]