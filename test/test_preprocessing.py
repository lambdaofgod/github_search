import preprocessing
import pytest


@pytest.mark.parametrize("url,expected", [
    ('https://github.com/mperham/sidekiq#requirements', 'xxurl #requirements'),
    ('https://github.com/mperham/sidekiq', 'xxurl ')
])
def test_url_replacement(url, expected):
    rule = preprocessing.Rules.url_replacement
    cleaned_url = preprocessing.replace_with_rules(url, [rule])
    assert cleaned_url == expected


@pytest.mark.parametrize("url,expected", [
    ('#', 'xxhashtag '),
    ('##', 'xxhashtag ')
])
def test_hashtag_replacement(url, expected):
    rule = preprocessing.Rules.hashtag_replacement
    cleaned_url = preprocessing.replace_with_rules(url, [rule])
    assert cleaned_url == expected
