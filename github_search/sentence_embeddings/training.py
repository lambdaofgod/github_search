





def make_sentence_similarity_pairs(record):
    return [(label, record["description"]) for label in record['labels']] +  [(record['full_name'], record["description"])]

def load_datasets():
    pass
