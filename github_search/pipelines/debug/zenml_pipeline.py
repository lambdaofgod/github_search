import ipdb
from zenml.client import Client

pipeline = Client().get_pipeline("generation_pipeline")
run = pipeline.last_run
artifacts = run.artifacts
generated_text_df = artifacts[-1].load()

ipdb.set_trace()
