import fire
import pandas as pd
import ipdb
from zenml.client import Client
import pickle
import ipdb
import logging
from pydantic import BaseModel
from pathlib import Path
from fastapi.encoders import jsonable_encoder
import json

logging.basicConfig(level="INFO")


class ArtifactSaver(BaseModel):
    output_dir: str

    def save_artifact(self, artifact, artifact_name):
        if artifact_name.endswith("df"):
            artifact_df = self.load_dataframe(artifact)
            out_path = f"{self.output_dir}/{artifact_name}.json"
            artifact_df.reset_index().to_json(out_path)
        else:
            try:
                out_path = f"{self.output_dir}/{artifact_name}.json"
                with open(out_path, "w") as f:
                    json.dump(jsonable_encoder(artifact), f)
            except Exception as e:
                print(e)

        logging.info(f"downloaded artifact: {artifact_name} to {out_path}")
        return out_path

    @classmethod
    def load_dataframe(cls, artifact):
        if type(artifact) is not pd.DataFrame:
            return pd.DataFrame.from_records(artifact)
        else:
            return artifact


class ArtifactLoader:

    @classmethod
    def load_artifact(cls, pipeline_name, artifact_name):
        """
        returns the artifact from most current pipeline run
        """

        pipeline_model = Client().get_pipeline(pipeline_name)
        runs = pipeline_model.runs
        run = runs[0]
        for artifact in run.artifacts:
            if artifact.name == artifact_name:
                return artifact.load()

        raise ValueError(f"no such artifact {artifact_name}")


class Main:
    @classmethod
    def download_artifacts(
        cls, pipeline_name: str = "generation_pipeline", output_dir="results"
    ):
        pipeline_model = Client().get_pipeline(pipeline_name)
        runs = pipeline_model.runs
        run = runs[0]

        steps = run.steps
        step_names = list(steps.keys())

        pipeline_output_dir = Path(output_dir) / pipeline_name
        pipeline_output_dir.mkdir(exist_ok=True, parents=True)
        saver = ArtifactSaver(output_dir=str(pipeline_output_dir))
        for step_name in step_names:
            pipeline_step = run.steps[step_name]
            logging.info(f"found pipeline_step: {pipeline_step.name}")
            for artifact_name in pipeline_step.outputs:
                artifact = pipeline_step.outputs[artifact_name]
                out_path = saver.save_artifact(artifact.load(), artifact_name)


if __name__ == "__main__":
    fire.Fire(Main)
