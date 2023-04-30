from abc import ABC, abstractmethod
from typing import Dict, Union
import clearml
import pandas as pd
from pydantic import BaseModel, Field


class ClearMLExperimentManager:
    def __init__(
        self,
        project_name: str,
        task_name: str,
        config: Dict,
        task_type="data_processing",
    ):
        self.project_name = project_name
        self.task_name = task_name
        self.config = config
        self.task_type = task_type

    def __enter__(self):
        import clearml

        self.task = clearml.Task.create(
            project_name=self.project_name,
            task_name=self.task_name,
            task_type=self.task_type,
        )
        self.task.connect(self.config)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.task.close()
        self.task = None
        # Add optional error handling if needed
        if exc_type:
            print(f"An error of type {exc_type} occurred with value: {exc_value}")
        return False  # re-raise the exception if any (change to `True` if you want to suppress it)

    def add_artifact(
        self, name: str, artifact: Union[str, pd.DataFrame], metadata=None
    ):
        self.task.upload_artifact(name, artifact, metadata=metadata)
