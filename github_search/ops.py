#!/usr/bin/env python3

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Dict


class ExperimentManager(AbstractContextManager, ABC):
    def __init__(self, project_name: str, task_name: str, project_config: Dict):
        self.project_name = project_name
        self.task_name = task_name
        self.project_config = project_config

    @abstractmethod
    @staticmethod
    def load_from_config(self, config_path) -> ExperimentManager:
        pass

    @abstractmethod
    def store_artifact(self, key: str, artifact: object):
        """
        Store an artifact with the given key.

        :param key: The key to store the artifact under.
        :param artifact: The artifact object to store.
        """
        pass
