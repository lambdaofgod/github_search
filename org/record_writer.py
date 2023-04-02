import json
from pydantic import BaseModel
from typing import Protocol, List
from contextlib import AbstractContextManager


class RecordWriter(Protocol, AbstractContextManager):
    def write_record(self, record):
        """writes a record to state managed by RecordWriter"""


class ListWriter(BaseModel):

    records_list: List[dict]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def write_record(self, record):
        json_str = json.dumps(record)
        self.file.write(json_str + "\n")


class JsonWriterContextManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None

    def __enter__(self):
        self.file = open(self.file_path, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def write_record(self, record):
        json_str = json.dumps(record)
        self.file.write(json_str + "\n")
