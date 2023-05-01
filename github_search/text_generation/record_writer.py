import json
from pydantic import BaseModel
from typing import Protocol, List
from contextlib import AbstractContextManager


class RecordWriterContextManager(AbstractContextManager):
    def write_record(self, record):
        """writes a record to state managed by RecordWriter"""


class ListWriter(BaseModel, RecordWriterContextManager):

    records_list: List[dict]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def write_record(self, record):
        json_str = json.dumps(record)
        self.file.write(json_str + "\n")


class JsonWriterContextManager(BaseModel, RecordWriterContextManager):
    def __init__(self, file_path):
        self.file_path = file_path

    def __enter__(self):
        self.file = open(self.file_path, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def write_record(self, record):
        json_str = json.dumps(record)
        self.file.write(json_str + "\n")


class RecordWriter(BaseModel):

    writer_ctx_manager: RecordWriterContextManager

    def map(self, fn, inputs, total=None, progress=True, **kwargs):
        """
        map a function over inputs, writing each result with a record writer
        """
        iterable = tqdm.tqdm(inputs, total=total) if progress else inputs
        with record_writer_cls(**kwargs) as writer:
            for item in iterable:
                result = fn(item)
                writer.write_record(result)
                yield result

    class Config:
        arbitrary_types_allowed = True
