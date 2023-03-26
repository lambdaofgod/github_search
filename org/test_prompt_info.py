#!/usr/bin/env python3

from prompt_info import PromptInfo


def test_prompt_args():
    repo_records = [
        {"repo": "r1", "files": ["f1"], "tasks": ["t1"]},
        {"repo": "r2", "files": ["f2"], "tasks": ["t2"]},
    ]
    predicted_repo_record = {"repo": "r3", "files": ["f3"], "tasks": ["t3"]}
    pinfo = PromptInfo(repo_records, predicted_repo_record, repo_text_field="files")

    assert len(pinfo.get_prompt_args()) == 9
    assert pinfo.get_prompt_args() == [
        "r1",
        "f1",
        "[t1]",
        "r2",
        "f2",
        "[t2]",
        "r3",
        "f3",
        "",
    ]
