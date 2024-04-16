import abc
from pydantic import BaseModel
from typing import List
from comment_parser import comment_parser
import re
import tqdm


class PythonCodeSelector(abc.ABC):
    @abc.abstractmethod
    def extract(self, code) -> List[dict]:
        pass

    def extract_str(self, code):
        return [selection["text"] for selection in self.extract(code)]

    def select_code(self, code):
        matches = self.extract(code)
        selected_code = "\n...\n".join(
            [m["text"] for m in self._merge_matches(matches)]
        )
        return selected_code

    @classmethod
    def _merge_matches(cls, matches):
        merged_matches = []
        tmp_match = matches[0]
        for match in matches[1:]:
            if (
                match["match_type"] == tmp_match["match_type"]
                and match["line_start"] == tmp_match["line_end"] + 1
            ):
                tmp_match = {
                    "text": tmp_match["text"] + "\n" + match["text"],
                    "line_start": tmp_match["line_start"],
                    "line_end": match["line_end"],
                    "match_type": tmp_match["match_type"],
                }
            else:
                merged_matches.append(tmp_match)
                tmp_match = match
        return merged_matches


class CombinedSelector(PythonCodeSelector, BaseModel):
    selectors: List[PythonCodeSelector]

    def extract(self, code):
        extracted_parts = []
        for selector in self.selectors:
            extracted_parts += selector.extract(code)
        return sorted(extracted_parts, key=lambda r: r["line_start"])

    class Config:
        arbitrary_types_allowed = True


class CommentSelector(PythonCodeSelector):
    def extract(self, code):
        comments = comment_parser.python_parser.extract_comments(code)
        return [
            {
                "text": "#" + c.text(),
                "line_start": c.line_number(),
                "line_end": c.line_number(),
                "match_type": "comment",
            }
            for c in comments
        ]


class SignatureSelector(PythonCodeSelector, BaseModel):
    pattern: re.Pattern = re.compile("(\s+ def|class) (.*:$)", re.MULTILINE)

    def extract(self, code):
        re_newline = re.compile(r"\n")
        matches = []
        for match in self.pattern.finditer(code):
            start = match.start()
            line_start = code.count("\n", 0, match.start())
            line_offset = code.count("\n", start, match.end()) + 1
            s = match.group()
            matches.append(
                {
                    "text": s,
                    "line_start": line_start,
                    "line_end": line_start + line_offset,
                    "match_type": "signature",
                }
            )
        return matches


def get_python_files_with_selected_code_df(python_files_df):
    selector = CombinedSelector(selectors=[CommentSelector(), SignatureSelector()])
    selected_python_code_contents = []

    for code in tqdm.tqdm(python_files_df["content"]):
        try:
            selected_python_code_contents.append(selector.select_code(code))
        except KeyboardInterrupt:
            break
        except:
            selected_python_code_contents.append(None)

    python_files_df["selected_code"] = selected_python_code_contents
    return python_files_df
