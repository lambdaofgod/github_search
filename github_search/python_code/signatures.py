from typing import List, Tuple, Callable
import re

Lines = List[str]

LineSplitter = Callable[[Lines], List[Tuple[int, int]]]


def _tag_line(line):
    first_strip_part = line.lstrip()
    if first_strip_part.startswith("def"):
        return "f"
    elif first_strip_part.startswith("@"):
        return "d"
    elif first_strip_part.startswith("class"):
        return "c"
    elif first_strip_part == "":
        return "e"
    else:
        return "o"


def get_split(lines, split_index):
    return lines[split_index[0] : split_index[1]]


def get_function_signature(def_lines):
    acc = []
    for line in def_lines:
        if line.endswith(":"):
            acc.append(line)
            break
        acc.append(line)
    return "\n".join(acc)


def _get_regex_line_matches(lines):
    line_reps = "".join([_tag_line(l) for l in lines])

    return re.finditer("(d[o]*)*f[oe]+(?!fc)", line_reps)


def regex_based_segment_splitter(lines: List[str]) -> List[Tuple[int, int]]:
    line_matches = _get_regex_line_matches(lines)
    return [(m.start(), m.end()) for m in line_matches]


def get_signatures(lines, line_splitter: LineSplitter = regex_based_segment_splitter):
    return [
        get_function_signature(get_split(lines, split_idx))
        for split_idx in line_splitter(lines)
    ]
