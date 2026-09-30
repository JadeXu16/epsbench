"""Parse the final JSON answer from an agent's last message (shared by all harnesses)."""
from __future__ import annotations

import json
import re


def extract_final_json(text: str) -> dict:
    """Extract the final JSON object from the agent's output text."""
    def _try(s: str):
        try:
            return json.loads(s.strip())
        except Exception:
            return None

    cleaned = text.strip()

    out = _try(cleaned)
    if out is not None:
        return out

    for block in reversed(re.findall(r"```(?:json)?\s*([\s\S]*?)```", cleaned)):
        out = _try(block)
        if out is not None:
            return out

    i = cleaned.find("{")
    while i != -1:
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(cleaned)):
            c = cleaned[j]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    out = _try(cleaned[i:j + 1])
                    if out is not None:
                        return out
                    break
        i = cleaned.find("{", i + 1)

    raise ValueError("no parseable JSON object in final message")


