"""Get an instance's prompts and information window from the TypeScript implementation shared by all harnesses."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

_SHARED_LIB = Path(__file__).resolve().parent


class CellSpec:
    """Everything the driver needs to run one cell, computed by shared code."""

    def __init__(self, d: dict):
        self.effective_cutoff: str = d["effectiveCutoff"]
        self.start_date: Optional[str] = d["startDate"]
        self.user_prompt: str = d["userPrompt"]
        self.system_prompt: str = d["systemPrompt"]
        self.system_prompt_file: str = d.get("systemPromptFile", "")
        self.flags: dict = d["flags"]


def load_cell_spec(ticker: str, fiscal_quarter: str, ect_date: str,
                   slice_days: int, timeout: int = 120) -> CellSpec:
    """Ask the shared TS implementation for this cell's prompt + dates."""
    cmd = [
        "npx", "tsx", str(_SHARED_LIB / "cell_spec.ts"),
        "--ticker", ticker,
        "--fiscal-quarter", fiscal_quarter,
        "--ect-date", ect_date,
        "--slice-days", str(slice_days),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, cwd=str(_SHARED_LIB),
                              env={**os.environ})
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"cell_spec.ts timed out after {timeout}s") from e
    if proc.returncode != 0:
        raise RuntimeError(
            f"cell_spec.ts failed (rc={proc.returncode})\n"
            f"stderr: {proc.stderr[-2000:]}")
    try:
        return CellSpec(json.loads(proc.stdout))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"cell_spec.ts emitted non-JSON:\n{proc.stdout[:2000]}") from e
