"""Record step for the Claude Code and vanilla harnesses. Kept under its original
name because both drivers call it; the logic lives in stage_2_shared/make_record.py
(EPS taken from the agent's calculator summary)."""
import runpy
import sys
from pathlib import Path

sys.argv[1:1] = ["--eps-source", "reported"]
runpy.run_path(str(Path(__file__).resolve().parents[1] / "make_record.py"), run_name="__main__")
