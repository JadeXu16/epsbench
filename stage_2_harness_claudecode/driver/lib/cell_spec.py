"""Re-export of stage_2_shared/agent_lib/cell_spec.py for the Claude Code driver."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "stage_2_shared" / "agent_lib"))
from cell_spec import CellSpec, load_cell_spec
