"""Claude Code harness: runs the `claude` CLI on each instance with the benchmark MCP tools and writes records."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
STAGE2_CLAUDECODE_DIR = _HERE.parent
FINBENCH_DIR = STAGE2_CLAUDECODE_DIR.parent
_SHARED_AGENT_LIB = FINBENCH_DIR / "stage_2_shared" / "agent_lib"
MCP_SERVER_PY = FINBENCH_DIR / "stage_2_harness_opencode" / "mcp_server.py"

sys.path.insert(0, str(FINBENCH_DIR / "stage_2_shared"))

sys.path.insert(0, str(_HERE))

from lib.cell_spec import load_cell_spec


ALLOWED_TOOLS = [
    "mcp__financial-tools__financial_search",
    "mcp__financial-tools__read_news_article",
    "mcp__financial-tools__list_financial_report_sections",
    "mcp__financial-tools__read_financial_report_section",
    "mcp__financial-tools__read_earnings_call_transcript",
    "mcp__financial-tools__financial_calculate",
]


def _find_claude() -> str:
    """Find the `claude` binary. Checks PATH first, then VSCode extension."""
    if shutil.which("claude"):
        return "claude"
    vscode_ext_base = Path.home() / ".vscode-server" / "extensions"
    if vscode_ext_base.exists():
        candidates = sorted(
            (d for d in vscode_ext_base.iterdir() if d.name.startswith("anthropic.claude-code")),
            reverse=True,
        )
        for ext_dir in candidates:
            binary = ext_dir / "resources" / "native-binary" / "claude"
            if binary.exists() and os.access(binary, os.X_OK):
                return str(binary)
    npm_bin = shutil.which("npx")
    if npm_bin:
        return "npx"
    raise RuntimeError(
        "claude binary not found. Install with:\n"
        "  npm install -g @anthropic-ai/claude-code\n"
        "or ensure the VSCode Claude Code extension is installed."
    )


def _load_targets(targets_csv: Path) -> list[tuple[str, str, str]]:
    targets = []
    with open(targets_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            targets.append((
                row["ticker"].strip().upper(),
                row["fiscal_quarter"].strip(),
                row["ect_date"].strip(),
            ))
    return targets


sys.path.insert(0, str(_SHARED_AGENT_LIB))
from final_json import extract_final_json as _extract_final_json


def _build_mcp_config(
    ticker: str,
    fiscal_quarter: str,
    cutoff_date: str,
    start_date: str = "",
    api_key: Optional[str] = None,
) -> dict:
    """Build the --mcp-config JSON for this cell's run."""
    env: dict[str, str] = {}
    for key in (
        "HOME", "PATH", "PYTHONPATH", "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH",
        "LD_PRELOAD", "TMPDIR", "TMP", "TEMP",
        "ANTHROPIC_API_KEY",
        "SCHEMA_VARIANT", "NEWS_ABLATION", "FILINGS_ABLATION", "ALLOW_NO_RESEARCH",
        "CC_NEWS_ROOT", "EMBEDDING_SERVER_URL", "EMBEDDING_MODEL_NAME",
        "NEWS_SEARCH_SERVER_URL",
    ):
        val = os.environ.get(key)
        if val is not None:
            env[key] = val

    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key

    env.update({
        "TICKER": ticker,
        "FISCAL_QUARTER": fiscal_quarter,
        "CUTOFF_DATE": cutoff_date,
        "START_DATE": start_date,
        "EMBEDDING_MODEL_NAME": os.environ.get(
            "EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-8B"
        ),
    })

    return {
        "mcpServers": {
            "financial-tools": {
                "command": "python3",
                "args": [str(MCP_SERVER_PY)],
                "cwd": str(MCP_SERVER_PY.parent),
                "env": env,
            }
        }
    }


def _write_tool_log(
    events: list[dict],
    log_path: Path,
    ticker: str,
    fiscal_quarter: str,
    slice_days: int,
    cutoff_date: str,
    model: str,
) -> None:
    """Write a human-readable tool call log for one cell."""
    tool_uses: list[dict] = []
    tool_results: dict[str, str] = {}

    for event in events:
        t = event.get("type")
        if t == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tool_uses.append({
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    })
        elif t == "user":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_result":
                    tid = block.get("tool_use_id", "")
                    content = block.get("content", "")
                    if isinstance(content, list):
                        text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                        content = " ".join(text_parts)
                    tool_results[tid] = str(content)

    result_event = next((e for e in reversed(events) if e.get("type") == "result"), {})
    turns = result_event.get("num_turns", "?")
    cost = result_event.get("cost_usd") or result_event.get("total_cost_usd")
    cost_str = f"${cost:.4f}" if cost else "N/A"

    lines = [
        f"ticker={ticker}  fiscal_quarter={fiscal_quarter}  t-{slice_days}  cutoff={cutoff_date}",
        f"model={model}  turns={turns}  cost={cost_str}",
        "=" * 70,
        "",
    ]

    for i, tu in enumerate(tool_uses, 1):
        name = tu["name"]
        inp = tu["input"]

        arg_parts = []
        for k, v in inp.items():
            v_str = str(v)
            if len(v_str) > 120:
                v_str = v_str[:117] + "..."
            arg_parts.append(f"{k}={v_str!r}")
        args_str = ", ".join(arg_parts)

        lines.append(f"[{i}] {name}({args_str})")

        raw_result = tool_results.get(tu["id"], "")
        try:
            result_obj = json.loads(raw_result)
            if name == "financial_search":
                returned = result_obj.get("returned", "?")
                results = result_obj.get("results", [])
                top = results[0].get("title", "")[:80] if results else ""
                lines.append(f"    → {returned} results  top: {top!r}")
            elif name == "read_news_article":
                title = result_obj.get("title", "")[:80]
                chars = len(result_obj.get("content", ""))
                lines.append(f"    → {title!r}  ({chars} chars)")
            elif name in ("read_earnings_call_transcript", "read_financial_report_section"):
                has_more = result_obj.get("has_more", False)
                total = result_obj.get("total_length", "?")
                offset = result_obj.get("offset", 0)
                lines.append(f"    → offset={offset}  total_length={total}  has_more={has_more}")
            elif name == "list_financial_report_sections":
                count = result_obj.get("count", "?")
                sections = [s.get("section", "") for s in result_obj.get("sections", [])]
                lines.append(f"    → {count} sections: {', '.join(sections[:5])}" +
                             ("..." if len(sections) > 5 else ""))
            elif name == "financial_calculate":
                km = result_obj.get("income_statement", {}).get("key_metrics", {})
                lines.append(f"    → eps={km.get('eps')}  total_revenue={km.get('total_revenue')}")
            elif "error" in result_obj:
                lines.append(f"    → ERROR: {result_obj['error'][:120]}")
            else:
                summary = raw_result[:200].replace("\n", " ")
                lines.append(f"    → {summary}")
        except (json.JSONDecodeError, AttributeError):
            summary = raw_result[:200].replace("\n", " ")
            lines.append(f"    → {summary}")

        lines.append("")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines), encoding="utf-8")


def _print_event(event: dict) -> None:
    """Print a human-readable summary of one stream-json event to stderr."""
    t = event.get("type")
    if t == "assistant":
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "text" and block.get("text", "").strip():
                print(f"  [text] {block['text'][:200].rstrip()}", flush=True)
            elif block.get("type") == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input", {})
                summary = next(iter(inp.values()), "") if inp else ""
                if isinstance(summary, str):
                    summary = summary[:120]
                elif isinstance(summary, dict):
                    summary = str(summary)[:120]
                print(f"  [tool] {name}({summary})", flush=True)
    elif t == "user":
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "tool_result":
                print(f"  [tool_result] id={block.get('tool_use_id', '?')[:16]}", flush=True)
    elif t == "result":
        sub = event.get("subtype", "?")
        cost = event.get("cost_usd") or event.get("total_cost_usd")
        turns = event.get("num_turns", "?")
        cost_str = f"  cost=${cost:.4f}" if cost else ""
        print(f"  [result] subtype={sub}  turns={turns}{cost_str}", flush=True)


def run_one_cell(
    ticker: str,
    fiscal_quarter: str,
    ect_date: str,
    slice_days: int,
    claude_binary: str,
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
    verbose_output: bool = False,
    trajectory_path: Optional[Path] = None,
    tool_log_path: Optional[Path] = None,
    timeout_seconds: int = 1800,
    effort: Optional[str] = None,
) -> dict:
    """Run one (ticker, fiscal_quarter, slice) cell."""
    spec = load_cell_spec(ticker, fiscal_quarter, ect_date, slice_days)
    effective_cutoff = spec.effective_cutoff
    user_prompt = spec.user_prompt
    mcp_config = _build_mcp_config(
        ticker, fiscal_quarter, effective_cutoff,
        start_date=spec.start_date or "", api_key=api_key)

    if claude_binary == "npx":
        cmd_prefix = ["npx", "--yes", "@anthropic-ai/claude-code"]
    else:
        cmd_prefix = [claude_binary]

    cmd = cmd_prefix + [
        "--print",
        "--verbose",
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--strict-mcp-config",
        "--tools", "",
        "--system-prompt", spec.system_prompt,
        "--mcp-config", json.dumps(mcp_config),
        "--allowed-tools", ",".join(ALLOWED_TOOLS),
        "--permission-mode", "bypassPermissions",
        "--model", model,
        "--no-session-persistence",
    ]
    if effort:
        cmd += ["--effort", effort]
    cmd.append(user_prompt)

    cli_env = _cli_env(api_key)

    import signal
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=cli_env,
            text=True,
            cwd=str(STAGE2_CLAUDECODE_DIR),
        )
    except Exception as e:
        return {"ok": False, "error": f"Subprocess error: {e}"}

    events: list[dict] = []
    final_text = ""
    stderr_lines: list[str] = []
    early_error: Optional[dict] = None

    import threading

    def _read_stderr():
        for line in proc.stderr:
            stderr_lines.append(line)

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    try:
        import time
        deadline = time.monotonic() + timeout_seconds
        for raw_line in proc.stdout:
            if time.monotonic() > deadline:
                proc.kill()
                return {"ok": False, "error": f"Timeout after {timeout_seconds}s"}
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
            if verbose_output:
                _print_event(event)

            event_type = event.get("type")
            if event_type == "result":
                if event.get("subtype") == "success":
                    final_text = event.get("result", final_text)
                elif event.get("is_error"):
                    err_msg = event.get("result") or event.get("error") or f"subtype={event.get('subtype')}"
                    early_error = {"ok": False, "error": f"Agent error: {err_msg}", "events": events}
            elif event_type == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        final_text = block.get("text", final_text)
    finally:
        proc.stdout.close()
        proc.wait()
        stderr_thread.join(timeout=5)

    if early_error:
        return early_error

    if trajectory_path is not None:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(json.dumps(events, indent=2), encoding="utf-8")

    if tool_log_path is not None:
        _write_tool_log(events, tool_log_path, ticker, fiscal_quarter, slice_days, effective_cutoff, model)

    if proc.returncode not in (0, None) and not final_text:
        stderr_excerpt = "".join(stderr_lines)[:2000]
        return {"ok": False, "error": f"claude exited {proc.returncode}: {stderr_excerpt}"}

    if not final_text:
        return {"ok": False, "error": "No text output from agent", "rawText": ""}

    try:
        result = _extract_final_json(final_text)
        return {"ok": True, "result": result,
                "run_config": {"model": model, "effort": effort or "vendor-default",
                               "claude_cli": claude_binary,
                               "auto_compact": os.environ.get("DISABLE_AUTO_COMPACT", "1") != "1",
                               "max_output_tokens": int(os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000"))}}
    except Exception as e:
        return {"ok": False, "error": f"Failed to parse final JSON: {e}", "rawText": final_text}


def _cli_env(api_key: Optional[str]) -> dict:
    """Environment for the `claude` subprocess (compaction disabled, output cap set, API key passed through)."""
    def _session_internal(k: str) -> bool:
        return k.startswith("CLAUDE_CODE_") and not k.startswith(("CLAUDE_CODE_USE_", "CLAUDE_CODE_SKIP_"))
    env = {k: v for k, v in os.environ.items() if not _session_internal(k)}
    env.setdefault("DISABLE_AUTO_COMPACT", "1")
    env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    if env.get("ANTHROPIC_API_KEY"):
        cfg = Path(env.get("FINBENCH_CLAUDE_CONFIG_DIR")
                   or (Path(env.get("TMPDIR", "/tmp")) / "finbench-claude-cfg"))
        cfg.mkdir(parents=True, exist_ok=True)
        marker = cfg / ".claude.json"
        if not marker.exists():
            marker.write_text(json.dumps({
                "hasCompletedOnboarding": True,
                "projects": {str(STAGE2_CLAUDECODE_DIR): {"hasTrustDialogAccepted": True}},
            }))
        env["CLAUDE_CONFIG_DIR"] = str(cfg)
    return env


def _run_market_metrics(
    outcome: dict,
    ticker: str,
    fiscal_quarter: str,
    slice_days: int,
    ect_date: str,
    record_path: Path,
    model: str = "",
    effort: Optional[str] = None,
) -> None:
    """Write the prediction record through the shared record step (stage_2_shared/make_record.py)."""
    script = _SHARED_AGENT_LIB / "process_result.py"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--ticker", ticker,
                "--fiscal-quarter", fiscal_quarter,
                "--slice-days", str(slice_days),
                "--ect-date", ect_date,
                "--agent-model", model,
                "--agent-provider", f"claudecode/effort={effort or 'vendor-default'}",
                "--out", str(record_path),
            ],
            input=json.dumps(outcome),
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  [market-metrics subprocess failed: {e}]", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Claude Code EPS forecasting benchmark driver")
    p.add_argument("--targets", required=True,
                   help="CSV with columns: ticker, fiscal_quarter, ect_date")
    p.add_argument("--slices", default="21",
                   help="Comma-separated time slices in days (e.g. 21,14,7)")
    p.add_argument("--out", default=None,
                   help="Output directory for per-cell JSON (default: ../stage_4_experiments/claudecode/outputs)")
    p.add_argument("--records", default=None,
                   help="Output directory for PredictionRecord JSON (default: ../stage_4_experiments/claudecode/records)")
    p.add_argument("--trajectory-dir", default=None,
                   help="Output directory for per-cell trajectories (default: ../stage_4_experiments/claudecode/trajectories)")
    p.add_argument("--model", default="claude-opus-4-6",
                   help="Claude model to use (default: claude-opus-4-6)")
    p.add_argument("--effort", default=os.environ.get("CLAUDE_EFFORT") or None,
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="claude --effort level. Default: unset (vendor default). "
                        "Also read from $CLAUDE_EFFORT so run_parallel.sh needs no change.")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Per-cell timeout in seconds (default: 1800 = 30 min)")
    p.add_argument("--claude-binary", default=None,
                   help="Path to claude binary (auto-detected if not set)")
    p.add_argument("--api-key", default=None,
                   help="Anthropic API key. If omitted, falls back to ANTHROPIC_API_KEY env var "
                        "or Claude Code subscription auth.")
    p.add_argument("--verbose-output", action="store_true",
                   help="Print each tool call and text chunk to stdout in real time.")
    p.add_argument("--tool-log-dir", default=None,
                   help="Directory for per-cell tool call logs (default: ../stage_4_experiments/claudecode/tool_logs). "
                        "Each cell writes TICKER/FISCAL_QUARTER/t-DAYS.log")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    exp_dir = FINBENCH_DIR / "stage_4_experiments" / "claudecode"
    out_dir = Path(args.out) if args.out else exp_dir / "outputs"
    records_dir = Path(args.records) if args.records else exp_dir / "records"
    trajectory_dir = Path(args.trajectory_dir) if args.trajectory_dir else exp_dir / "trajectories"
    tool_log_dir = Path(args.tool_log_dir) if args.tool_log_dir else exp_dir / "tool_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)

    claude_binary = args.claude_binary or _find_claude()
    api_key: Optional[str] = args.api_key or os.environ.get("ANTHROPIC_API_KEY") or None
    print(f"Using claude binary: {claude_binary}")
    print(f"Auth: {'API key' if api_key else 'subscription (no API key provided)'}")

    slice_days_list = [int(s.strip()) for s in args.slices.split(",") if s.strip()]
    targets = _load_targets(Path(args.targets))
    print(f"Loaded {len(targets)} targets, slices={slice_days_list}")

    for ticker, fiscal_quarter, ect_date in targets:
        for slice_days in slice_days_list:
            tag = f"[{ticker}] {fiscal_quarter} t-{slice_days} (ect={ect_date})"
            print(f"{tag}...", flush=True)

            trajectory_path = trajectory_dir / ticker / fiscal_quarter / f"t-{slice_days}.json"
            tool_log_path = tool_log_dir / ticker / fiscal_quarter / f"t-{slice_days}.log"
            outcome = run_one_cell(
                ticker=ticker,
                fiscal_quarter=fiscal_quarter,
                ect_date=ect_date,
                slice_days=slice_days,
                claude_binary=claude_binary,
                model=args.model,
                api_key=api_key,
                verbose_output=args.verbose_output,
                trajectory_path=trajectory_path,
                tool_log_path=tool_log_path,
                timeout_seconds=args.timeout,
                effort=args.effort,
            )

            out_path = out_dir / ticker / fiscal_quarter / f"t-{slice_days}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(outcome, indent=2), encoding="utf-8")

            if outcome.get("ok"):
                km = outcome.get("result", {}).get("calculated_results_summary", {}).get("key_metrics", {})
                print(f"  OK  eps={km.get('eps')}  total_revenue={km.get('total_revenue')}")
                record_path = records_dir / ticker / fiscal_quarter / f"t-{slice_days}.record.json"
                _run_market_metrics(outcome, ticker, fiscal_quarter, slice_days, ect_date, record_path,
                                    model=args.model, effort=args.effort)
            else:
                print(f"  FAIL  {outcome.get('error')}", file=sys.stderr)


if __name__ == "__main__":
    main()
