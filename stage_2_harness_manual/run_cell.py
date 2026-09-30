#!/usr/bin/env python3
"""Vanilla harness: a plain OpenAI SDK tool-calling loop over the benchmark tools, one instance per process."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FINBENCH = _HERE.parent
_AGENT_LIB = _FINBENCH / "stage_2_shared" / "agent_lib"
_OPENCODE = _FINBENCH / "stage_2_harness_opencode"
sys.path.insert(0, str(_AGENT_LIB))
from cell_spec import load_cell_spec
from final_json import extract_final_json

MAX_TURNS = 80
DEFAULT_AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"


def _load_tools(ticker: str, fiscal_quarter: str, cutoff: str, start: str | None):
    """Import mcp_server.py with this instance's context bound and return its tool functions."""
    os.environ["TICKER"] = ticker
    os.environ["FISCAL_QUARTER"] = fiscal_quarter
    os.environ["CUTOFF_DATE"] = cutoff
    os.environ["START_DATE"] = start or ""
    sys.path.insert(0, str(_OPENCODE))
    import mcp_server

    if mcp_server._FAISS_AVAILABLE and not mcp_server.EMBEDDING_SERVER_URL and not mcp_server.NEWS_ABLATION:
        mcp_server._preload_embed_model()
    else:
        mcp_server._embed_model_ready.set()
    mcp_server._register_financial_calculate_tool()

    specs, fns = [], {}
    for t in mcp_server.mcp._tool_manager.list_tools():
        params = dict(t.parameters)
        params.pop("title", None)
        for p in params.get("properties", {}).values():
            p.pop("title", None)
        specs.append({"type": "function",
                      "function": {"name": t.name, "description": t.description or "", "parameters": params}})
        fns[t.name] = t.fn
    return specs, fns


def _make_client(base_url: str | None):
    from openai import AzureOpenAI, OpenAI
    if base_url:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("--base-url given but OPENAI_API_KEY is not set")
        return OpenAI(base_url=base_url, api_key=key), base_url
    key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set (or pass --base-url + OPENAI_API_KEY)")
    ep = os.environ.get("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT)
    return AzureOpenAI(api_version=DEFAULT_AZURE_API_VERSION, azure_endpoint=ep, api_key=key), ep


def run_cell(ticker: str, fiscal_quarter: str, ect_date: str, slice_days: int,
             model: str, base_url: str | None, reasoning_effort: str | None,
             timeout_s: int) -> tuple[dict, dict]:
    """Run one instance; returns (outcome, trajectory), where outcome is the {"ok", "result", ...} wrapper."""
    spec = load_cell_spec(ticker, fiscal_quarter, ect_date, slice_days)
    tools, fns = _load_tools(ticker, fiscal_quarter, spec.effective_cutoff, spec.start_date)
    client, endpoint = _make_client(base_url)

    messages = [{"role": "system", "content": spec.system_prompt},
                {"role": "user", "content": spec.user_prompt}]
    traj = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0}
    tool_calls_n = 0
    started = time.time()
    finish = None
    final_text = ""

    extra = {}
    if reasoning_effort:
        extra["reasoning_effort"] = reasoning_effort

    for turn in range(1, MAX_TURNS + 1):
        if time.time() - started > timeout_s:
            finish = "timeout"
            break
        resp = client.chat.completions.create(model=model, messages=messages, tools=tools, **extra)
        u = getattr(resp, "usage", None)
        if u is not None:
            usage["input_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            usage["output_tokens"] += getattr(u, "completion_tokens", 0) or 0
            d = getattr(u, "prompt_tokens_details", None)
            usage["cached_input_tokens"] += (getattr(d, "cached_tokens", 0) or 0) if d else 0
            d = getattr(u, "completion_tokens_details", None)
            usage["reasoning_tokens"] += (getattr(d, "reasoning_tokens", 0) or 0) if d else 0
        choice = resp.choices[0]
        msg = choice.message
        finish = choice.finish_reason
        messages.append(msg)
        traj.append({"role": "assistant", "content": msg.content, "finish_reason": finish,
                     "tool_calls": [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                                    for tc in (msg.tool_calls or [])]})
        if not msg.tool_calls:
            final_text = msg.content or ""
            break
        for tc in msg.tool_calls:
            tool_calls_n += 1
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                fn = fns.get(name)
                if fn is None:
                    out = {"error": f"unknown tool {name}"}
                else:
                    out = fn(**args)
            except Exception as e:
                out = {"error": f"{type(e).__name__}: {e}"}
            out_s = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out_s})
            traj.append({"role": "tool", "tool_call_id": tc.id, "tool_name": name, "content": out_s})
    else:
        finish = "max_turns"

    trajectory = {
        "metadata": {
            "ticker": ticker, "fiscal_quarter": fiscal_quarter, "slice_days_before": slice_days,
            "ect_date": ect_date, "effective_cutoff": spec.effective_cutoff, "start_date": spec.start_date,
            "model": model, "endpoint": endpoint,
            "reasoning_effort": reasoning_effort or "vendor-default",
            "system_prompt_file": spec.system_prompt_file, "flags": spec.flags,
            "turns": len([m for m in traj if m["role"] == "assistant"]),
            "tool_calls": tool_calls_n, "finish": finish, "elapsed_s": round(time.time() - started, 1),
            **usage,
        },
        "messages": [{"role": "system", "content": spec.system_prompt},
                     {"role": "user", "content": spec.user_prompt}] + traj,
    }
    run_config = {"model": model, "reasoning_effort": reasoning_effort or "vendor-default", "endpoint": endpoint}
    if not final_text:
        return {"ok": False, "error": f"no final text (finish={finish})", "rawText": "",
                "run_config": run_config}, trajectory
    try:
        result = extract_final_json(final_text)
    except Exception as e:
        return {"ok": False, "error": f"Failed to parse final JSON: {e}", "rawText": final_text,
                "run_config": run_config}, trajectory
    return {"ok": True, "result": result, "rawText": final_text, "run_config": run_config}, trajectory


def _process(outcome: dict, ticker: str, fq: str, ect: str, slice_days: int, record_path: Path) -> bool:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    rc = outcome.get("run_config", {})
    proc = subprocess.run(
        [sys.executable, str(_AGENT_LIB / "process_result.py"),
         "--ticker", ticker, "--fiscal-quarter", fq, "--slice-days", str(slice_days),
         "--ect-date", ect, "--agent-model", rc.get("model", ""),
         "--agent-provider", f"manual/effort={rc.get('reasoning_effort', 'vendor-default')}",
         "--out", str(record_path)],
        input=json.dumps(outcome), capture_output=True, text=True, env={**os.environ})
    if proc.returncode != 0:
        print(f"  [process_result FAILED] {proc.stderr[-800:]}", flush=True)
        return False
    return True


def _run_one_in_subprocess(args, ticker, fq, ect) -> None:
    """Each cell is its own process (see module docstring)."""
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--ticker", ticker, "--fiscal-quarter", fq, "--ect-date", ect,
           "--slice-days", str(args.slice_days), "--model", args.model,
           "--out-root", args.out_root, "--timeout", str(args.timeout)]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    if args.reasoning_effort:
        cmd += ["--reasoning-effort", args.reasoning_effort]
    subprocess.run(cmd, env={**os.environ})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", help="CSV with ticker,fiscal_quarter,ect_date (loops, resume-safe)")
    ap.add_argument("--ticker"); ap.add_argument("--fiscal-quarter"); ap.add_argument("--ect-date")
    ap.add_argument("--slice-days", type=int, default=1)
    ap.add_argument("--model", required=True, help="deployment / model id, e.g. gpt-5.5")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint; omit for our Azure resource")
    ap.add_argument("--reasoning-effort", default=None, choices=["none", "low", "medium", "high", "xhigh"],
                    help="NOT set by default (vendor default), same as the OpenCode arms")
    ap.add_argument("--out-root", required=True, help="…/<batch>/2_records (outputs/ records/ trajectories/ go under it)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell wall clock, seconds")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if args.targets:
        rows = list(csv.DictReader(open(args.targets)))
        for r in rows:
            t, fq, ect = r["ticker"].strip().upper(), r["fiscal_quarter"].strip(), r["ect_date"].strip()
            rec = out_root / "records" / t / fq / f"t-{args.slice_days}.record.json"
            if rec.exists():
                continue
            print(f"[{t}] {fq} t-{args.slice_days} (ect={ect})…", flush=True)
            _run_one_in_subprocess(args, t, fq, ect)
        return

    if not (args.ticker and args.fiscal_quarter and args.ect_date):
        ap.error("either --targets or all of --ticker/--fiscal-quarter/--ect-date")
    t, fq, ect, s = args.ticker.upper(), args.fiscal_quarter, args.ect_date, args.slice_days
    outcome, trajectory = run_cell(t, fq, ect, s, args.model, args.base_url, args.reasoning_effort, args.timeout)

    for sub, obj in (("outputs", outcome), ("trajectories", trajectory)):
        p = out_root / sub / t / fq / f"t-{s}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    m = trajectory["metadata"]
    if outcome.get("ok"):
        km = outcome["result"].get("calculated_results_summary", {}).get("key_metrics", {})
        ok = _process(outcome, t, fq, ect, s, out_root / "records" / t / fq / f"t-{s}.record.json")
        print(f"  {'OK ' if ok else 'ERR'} eps={km.get('eps')} turns={m['turns']} tools={m['tool_calls']} "
              f"in={m['input_tokens']} cached={m['cached_input_tokens']} out={m['output_tokens']}", flush=True)
        sys.exit(0 if ok else 1)
    print(f"  FAIL {outcome.get('error')} (turns={m['turns']} tools={m['tool_calls']})", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
