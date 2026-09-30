// Start of each instance's news window: the date of the previous quarter's
// earnings call (data/instances.csv, column news_window_start).

import { readFileSync, existsSync } from "node:fs";
import path from "node:path";

const INSTANCES_CSV = path.resolve(import.meta.dirname, "..", "..", "data", "instances.csv");

let _map: Map<string, string> | null = null;

function windowStarts(): Map<string, string> {
  if (_map) return _map;
  _map = new Map();
  if (!existsSync(INSTANCES_CSV)) return _map;
  const lines = readFileSync(INSTANCES_CSV, "utf-8").trim().split("\n");
  const header = lines[0].split(",").map((h) => h.trim());
  const iT = header.indexOf("ticker");
  const iQ = header.indexOf("fiscal_quarter");
  const iS = header.indexOf("news_window_start");
  for (const line of lines.slice(1)) {
    const c = line.split(",");
    const s = (c[iS] ?? "").trim();
    if (s) _map.set(`${(c[iT] ?? "").trim().toUpperCase()}|${(c[iQ] ?? "").trim()}`, s);
  }
  return _map;
}

export function priorQuarter(fiscalQuarter: string): string {
  const year = parseInt(fiscalQuarter.slice(0, 4), 10);
  const q = parseInt(fiscalQuarter.slice(5), 10);
  return q > 1 ? `${year}Q${q - 1}` : `${year - 1}Q4`;
}

export function prevEctDate(
  ticker: string,
  fiscalQuarter: string,
  ectDate?: string,
): string | undefined {
  const prev = windowStarts().get(`${ticker.toUpperCase()}|${fiscalQuarter}`);
  if (!prev) return undefined;
  if (ectDate && prev >= ectDate) return undefined;
  return prev;
}
