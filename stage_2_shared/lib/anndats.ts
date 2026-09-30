// Cap each instance's information cutoff at the day before its release date.
//
// The release date is the earlier of the earnings call and the official results
// announcement (data/instances.csv, column release_date). Some companies publish
// results a day before the call, so a cutoff derived from the call date alone
// could reach the day the results became public.

import { readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { shiftDate } from "./prompt.ts";

const INSTANCES_CSV = path.resolve(import.meta.dirname, "..", "..", "data", "instances.csv");

let _map: Map<string, string> | null = null;

function releaseDates(): Map<string, string> {
  if (_map) return _map;
  _map = new Map();
  if (!existsSync(INSTANCES_CSV)) return _map;
  const lines = readFileSync(INSTANCES_CSV, "utf-8").trim().split("\n");
  const header = lines[0].split(",").map((h) => h.trim());
  const iT = header.indexOf("ticker");
  const iQ = header.indexOf("fiscal_quarter");
  const iR = header.indexOf("release_date");
  if (iT < 0 || iQ < 0 || iR < 0) return _map;
  for (const line of lines.slice(1)) {
    const c = line.split(",");
    const rel = (c[iR] ?? "").trim();
    if (rel) _map.set(`${(c[iT] ?? "").trim().toUpperCase()}|${(c[iQ] ?? "").trim()}`, rel);
  }
  return _map;
}

// Only ever moves the cutoff earlier. shiftDate returns ISO "YYYY-MM-DD", so a
// string comparison is a date comparison.
export function guardCutoff(ticker: string, fiscalQuarter: string, cutoff: string): string {
  const rel = releaseDates().get(`${ticker.toUpperCase()}|${fiscalQuarter}`);
  if (!rel) return cutoff;
  const cap = shiftDate(rel, -1);
  return cap < cutoff ? cap : cutoff;
}
