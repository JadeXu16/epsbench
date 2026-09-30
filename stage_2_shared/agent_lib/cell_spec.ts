// Prints one instance's system prompt, user prompt and information window as JSON
// (used by the Claude Code and vanilla harnesses, so all three harnesses share one prompt builder).

import { readFileSync } from "node:fs";
import path from "node:path";

import { loadSchema } from "./schema.ts";
import { buildUserPrompt, shiftDate } from "./prompt.ts";
import { prevEctDate } from "./prevEct.ts";
import { guardCutoff } from "./anndats.ts";

const SHARED_DIR = import.meta.dirname;

export function systemPromptFile(): string {
  const news = process.env.NEWS_ABLATION === "1";
  const filings = process.env.FILINGS_ABLATION === "1";
  return news && filings
    ? "financial-forecaster-noresearch.md"
    : filings
    ? "financial-forecaster-nofilings.md"
    : "financial-forecaster.md";
}

export function systemPrompt(): string {
  const file = systemPromptFile();
  const md = readFileSync(path.join(SHARED_DIR, file), "utf-8");
  const m = md.match(/^---\n[\s\S]*?\n---\n([\s\S]*)$/);
  if (!m) throw new Error(`could not strip frontmatter from ${file}`);
  return m[1].trim();
}

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : undefined;
}

function main(): void {
  const ticker = arg("ticker");
  const fiscalQuarter = arg("fiscal-quarter");
  const ectDate = arg("ect-date");
  const sliceDaysRaw = arg("slice-days");
  if (!ticker || !fiscalQuarter || !ectDate || sliceDaysRaw === undefined) {
    console.error("usage: cell_spec.ts --ticker T --fiscal-quarter 2025Q4 " +
                  "--ect-date YYYY-MM-DD --slice-days N");
    process.exit(2);
  }
  const sliceDays = parseInt(sliceDaysRaw, 10);

  let effectiveCutoff = shiftDate(ectDate, -sliceDays);
  if (process.env.ANNDATS_CUTOFF !== "0") {
    effectiveCutoff = guardCutoff(ticker, fiscalQuarter, effectiveCutoff);
  }
  const schema = loadSchema(ticker, fiscalQuarter);
  const startDate = process.env.ANCHOR_PREV_ECT !== "0"
    ? prevEctDate(ticker, fiscalQuarter, ectDate)
    : undefined;
  const userPrompt = buildUserPrompt(
    ticker, fiscalQuarter, schema, effectiveCutoff, startDate,
    process.env.ELICIT_UNCERTAINTY !== "0",
    process.env.NEWS_ABLATION === "1",
    process.env.FILINGS_ABLATION === "1",
  );

  process.stdout.write(JSON.stringify({
    effectiveCutoff,
    startDate: startDate ?? null,
    userPrompt,
    systemPrompt: systemPrompt(),
    systemPromptFile: systemPromptFile(),
    flags: {
      ANNDATS_CUTOFF: process.env.ANNDATS_CUTOFF !== "0",
      ANCHOR_PREV_ECT: process.env.ANCHOR_PREV_ECT !== "0",
      ELICIT_UNCERTAINTY: process.env.ELICIT_UNCERTAINTY !== "0",
      NEWS_ABLATION: process.env.NEWS_ABLATION === "1",
      FILINGS_ABLATION: process.env.FILINGS_ABLATION === "1",
      SCHEMA_VARIANT: process.env.SCHEMA_VARIANT ?? "",
    },
  }));
}

main();
