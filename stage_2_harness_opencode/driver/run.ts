// OpenCode harness driver: for each instance, starts an OpenCode session with the
// benchmark MCP tools and the system prompt, sends the user prompt, and writes the
// output, trajectory and prediction record.

import { readFileSync, mkdirSync, writeFileSync, existsSync } from "node:fs";
import path from "node:path";
import { execFile } from "node:child_process";
import { createOpencode, type TextPart } from "@opencode-ai/sdk";
import { setGlobalDispatcher, Agent as UndiciAgent } from "undici";

import { loadSchema } from "../../stage_2_shared/lib/schema.ts";
import { buildUserPrompt, shiftDate } from "../../stage_2_shared/lib/prompt.ts";
import { buildConfig } from "./lib/config.ts";
import { prevEctDate } from "../../stage_2_shared/lib/prevEct.ts";
import { guardCutoff } from "../../stage_2_shared/lib/anndats.ts";

setGlobalDispatcher(new UndiciAgent({ headersTimeout: 0, bodyTimeout: 0 }));

const STAGE2_OPENCODE_DIR = path.resolve(import.meta.dirname, "..");
const DRIVER_DIR = path.resolve(import.meta.dirname);

interface Target {
  ticker: string;
  fiscalQuarter: string;
  ectDate: string;
}

function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      out[argv[i].slice(2)] = argv[i + 1];
      i++;
    }
  }
  return out;
}

function loadTargets(csvPath: string): Target[] {
  const lines = readFileSync(csvPath, "utf-8").trim().split("\n");
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cols = line.split(",");
    const row: Record<string, string> = {};
    header.forEach((h, i) => (row[h.trim()] = cols[i].trim()));
    return { ticker: row.ticker.toUpperCase(), fiscalQuarter: row.fiscal_quarter, ectDate: row.ect_date };
  });
}

function extractFinalJson(text: string): any {
  const tryParse = (s: string): any | undefined => {
    try { return JSON.parse(s.trim()); } catch { return undefined; }
  };
  const cleaned = text.trim();

  let out = tryParse(cleaned);
  if (out !== undefined) return out;

  const blocks = [...cleaned.matchAll(/```(?:json)?\s*([\s\S]*?)```/g)].map((m) => m[1]);
  for (const b of blocks.reverse()) {
    out = tryParse(b);
    if (out !== undefined) return out;
  }

  for (let i = cleaned.indexOf("{"); i !== -1; i = cleaned.indexOf("{", i + 1)) {
    let depth = 0, inStr = false, esc = false;
    for (let j = i; j < cleaned.length; j++) {
      const c = cleaned[j];
      if (esc) { esc = false; continue; }
      if (c === "\\") { esc = true; continue; }
      if (c === '"') { inStr = !inStr; continue; }
      if (inStr) continue;
      if (c === "{") depth++;
      else if (c === "}") {
        depth--;
        if (depth === 0) {
          out = tryParse(cleaned.slice(i, j + 1));
          if (out !== undefined) return out;
          break;
        }
      }
    }
  }
  throw new SyntaxError("no parseable JSON object in final message");
}

let AGENT_PROVIDER = "azure";
let AGENT_MODEL = "gpt-5.4";

async function runOneCell(
  ticker: string,
  fiscalQuarter: string,
  ectDate: string,
  sliceDays: number,
  trajectoryPath?: string,
  portOffset: number = 0,
): Promise<{ ok: true; result: any } | { ok: false; error: string; rawText?: string }> {
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

  const config = buildConfig(
    { ticker, fiscalQuarter, cutoffDate: effectiveCutoff, startDate },
    STAGE2_OPENCODE_DIR,
    AGENT_PROVIDER,
    AGENT_MODEL,
  );

  const port = parseInt(process.env.OPENCODE_PORT ?? "4096", 10) + portOffset;
  const { client, server } = await createOpencode({ config, port, timeout: 60_000 });
  try {
    const session = await client.session.create({ body: { title: `${ticker}-${fiscalQuarter}-t${sliceDays}` } });
    if (!session.data) return { ok: false, error: `session.create failed: ${JSON.stringify(session.error)}` };

    const response = await client.session.prompt({
      path: { id: session.data.id },
      body: {
        model: { providerID: AGENT_PROVIDER, modelID: AGENT_MODEL },
        agent: (process.env.NEWS_ABLATION === "1" && process.env.FILINGS_ABLATION === "1")
          ? "financial-forecaster-noresearch"
          : process.env.FILINGS_ABLATION === "1"
          ? "financial-forecaster-nofilings"
          : "financial-forecaster",
        parts: [{ type: "text", text: userPrompt }],
      },
    });

    if (trajectoryPath) {
      try {
        const messages = await client.session.messages({ path: { id: session.data.id } });
        mkdirSync(path.dirname(trajectoryPath), { recursive: true });
        writeFileSync(trajectoryPath, JSON.stringify(messages.data ?? messages, null, 2));
      } catch (e) {
        console.log(`  [trajectory fetch failed: ${e}]`);
      }
    }

    if (!response.data) return { ok: false, error: `session.prompt failed: ${JSON.stringify(response.error)}` };

    const textParts = response.data.parts.filter(
      (p): p is TextPart => p.type === "text",
    );
    const lastText = textParts[textParts.length - 1]?.text ?? "";

    try {
      return { ok: true, result: extractFinalJson(lastText) };
    } catch (e) {
      return { ok: false, error: `Failed to parse final JSON: ${e}`, rawText: lastText };
    }
  } finally {
    server.close();
  }
}

function runRecordAssemblySubprocess(
  outcome: { ok: true; result: any } | { ok: false; error: string },
  ticker: string,
  fiscalQuarter: string,
  sliceDays: number,
  ectDate: string,
  recordsDir: string,
): Promise<void> {
  const recordPath = path.join(recordsDir, ticker, fiscalQuarter, `t-${sliceDays}.record.json`);
  mkdirSync(path.dirname(recordPath), { recursive: true });
  return new Promise((resolve) => {
    const child = execFile(
      "python3",
      [
        path.join(STAGE2_OPENCODE_DIR, "scripts", "process_opencode_result.py"),
        "--ticker", ticker,
        "--fiscal-quarter", fiscalQuarter,
        "--slice-days", String(sliceDays),
        "--ect-date", ectDate,
        "--agent-model", AGENT_MODEL,
        "--agent-provider", AGENT_PROVIDER,
        "--out", recordPath,
      ],
      (err, stdout, stderr) => {
        if (stdout) process.stdout.write(stdout);
        if (stderr) process.stderr.write(stderr);
        if (err) console.log(`  [record-assembly subprocess failed: ${err}]`);
        resolve();
      },
    );
    child.stdin?.write(JSON.stringify(outcome));
    child.stdin?.end();
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.provider) AGENT_PROVIDER = args.provider;
  if (args.model) AGENT_MODEL = args.model;
  const targetsPath = args.targets ?? path.join(DRIVER_DIR, "targets_149.csv");
  const slicesDays = (args.slices ?? "1").split(",").map((s) => parseInt(s.trim(), 10));
  const outDir = args.out ?? path.join(DRIVER_DIR, "..", "..", "stage_4_experiments", "opencode", "manual", "outputs");
  const recordsDir = args.records ?? path.join(DRIVER_DIR, "..", "..", "stage_4_experiments", "opencode", "manual", "records");
  const trajectoryDir = args.trajectoryDir ?? path.join(DRIVER_DIR, "..", "..", "stage_4_experiments", "opencode", "manual", "trajectories");

  mkdirSync(outDir, { recursive: true });
  mkdirSync(recordsDir, { recursive: true });

  const concurrency = Math.max(1, parseInt(args.concurrency ?? "1", 10));
  const force = "force" in args;

  const targets = loadTargets(targetsPath);
  console.log(
    `Loaded ${targets.length} targets, slices=${slicesDays.join(",")}, ` +
    `model=${AGENT_PROVIDER}/${AGENT_MODEL}, ` +
    `concurrency=${concurrency}${force ? ", force" : ""}`,
  );

  interface Cell { ticker: string; fiscalQuarter: string; ectDate: string; sliceDays: number }
  const cells: Cell[] = [];
  let skipped = 0;
  for (const { ticker, fiscalQuarter, ectDate } of targets) {
    for (const sliceDays of slicesDays) {
      const outPath = path.join(outDir, ticker, fiscalQuarter, `t-${sliceDays}.json`);
      if (!force && existsSync(outPath)) {
        try {
          if (JSON.parse(readFileSync(outPath, "utf-8")).ok === true) {
            skipped++;
            continue;
          }
        } catch { /* unreadable -> re-run */ }
      }
      cells.push({ ticker, fiscalQuarter, ectDate, sliceDays });
    }
  }
  if (skipped) console.log(`Resume: skipping ${skipped} already-completed cells, ${cells.length} to run`);

  let ok = 0, fail = 0;
  async function worker(portOffset: number): Promise<void> {
    for (;;) {
      const cell = cells.shift();
      if (!cell) return;
      const { ticker, fiscalQuarter, ectDate, sliceDays } = cell;
      console.log(`[${ticker}] ${fiscalQuarter} t-${sliceDays} (ect=${ectDate})...`);
      const trajectoryPath = path.join(trajectoryDir, ticker, fiscalQuarter, `t-${sliceDays}.json`);
      let outcome: Awaited<ReturnType<typeof runOneCell>>;
      try {
        outcome = await runOneCell(ticker, fiscalQuarter, ectDate, sliceDays,
                                   trajectoryPath, portOffset);
      } catch (e) {
        const cause = e instanceof Error && e.cause ? ` (cause: ${e.cause})` : "";
        outcome = { ok: false, error: `runOneCell threw: ${e}${cause}` };
      }

      const outPath = path.join(outDir, ticker, fiscalQuarter, `t-${sliceDays}.json`);
      mkdirSync(path.dirname(outPath), { recursive: true });
      writeFileSync(outPath, JSON.stringify(outcome, null, 2));

      if (outcome.ok) {
        ok++;
        const km = outcome.result?.calculated_results_summary?.key_metrics ?? {};
        console.log(`  [${ticker} t-${sliceDays}] OK  eps=${km.eps}  total_revenue=${km.total_revenue}`);
        await runRecordAssemblySubprocess(outcome, ticker, fiscalQuarter, sliceDays, ectDate, recordsDir);
      } else {
        fail++;
        console.log(`  [${ticker} t-${sliceDays}] FAIL  ${outcome.error}`);
      }
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(concurrency, cells.length) }, (_v, i) => worker(i)));
  console.log(`\nDone: ok=${ok} fail=${fail} skipped=${skipped}${fail ? "  — re-run the same command to retry failures" : ""}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
