// Builds the user prompt for an instance from its schema and information window.

export function shiftDate(dateStr: string, days: number): string {
  const s = dateStr.replace(/-/g, "").slice(0, 8);
  const year = parseInt(s.slice(0, 4), 10);
  const month = parseInt(s.slice(4, 6), 10) - 1;
  const day = parseInt(s.slice(6, 8), 10);
  const dt = new Date(Date.UTC(year, month, day));
  dt.setUTCDate(dt.getUTCDate() + days);
  return dt.toISOString().slice(0, 10);
}

export function adjacentQuarters(fiscalQuarter: string): [string, string] {
  const year = parseInt(fiscalQuarter.slice(0, 4), 10);
  const q = parseInt(fiscalQuarter.slice(5), 10);
  const q1Num = q > 1 ? q - 1 : 4;
  const q1Year = q > 1 ? year : year - 1;
  const q4Num = q;
  const q4Year = year - 1;
  return [`Q${q1Num}_${q1Year}`, `Q${q4Num}_${q4Year}`];
}

export function priorQuarters(fiscalQuarter: string, n: number): string[] {
  let year = parseInt(fiscalQuarter.slice(0, 4), 10);
  let q = parseInt(fiscalQuarter.slice(5), 10);
  const out: string[] = [];
  for (let i = 0; i < n; i++) {
    q -= 1;
    if (q < 1) {
      q = 4;
      year -= 1;
    }
    out.push(`Q${q}_${year}`);
  }
  return out;
}

function formatBaselineValue(v: number): string {
  return v.toLocaleString("en-US", { minimumFractionDigits: 4, maximumFractionDigits: 4 });
}

function uncertaintyBlock(): string {
  return (
    `\n\n=== UNCERTAINTY (additional required output) ===\n` +
    `After financial_calculate returns, you must ALSO state an 80% prediction interval ` +
    `for the GAAP diluted EPS it computed. Add ONE extra key to the final JSON, ` +
    `alongside the keys the system prompt specifies:\n\n` +
    `  "eps_interval": { "p10": <float>, "p90": <float> }\n\n` +
    `Semantics: there should be a 10% chance the actual reported GAAP diluted EPS comes ` +
    `in BELOW p10, and a 10% chance it comes in ABOVE p90.\n` +
    `The point forecast from financial_calculate stays exactly as computed — do NOT ` +
    `revise it to sit at the interval's midpoint, and do not call the calculator again.\n` +
    `Both under- and over-confidence are penalised: the interval is scored on how often ` +
    `the actual lands inside it AND on its width, so a deliberately wide interval scores ` +
    `no better than a well-judged one.`
  );
}

export function buildUserPrompt(
  ticker: string,
  fiscalQuarter: string,
  schema: any,
  cutoffDate: string,
  startDate?: string,
  elicitUncertainty = false,
  newsAblation = false,
  filingsAblation = false,
): string {
  let dateHint: string;
  if (cutoffDate && startDate && !newsAblation) {
    dateHint = `News window: ${startDate} to ${cutoffDate}. financial_search will only return articles with pub_date in this range.`;
  } else if (cutoffDate) {
    dateHint = `Cutoff date: ${cutoffDate}.`;
  } else {
    dateHint = `Information cutoff: fixed by the environment this run was started with.`;
  }

  const [qMinus1, qMinus4] = adjacentQuarters(fiscalQuarter);
  const isMultiQuarter = schema?._validation?.target_fq_pretax !== undefined;
  const goesToEps = schema?.formulas?.eps !== undefined;
  const noResearch = newsAblation && filingsAblation;
  const historyClause = noResearch
    ? `Your only historical figures are the Q-4 baseline below and the GAAP ` +
      `tax/share anchors printed with the prediction fields — there is no other ` +
      `source this run, so judge each line from those alone`
    : filingsAblation
    ? `Your only historical figures are the Q-4 baseline below and the GAAP ` +
      `tax/share anchors printed with the prediction fields, so judge each line ` +
      `from those plus what the news tells you`
    : `Use the last four quarters to judge each line's trend/seasonality`;
  const forecastScope = goesToEps
    ? `you must forecast the WHOLE income statement yourself — every operating and ` +
      `non-operating line down to pretax income, PLUS the below-pretax lines (GAAP ` +
      `effective tax rate and diluted share count). ${historyClause}` +
      (filingsAblation
        ? `.`
        : `, and the GAAP tax/share history shown with the prediction fields to ` +
          `anchor the tax rate and share count.`)
    : `you must forecast EVERY line down to pretax income yourself, including ` +
      `non-operating lines (interest, other income/expense, amortization, etc.), so ` +
      `${noResearch
          ? `judge each such line from the Q-4 baseline below alone`
          : filingsAblation
          ? `judge each such line from the Q-4 baseline below plus the news`
          : `use the last four quarters to decide whether each such line is carried ` +
            `forward, trending, seasonal, or one-off`}.`;
  const [q2, q3] = priorQuarters(fiscalQuarter, 4).slice(1, 3);

  const variables: Record<string, number> = schema.variables_baseline ?? {};
  const staticRules: Record<string, any> = schema.static_rules ?? {};
  const baselineLines = Object.entries(variables)
    .filter(([k]) => staticRules[k]?.strategy !== "oracle")
    .map(([k, v]) => `  ${k}: ${formatBaselineValue(v)}`)
    .join("\n");

  const predSchema: Record<string, any> = schema.prediction_schema ?? {};
  const fmtRefs = (r: Record<string, any> | undefined): string => {
    if (!r) return "";
    const parts = ["last_q", "ttm", "q4"]
      .filter((k) => r[k] !== null && r[k] !== undefined)
      .map((k) => `${k}=${r[k]}`);
    return parts.length ? `  [GAAP history — ${parts.join(", ")}]` : "";
  };
  const fieldLines = Object.entries(predSchema).map(([fieldName, meta]) => {
    const fieldType = meta.type ?? "yoy";
    const desc = meta.description ?? fieldName;
    const unitHint =
      fieldType === "yoy"
        ? "Decimal fraction. e.g. 0.05 = +5% YoY, -0.03 = -3% YoY, 0.0 = flat."
        : fieldType === "rate_override"
        ? `Decimal fraction (e.g. 0.14 = 14%; may be NEGATIVE in a tax-benefit quarter). ` +
          `This is a GAAP effective rate: choose from the GAAP historical anchors shown — do ` +
          `NOT use management's guided tax rate, which is a non-GAAP figure and the wrong basis ` +
          `here. A discrete tax item can push the actual far from any anchor. Default ${meta.default}.`
        : fieldType === "absolute"
        ? "Dollar amount for the target quarter, SAME units/scale as the Q-4 baseline above " +
          "(not a growth rate). Predict the value directly; use 0 if none is expected. " +
          "Sign matters (a loss/expense may be negative)."
        : "Basis points. e.g. -20 = margin improved 20 bps, 50 = cost ratio rose 50 bps, 0 = unchanged.";
    return `  ${fieldName}: <float>  // ${desc}. ${unitHint}${fmtRefs(meta.references)}`;
  });
  const fieldsBlock = fieldLines.join("\n");

  const readingBlock = noResearch
    ? `Required reading:\n` +
      `  - None. Neither financial reports, earnings call transcripts, nor news are ` +
      `available in this run, and there is no tool to request any of them — do not ` +
      `attempt to read or search for anything beyond what is given below. ` +
      `This matters here: ${forecastScope}\n\n`
    : filingsAblation
    ? `Required reading:\n` +
      `  - Recent news via financial_search, then read_news_article on the most relevant hits.\n` +
      `Historical filings and earnings call transcripts are NOT available in this run, ` +
      `and there is no tool to request them — do not attempt to read reports or ` +
      `transcripts. This matters here: ${forecastScope}\n\n`
    : isMultiQuarter
    ? `Required reading:\n` +
      `  - Earnings call transcript (Q-1): quarter=${qMinus1}\n` +
      `  - Financial report sections (Q-1): quarter=${qMinus1}\n` +
      `  - Financial report sections (Q-4): quarter=${qMinus4}\n` +
      `You MAY also read the intervening quarters (Q-2=${q2}, Q-3=${q3}) — their ` +
      `financial reports — to judge trend and ` +
      `seasonality. This matters here: ${forecastScope}\n\n`
    : `Required reading:\n` +
      `  - Earnings call transcript (Q-1): quarter=${qMinus1}\n` +
      `  - Financial report sections (Q-1): quarter=${qMinus1}\n` +
      `  - Financial report sections (Q-4): quarter=${qMinus4}\n\n`;

  return (
    `Ticker: ${ticker}\n` +
    `Forecast target fiscal quarter: ${fiscalQuarter}\n` +
    `${dateHint}\n` +
    readingBlock +
    `All YoY values are relative to the same fiscal quarter one year ago (Q-4 = year-ago baseline).\n\n` +
    `=== Q-4 BASELINE (from most recent 10-Q filing) ===\n` +
    `${baselineLines}\n\n` +
    (noResearch
      ? `No research tools are available this run — work only from the Q-4 baseline above. `
      : filingsAblation
      ? `Research the news above first. `
      : `Read the required filings above first (ECT and financial report sections)` +
        (newsAblation ? `. ` : `, then use financial_search to discover relevant recent news. `)) +
    `When ready, call financial_calculate with your predicted driver values to see the ` +
    `full computed income statement.\n\n` +
    `PREDICTION FIELDS — you must predict EXACTLY these fields, no more and no fewer.\n` +
    `Only these fields are valid inputs to financial_calculate; do not invent or add others:\n` +
    `${fieldsBlock}` +
    (elicitUncertainty ? uncertaintyBlock() : "")
  );
}
