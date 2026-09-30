// Builds the OpenCode configuration for one run: the model provider, and a fresh
// MCP tool server whose environment carries this run's ticker, quarter and
// information window.
import path from "node:path";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export interface RunContext {
  ticker: string;
  fiscalQuarter: string;
  cutoffDate: string;
  startDate?: string;
}

// OpenAI-compatible endpoints: they differ only in base URL and in which
// environment variable holds the key. GLM's per-provider limit is overridden per
// model in buildProvider().
const OPENAI_COMPATIBLE: Record<string, { name: string; baseURL: string; env: string;
  limit?: { context: number; output: number } }> = {
  qwen_us: {
    name: "Alibaba Model Studio (Qwen, US region)",
    baseURL: "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    env: "DASHSCOPE_API_KEY_US",
  },
  qwen: {
    name: "Alibaba Model Studio (Qwen, international)",
    baseURL: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    env: "DASHSCOPE_API_KEY",
  },
  zhipu: {
    name: "Z.ai (GLM)",
    baseURL: "https://api.z.ai/api/paas/v4",
    env: "ZHIPU_API_KEY",
    limit: { context: 128000, output: 16384 },
  },
};

// Per-model settings, applied on top of the provider defaults:
//   FINBENCH_MODEL_OPTIONS_JSON  provider options for the model, e.g. the reasoning
//                                effort ({"reasoningEffort":"xhigh"} for OpenAI models)
//   FINBENCH_MODEL_LIMIT_JSON    context/output limits, e.g. {"context":200000,"output":32768}
//   finbench_overrides.json      the same per model id, plus "opencode" keys merged into
//                                the top-level OpenCode config
import { existsSync } from "node:fs";
const OVERRIDES_FILE = process.env.FINBENCH_OVERRIDES_FILE
  ?? path.join(path.dirname(fileURLToPath(import.meta.url)), "finbench_overrides.json");
function loadOverrides(model: string): { options?: unknown; limit?: unknown; opencode?: Record<string, unknown> } {
  try {
    if (!existsSync(OVERRIDES_FILE)) return {};
    const all = JSON.parse(readFileSync(OVERRIDES_FILE, "utf-8"));
    return all[model] ?? {};
  } catch (e) {
    console.error(`[config] ignoring unreadable overrides file ${OVERRIDES_FILE}: ${e}`);
    return {};
  }
}
function modelBlock(model: string, defaultLimit?: { context: number; output: number }) {
  const block: Record<string, unknown> = { name: model };
  if (defaultLimit) block.limit = defaultLimit;
  const opt = process.env.FINBENCH_MODEL_OPTIONS_JSON;
  const lim = process.env.FINBENCH_MODEL_LIMIT_JSON;
  if (opt) block.options = JSON.parse(opt);
  if (lim) block.limit = JSON.parse(lim);
  const ov = loadOverrides(model);
  if (ov.options !== undefined) block.options = ov.options;
  if (ov.limit !== undefined) block.limit = ov.limit;
  return { [model]: block };
}

function buildProvider(provider: string, model: string) {
  if (provider === "azure") {
    const apiKey = process.env.AZURE_OPENAI_API_KEY;
    if (!apiKey) throw new Error("AZURE_OPENAI_API_KEY is not set");
    const resourceName = process.env.AZURE_OPENAI_RESOURCE;
    if (!resourceName) throw new Error("AZURE_OPENAI_RESOURCE is not set");
    return {
      azure: {
        npm: "@ai-sdk/azure",
        name: "Azure OpenAI",
        options: { resourceName, apiKey },
        models: modelBlock(model),
      },
    };
  }
  if (provider === "openai") {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error("OPENAI_API_KEY is not set");
    return {
      openai: {
        npm: "@ai-sdk/openai",
        name: "OpenAI (direct)",
        options: { apiKey },
        models: { [model]: { name: model } },
      },
    };
  }
  if (provider === "anthropic") {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) throw new Error("ANTHROPIC_API_KEY is not set");
    return {
      anthropic: {
        npm: "@ai-sdk/anthropic",
        name: "Anthropic (direct)",
        options: { apiKey },
        models: modelBlock(model),
      },
    };
  }
  const oc = OPENAI_COMPATIBLE[provider];
  if (oc) {
    const apiKey = process.env[oc.env];
    if (!apiKey) throw new Error(`${oc.env} is not set`);
    return {
      [provider]: {
        npm: "@ai-sdk/openai-compatible",
        name: oc.name,
        options: { baseURL: oc.baseURL, apiKey },
        models: (function () {
          const perModel: Record<string, { context: number; output: number }> = {
            "glm-4.5-air": { context: 131072, output: 98304 },
          };
          return modelBlock(model, perModel[model] || oc.limit);
        })(),
      },
    };
  }
  const known = ["azure", "openai", "anthropic", ...Object.keys(OPENAI_COMPATIBLE)];
  throw new Error(`unknown provider "${provider}" (expected ${known.join(" | ")})`);
}

export function buildConfig(
  ctx: RunContext,
  stage2OpenCodeDir: string,
  provider = "azure",
  model = "gpt-5.4",
) {
  return {
    $schema: "https://opencode.ai/config.json",
    ...(loadOverrides(model).opencode ?? {}),
    model: `${provider}/${model}`,
    provider: buildProvider(provider, model),
    mcp: {
      "financial-tools": {
        type: "local" as const,
        command: ["python3", "mcp_server.py"],
        cwd: stage2OpenCodeDir,
        timeout: 180000,
        environment: {
          TICKER: ctx.ticker,
          FISCAL_QUARTER: ctx.fiscalQuarter,
          CUTOFF_DATE: ctx.cutoffDate,
          START_DATE: ctx.startDate ?? "",
          EMBEDDING_SERVER_URL: process.env.EMBEDDING_SERVER_URL ?? "",
          NEWS_ABLATION: process.env.NEWS_ABLATION ?? "",
          FILINGS_ABLATION: process.env.FILINGS_ABLATION ?? "",
          ALLOW_NO_RESEARCH: process.env.ALLOW_NO_RESEARCH ?? "",
        },
      },
    },
  };
}
