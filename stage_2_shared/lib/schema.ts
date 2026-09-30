// Loads an instance's accounting schema (stage_1_schema_building/ticker_schemas_v4).

import { readFileSync } from "node:fs";
import path from "node:path";

function schemaDir(): string {
  return path.resolve(import.meta.dirname, "../../stage_1_schema_building", "ticker_schemas_v4");
}

const schemaCache = new Map<string, any>();

export function loadSchema(ticker: string, fiscalQuarter: string): any {
  const key = `${ticker.toUpperCase()}_${fiscalQuarter}`;
  if (schemaCache.has(key)) return schemaCache.get(key);
  const schemaPath = path.join(schemaDir(), `${ticker.toUpperCase()}_${fiscalQuarter}.json`);
  const schema = JSON.parse(readFileSync(schemaPath, "utf-8"));
  schemaCache.set(key, schema);
  return schema;
}
