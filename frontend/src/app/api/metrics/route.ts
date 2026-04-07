import fs from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";

async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export async function GET() {
  const repoRoot = path.resolve(process.cwd(), "..");
  const advPath = path.join(repoRoot, "results", "adv_training_clean_metrics.json");
  const defensePath = path.join(repoRoot, "results", "defense_metrics.json");

  const [attackMetrics, defenseMetrics] = await Promise.all([
    readJsonFile<Record<string, unknown>>(advPath),
    readJsonFile<Record<string, unknown>>(defensePath),
  ]);

  return NextResponse.json({
    attackMetrics,
    defenseMetrics,
    files: {
      attack: advPath,
      defense: defensePath,
    },
  });
}
