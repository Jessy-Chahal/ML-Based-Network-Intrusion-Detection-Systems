"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from "recharts";

type JobType = "attack_evaluation" | "defense_evaluation";
type JobStatus = "queued" | "running" | "completed" | "failed";

type JobSummary = {
  id: string;
  type: JobType;
  status: JobStatus;
  createdAt: string;
  startedAt?: string;
  finishedAt?: string;
  exitCode?: number;
  logs: string[];
};

type AttackDatasetSummary = {
  dataset: string;
  overallPass: boolean;
  avgDropPP: number;
  worstDropPP: number;
  modelCount: number;
};

type DefenseDatasetSummary = {
  dataset: string;
  cleanBaselinePct: number;
  cleanAdversarialPct: number;
  targetRecoveryPP: number;
  avgRecoveryPP: number;
  metTargetCount: number;
  evaluatedCount: number;
};

type MetricsSummary = {
  generatedAt?: string;
  attackTargetMaxDropPP: number;
  attackDatasets: AttackDatasetSummary[];
  defenseDatasets: DefenseDatasetSummary[];
};

function formatDropLabel(value: number): string {
  if (value == null || Number.isNaN(value)) return "";
  return `${Number(value).toFixed(2)} pp`;
}

function formatPctLabel(value: number): string {
  if (value == null || Number.isNaN(value)) return "";
  return `${Number(value).toFixed(1)}%`;
}

function labelDrop(value: unknown): string {
  const n = typeof value === "number" ? value : Number(value);
  return formatDropLabel(n);
}

function labelPct(value: unknown): string {
  const n = typeof value === "number" ? value : Number(value);
  return formatPctLabel(n);
}

function getBadgeClass(status?: JobStatus): string {
  switch (status) {
    case "running":
      return "bg-amber-100 text-amber-700";
    case "completed":
      return "bg-emerald-100 text-emerald-700";
    case "failed":
      return "bg-rose-100 text-rose-700";
    case "queued":
      return "bg-sky-100 text-sky-700";
    default:
      return "bg-primary-soft text-primary";
  }
}

export default function Home() {
  const [jobs, setJobs] = useState<Record<JobType, JobSummary | null>>({
    attack_evaluation: null,
    defense_evaluation: null,
  });
  const [isSubmitting, setIsSubmitting] = useState<Record<JobType, boolean>>({
    attack_evaluation: false,
    defense_evaluation: false,
  });
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  const matrixRows = useMemo(
    () => [
      {
        pipeline: "Attack Evaluation",
        type: "attack_evaluation" as const,
        command: "python src/evaluate_adv_training_clean.py",
        scope: "Runs attack evaluation script from the project root.",
      },
      {
        pipeline: "Defense Evaluation",
        type: "defense_evaluation" as const,
        command: "python src/defense/evaluate_defense.py",
        scope: "Runs defense evaluation script from the project root.",
      },
    ],
    [],
  );

  async function loadMetrics() {
    setMetricsError(null);
    try {
      const response = await fetch("/api/metrics");
      if (!response.ok) {
        throw new Error("Failed to load metrics.");
      }

      const payload = (await response.json()) as {
        attackMetrics: {
          generated_at_utc?: string;
          datasets?: Record<
            string,
            {
              target_max_drop_pp?: number;
              overall_pass?: boolean;
              models?: Record<string, { accuracy_drop_pp?: number }>;
            }
          >;
        } | null;
        defenseMetrics: {
          datasets?: Array<{
            dataset: string;
            target_recovery_pp?: number;
            clean_detection?: {
              baseline_detection_rate?: number;
              adversarial_detection_rate?: number;
            };
            mutations?: Array<{
              skipped?: boolean;
              recovery_delta_pp?: number;
              meets_25pp_target?: boolean;
            }>;
          }>;
        } | null;
      };

      const attackDatasets: AttackDatasetSummary[] = [];
      const attackEntries = Object.entries(payload.attackMetrics?.datasets ?? {});
      let attackTargetMaxDropPP = 3;
      for (const [, d] of attackEntries) {
        if (typeof d.target_max_drop_pp === "number") {
          attackTargetMaxDropPP = d.target_max_drop_pp;
          break;
        }
      }

      for (const [dataset, data] of attackEntries) {
        const drops = Object.values(data.models ?? {})
          .map((model) => model.accuracy_drop_pp ?? 0)
          .filter((value) => Number.isFinite(value));
        const absDrops = drops.map((d) => Math.abs(d));
        const avgDropPP =
          absDrops.length > 0
            ? absDrops.reduce((sum, value) => sum + value, 0) / absDrops.length
            : 0;
        const worstDropPP = absDrops.length > 0 ? Math.max(...absDrops) : 0;

        attackDatasets.push({
          dataset,
          overallPass: Boolean(data.overall_pass),
          avgDropPP,
          worstDropPP,
          modelCount: drops.length,
        });
      }

      const defenseDatasets: DefenseDatasetSummary[] = (
        payload.defenseMetrics?.datasets ?? []
      ).map((entry) => {
        const evaluated = (entry.mutations ?? []).filter((m) => !m.skipped);
        const metTarget = evaluated.filter((m) => Boolean(m.meets_25pp_target));
        const avgRecoveryPP =
          evaluated.length > 0
            ? evaluated.reduce((sum, item) => sum + (item.recovery_delta_pp ?? 0), 0) /
              evaluated.length
            : 0;

        return {
          dataset: entry.dataset,
          cleanBaselinePct: (entry.clean_detection?.baseline_detection_rate ?? 0) * 100,
          cleanAdversarialPct:
            (entry.clean_detection?.adversarial_detection_rate ?? 0) * 100,
          targetRecoveryPP: entry.target_recovery_pp ?? 25,
          avgRecoveryPP,
          metTargetCount: metTarget.length,
          evaluatedCount: evaluated.length,
        };
      });

      setMetrics({
        generatedAt: payload.attackMetrics?.generated_at_utc,
        attackTargetMaxDropPP,
        attackDatasets,
        defenseDatasets,
      });
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "Unable to load metrics files.";
      setMetricsError(message);
    }
  }

  async function startJob(type: JobType) {
    setError(null);
    setIsSubmitting((prev) => ({ ...prev, [type]: true }));

    try {
      const response = await fetch("/api/jobs/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type }),
      });

      if (!response.ok) {
        throw new Error("Failed to start job.");
      }

      const created = (await response.json()) as {
        id: string;
        type: JobType;
        status: JobStatus;
        createdAt: string;
      };

      setJobs((prev) => ({
        ...prev,
        [type]: {
          ...created,
          logs: [],
        },
      }));
      await loadMetrics();
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "Unexpected error while starting job.";
      setError(message);
    } finally {
      setIsSubmitting((prev) => ({ ...prev, [type]: false }));
    }
  }

  useEffect(() => {
    void loadMetrics();
  }, []);

  useEffect(() => {
    const interval = window.setInterval(async () => {
      const currentJobs = Object.values(jobs).filter(
        (job): job is JobSummary => Boolean(job),
      );
      if (currentJobs.length === 0) {
        return;
      }

      const updates = await Promise.all(
        currentJobs.map(async (job) => {
          try {
            const [statusRes, resultsRes] = await Promise.all([
              fetch(`/api/jobs/${job.id}/status`),
              fetch(`/api/jobs/${job.id}/results`),
            ]);

            // If one of the endpoints no longer has this job (dev restart/HMR),
            // stop polling it and keep the UI responsive.
            if (statusRes.status === 404 || resultsRes.status === 404) {
              return { type: job.type, missing: true as const };
            }

            if (!statusRes.ok || !resultsRes.ok) {
              return null;
            }

            const statusData = (await statusRes.json()) as Omit<
              JobSummary,
              "logs"
            >;
            const resultData = (await resultsRes.json()) as Pick<
              JobSummary,
              "logs"
            >;

            return {
              ...statusData,
              logs: resultData.logs,
            } satisfies JobSummary;
          } catch {
            return null;
          }
        }),
      );

      const validUpdates = updates.filter(
        (item): item is JobSummary | { type: JobType; missing: true } =>
          item !== null,
      );
      if (validUpdates.length === 0) {
        return;
      }

      setJobs((prev) => {
        const next = { ...prev };
        for (const update of validUpdates) {
          if ("missing" in update) {
            next[update.type] = null;
            continue;
          }
          next[update.type] = update;
        }
        return next;
      });
    }, 1500);

    return () => window.clearInterval(interval);
  }, [jobs]);

  return (
    <div className="min-h-screen bg-background px-6 py-10 text-foreground md:px-10">
      <main className="mx-auto flex w-full max-w-6xl flex-col gap-8">
        <header className="rounded-2xl border border-surface-border bg-surface p-7 shadow-sm">
          <p className="text-sm font-medium text-muted">ML-Based NIDS</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight">
            Evaluation Dashboard
          </h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-muted">
            Frontend orchestration for attack and defense script execution. This
            only runs evaluations and does not train new models.
          </p>
        </header>

        {error ? (
          <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {error}
          </div>
        ) : null}

        <section className="grid gap-4 md:grid-cols-2">
          {matrixRows.map((row) => {
            const job = jobs[row.type];
            const status = job?.status ?? "queued";

            return (
              <article
                key={row.pipeline}
                className="rounded-2xl border border-surface-border bg-surface p-6 shadow-sm"
              >
                <div className="flex items-start justify-between gap-4">
                  <h2 className="text-lg font-semibold">{row.pipeline}</h2>
                  <span
                    className={`rounded-full px-3 py-1 text-xs font-semibold ${getBadgeClass(status)}`}
                  >
                    {job ? status : "idle"}
                  </span>
                </div>
                <p className="mt-4 text-sm text-muted">{row.scope}</p>
                <code className="mt-4 block rounded-xl border border-surface-border bg-slate-50 p-3 text-xs text-slate-700">
                  {row.command}
                </code>
                <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-600">
                  {job?.id ? <span>Job ID: {job.id}</span> : null}
                  {job?.exitCode !== undefined ? (
                    <span>Exit Code: {job.exitCode}</span>
                  ) : null}
                </div>
              </article>
            );
          })}
        </section>

        <section className="rounded-2xl border border-surface-border bg-surface p-6 shadow-sm">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <h3 className="text-lg font-semibold">Run Evaluations</h3>
              <p className="mt-1 text-sm text-muted">
                Trigger attack and defense scripts and monitor progress.
              </p>
            </div>
            <div className="flex gap-3">
              <button
                type="button"
                onClick={() => startJob("attack_evaluation")}
                disabled={isSubmitting.attack_evaluation}
                className="rounded-xl bg-primary px-4 py-2 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Start Attack Job
              </button>
              <button
                type="button"
                onClick={() => startJob("defense_evaluation")}
                disabled={isSubmitting.defense_evaluation}
                className="rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Start Defense Job
              </button>
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-surface-border bg-surface p-6 shadow-sm">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h3 className="text-lg font-semibold">Results Visualization</h3>
              <p className="mt-1 text-sm text-muted">
                Visual summary from `results/adv_training_clean_metrics.json` and
                `results/defense_metrics.json`.
              </p>
            </div>
            <button
              type="button"
              onClick={loadMetrics}
              className="rounded-xl border border-surface-border px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
            >
              Refresh Metrics
            </button>
          </div>

          {metricsError ? (
            <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {metricsError}
            </div>
          ) : null}

          <div className="mt-6 grid gap-4 md:grid-cols-3">
            <article className="rounded-xl border border-surface-border bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted">
                Attack Datasets
              </p>
              <p className="mt-2 text-2xl font-semibold">
                {metrics?.attackDatasets.length ?? 0}
              </p>
            </article>
            <article className="rounded-xl border border-surface-border bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted">
                Defense Datasets
              </p>
              <p className="mt-2 text-2xl font-semibold">
                {metrics?.defenseDatasets.length ?? 0}
              </p>
            </article>
            <article className="rounded-xl border border-surface-border bg-slate-50 p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted">
                Last Updated
              </p>
              <p className="mt-2 text-sm font-medium text-slate-700">
                {metrics?.generatedAt
                  ? new Date(metrics.generatedAt).toLocaleString()
                  : "Not available"}
              </p>
            </article>
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            <div className="rounded-xl border border-surface-border p-4">
              <h4 className="text-sm font-semibold">
                Adv training — clean test accuracy drop
              </h4>
              <p className="mt-1 text-xs leading-relaxed text-muted">
                Target: worst drop per dataset should stay ≤{" "}
                <span className="font-semibold text-foreground">
                  {metrics?.attackTargetMaxDropPP ?? 3} pp
                </span>{" "}
                (~≤{metrics?.attackTargetMaxDropPP ?? 3}% absolute). Anything above the red line
                fails that dataset.
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {(metrics?.attackDatasets ?? []).map((d) => (
                  <span
                    key={d.dataset}
                    className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                      d.overallPass
                        ? "bg-emerald-100 text-emerald-800"
                        : "bg-rose-100 text-rose-800"
                    }`}
                  >
                    {d.dataset}: {d.overallPass ? "pass" : "fail"}
                  </span>
                ))}
              </div>
              <div className="mt-4 h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={metrics?.attackDatasets ?? []}
                    margin={{ top: 28, right: 8, left: 8, bottom: 8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                    <XAxis dataKey="dataset" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                    <YAxis
                      tickLine={false}
                      axisLine={false}
                      tick={{ fontSize: 11 }}
                      label={{ value: "pp", angle: -90, position: "insideLeft", fontSize: 10 }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <ReferenceLine
                      y={metrics?.attackTargetMaxDropPP ?? 3}
                      stroke="#dc2626"
                      strokeWidth={2}
                      strokeDasharray="6 4"
                      label={{
                        value: `≤${metrics?.attackTargetMaxDropPP ?? 3} pp`,
                        position: "insideTopRight",
                        fill: "#dc2626",
                        fontSize: 11,
                      }}
                    />
                    <Bar
                      dataKey="avgDropPP"
                      fill="#2563eb"
                      name="Avg |drop| (pp)"
                      radius={[6, 6, 0, 0]}
                    >
                      <LabelList
                        dataKey="avgDropPP"
                        position="top"
                        formatter={labelDrop}
                        style={{ fontSize: 10, fill: "#334155" }}
                      />
                    </Bar>
                    <Bar
                      dataKey="worstDropPP"
                      fill="#7c3aed"
                      name="Worst |drop| (pp)"
                      radius={[6, 6, 0, 0]}
                    >
                      <LabelList
                        dataKey="worstDropPP"
                        position="top"
                        formatter={labelDrop}
                        style={{ fontSize: 10, fill: "#334155" }}
                      />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="rounded-xl border border-surface-border p-4">
              <h4 className="text-sm font-semibold">Defense — clean attack detection rate</h4>
              <p className="mt-1 text-xs leading-relaxed text-muted">
                On unperturbed attack samples: both ensembles should stay near 100%.
              </p>
              <p className="mt-1 text-xs leading-relaxed text-muted">
                Mutation recovery target in{" "}
                <code className="rounded bg-slate-100 px-1 py-0.5 text-[11px] text-slate-700">
                  defense_metrics.json
                </code>{" "}
                is{" "}
                <span className="font-semibold text-foreground">
                  ≥{" "}
                  {(metrics?.defenseDatasets ?? [])[0]?.targetRecoveryPP ?? 25} pp
                </span>{" "}
                detection-rate lift for a mutation to count as meeting the target.
              </p>
              <p className="mt-1 text-xs leading-relaxed text-muted">
                Summary:{" "}
                {(metrics?.defenseDatasets ?? [])
                  .map((d) => `${d.dataset} ${d.metTargetCount}/${d.evaluatedCount} met`)
                  .join(" · ") || "—"}
                .
              </p>
              <div className="mt-4 h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={metrics?.defenseDatasets ?? []}
                    margin={{ top: 28, right: 8, left: 8, bottom: 8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                    <XAxis dataKey="dataset" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                    <YAxis
                      tickLine={false}
                      axisLine={false}
                      domain={[0, 100]}
                      tick={{ fontSize: 11 }}
                      label={{ value: "%", angle: -90, position: "insideLeft", fontSize: 10 }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <ReferenceLine
                      y={100}
                      stroke="#94a3b8"
                      strokeDasharray="4 4"
                      label={{ value: "100%", position: "insideTopRight", fontSize: 10, fill: "#64748b" }}
                    />
                    <Bar
                      dataKey="cleanBaselinePct"
                      fill="#94a3b8"
                      name="Baseline (%)"
                      radius={[6, 6, 0, 0]}
                    >
                      <LabelList
                        dataKey="cleanBaselinePct"
                        position="top"
                        formatter={labelPct}
                        style={{ fontSize: 10, fill: "#334155" }}
                      />
                    </Bar>
                    <Bar
                      dataKey="cleanAdversarialPct"
                      fill="#059669"
                      name="Adversarial-trained (%)"
                      radius={[6, 6, 0, 0]}
                    >
                      <LabelList
                        dataKey="cleanAdversarialPct"
                        position="top"
                        formatter={labelPct}
                        style={{ fontSize: 10, fill: "#334155" }}
                      />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-surface-border bg-surface p-6 shadow-sm">
          <h3 className="text-lg font-semibold">Live Logs</h3>
          <p className="mt-1 text-sm text-muted">
            Latest output from running evaluation jobs.
          </p>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            {(["attack_evaluation", "defense_evaluation"] as JobType[]).map(
              (type) => {
                const job = jobs[type];
                const label =
                  type === "attack_evaluation" ? "Attack Logs" : "Defense Logs";

                return (
                  <div
                    key={type}
                    className="rounded-xl border border-surface-border bg-slate-950 p-4"
                  >
                    <p className="text-sm font-semibold text-slate-100">{label}</p>
                    <pre className="mt-3 h-56 overflow-auto whitespace-pre-wrap text-xs leading-5 text-slate-300">
                      {job?.logs.length
                        ? job.logs.slice(-40).join("\n")
                        : "No logs yet."}
                    </pre>
                  </div>
                );
              },
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
