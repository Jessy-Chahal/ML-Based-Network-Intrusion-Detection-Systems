import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

export type JobType = "attack_evaluation" | "defense_evaluation";
type JobStatus = "queued" | "running" | "completed" | "failed";

export type JobRecord = {
  id: string;
  type: JobType;
  status: JobStatus;
  createdAt: string;
  startedAt?: string;
  finishedAt?: string;
  command: string;
  args: string[];
  exitCode?: number;
  logs: string[];
};

const jobs = new Map<string, JobRecord>();
const MAX_LOG_LINES = 300;

function getRepoRoot(): string {
  const candidates = [
    process.cwd(),
    path.resolve(process.cwd(), ".."),
    path.resolve(process.cwd(), "../.."),
  ];

  for (const candidate of candidates) {
    const attackScript = path.join(
      candidate,
      "src",
      "evaluate_adv_training_clean.py",
    );
    if (fs.existsSync(attackScript)) {
      return candidate;
    }
  }

  return path.resolve(process.cwd(), "..");
}

function toWslPath(windowsPath: string): string {
  const normalized = windowsPath.replace(/\\/g, "/");
  const driveMatch = normalized.match(/^([A-Za-z]):\/(.*)$/);
  if (!driveMatch) {
    return normalized;
  }
  const drive = driveMatch[1].toLowerCase();
  const rest = driveMatch[2];
  return `/mnt/${drive}/${rest}`;
}

function resolveCommandAndArgs(
  repoRoot: string,
  scriptPath: string,
): { command: string; args: string[] } {
  const override = process.env.PYTHON_PATH;
  if (override) {
    return { command: override, args: [scriptPath] };
  }

  const windowsVenv = path.join(repoRoot, ".venv", "Scripts", "python.exe");
  if (fs.existsSync(windowsVenv)) {
    return { command: windowsVenv, args: [scriptPath] };
  }

  const unixVenv = path.join(repoRoot, ".venv", "bin", "python");
  if (fs.existsSync(unixVenv) && process.platform !== "win32") {
    return { command: unixVenv, args: [scriptPath] };
  }

  if (process.platform === "win32") {
    const repoRootWsl = toWslPath(repoRoot);
    const scriptWsl = toWslPath(scriptPath);
    const command = "wsl";
    const args = [
      "bash",
      "-lc",
      `cd "${repoRootWsl}" && if [ -x "${repoRootWsl}/.venv/bin/python" ]; then "${repoRootWsl}/.venv/bin/python" "${scriptWsl}"; else python "${scriptWsl}"; fi`,
    ];
    return { command, args };
  }

  return { command: "python", args: [scriptPath] };
}

function pushLog(job: JobRecord, message: string): void {
  job.logs.push(message);
  if (job.logs.length > MAX_LOG_LINES) {
    job.logs.splice(0, job.logs.length - MAX_LOG_LINES);
  }
}

function createJob(type: JobType): JobRecord {
  const id = randomUUID();
  const repoRoot = getRepoRoot();
  const scriptPath =
    type === "attack_evaluation"
      ? path.join(repoRoot, "src", "evaluate_adv_training_clean.py")
      : path.join(repoRoot, "src", "defense", "evaluate_defense.py");
  const runConfig = resolveCommandAndArgs(repoRoot, scriptPath);

  const job: JobRecord = {
    id,
    type,
    status: "queued",
    createdAt: new Date().toISOString(),
    command: runConfig.command,
    args: runConfig.args,
    logs: [],
  };

  jobs.set(id, job);
  return job;
}

function runJob(job: JobRecord): void {
  job.status = "running";
  job.startedAt = new Date().toISOString();

  const repoRoot = getRepoRoot();
  pushLog(job, `[info] Working directory: ${repoRoot}`);
  pushLog(job, `[info] Command strategy: ${job.command}`);
  pushLog(job, `[info] Starting: ${job.command} ${job.args.join(" ")}`);

  const child = spawn(job.command, job.args, {
    cwd: repoRoot,
    shell: false,
  });

  child.stdout.on("data", (chunk: Buffer) => {
    const output = chunk.toString().trimEnd();
    if (output) {
      pushLog(job, output);
    }
  });

  child.stderr.on("data", (chunk: Buffer) => {
    const output = chunk.toString().trimEnd();
    if (output) {
      pushLog(job, `[stderr] ${output}`);
    }
  });

  child.on("error", (error) => {
    job.status = "failed";
    job.finishedAt = new Date().toISOString();
    pushLog(job, `[error] Failed to start process: ${error.message}`);
  });

  child.on("close", (code) => {
    job.exitCode = code ?? -1;
    job.finishedAt = new Date().toISOString();
    job.status = code === 0 ? "completed" : "failed";
    pushLog(job, `[info] Process finished with exit code ${job.exitCode}`);
  });
}

export function startJob(type: JobType): JobRecord {
  const job = createJob(type);
  runJob(job);
  return job;
}

export function getJob(jobId: string): JobRecord | undefined {
  return jobs.get(jobId);
}

export function getAllJobs(): JobRecord[] {
  return [...jobs.values()].sort((a, b) =>
    b.createdAt.localeCompare(a.createdAt),
  );
}
