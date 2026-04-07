import { NextResponse } from "next/server";
import { startJob } from "@/lib/jobs";

type StartJobBody = {
  type?: "attack_evaluation" | "defense_evaluation";
};

export async function POST(request: Request) {
  let body: StartJobBody = {};

  try {
    body = (await request.json()) as StartJobBody;
  } catch {
    body = {};
  }

  if (!body.type) {
    return NextResponse.json(
      { error: "Missing job type. Use attack_evaluation or defense_evaluation." },
      { status: 400 },
    );
  }

  const job = startJob(body.type);

  return NextResponse.json({
    id: job.id,
    type: job.type,
    status: job.status,
    createdAt: job.createdAt,
  });
}
