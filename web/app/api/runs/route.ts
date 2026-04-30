import { NextResponse } from "next/server";
import { listRuns } from "@/lib/runs";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json(await listRuns());
}
