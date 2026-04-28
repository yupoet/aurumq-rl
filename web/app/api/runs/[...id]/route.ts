import { NextResponse } from "next/server";
import {
  readBacktest,
  readBacktestSeries,
  readMetricsJsonl,
  readSummary,
} from "@/lib/runs";

export const dynamic = "force-dynamic";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string[] }> }
) {
  const { id } = await params;
  const decoded = id.map((s) => decodeURIComponent(s)).join("/");

  const url = new URL(request.url);
  const part = url.searchParams.get("part");

  if (part === "metrics") {
    return NextResponse.json(await readMetricsJsonl(decoded));
  }
  if (part === "backtest") {
    return NextResponse.json(await readBacktest(decoded));
  }
  if (part === "summary") {
    return NextResponse.json(await readSummary(decoded));
  }
  if (part === "backtest_series") {
    return NextResponse.json(await readBacktestSeries(decoded));
  }

  const [summary, metrics, backtest] = await Promise.all([
    readSummary(decoded),
    readMetricsJsonl(decoded),
    readBacktest(decoded),
  ]);
  return NextResponse.json({ summary, metrics, backtest });
}
