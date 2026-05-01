import { NextResponse } from "next/server";
import {
  readBacktest,
  readBacktestSeries,
  readFactorImportance,
  readGpuJsonl,
  readMetricsJsonl,
  readSummary,
  tailMetricsJsonl,
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

  if (part === "stream") {
    const initialOffset = Number(url.searchParams.get("offset") ?? 0);
    return streamMetrics(decoded, initialOffset, request.signal);
  }

  if (part === "metrics") {
    return NextResponse.json(await readMetricsJsonl(decoded));
  }
  if (part === "backtest") {
    return NextResponse.json(await readBacktest(decoded));
  }
  if (part === "backtest_series") {
    return NextResponse.json(await readBacktestSeries(decoded));
  }
  if (part === "summary") {
    return NextResponse.json(await readSummary(decoded));
  }
  if (part === "gpu") {
    return NextResponse.json(await readGpuJsonl(decoded));
  }
  if (part === "factor-importance") {
    return NextResponse.json(await readFactorImportance(decoded));
  }

  const [summary, metrics, backtest, gpu] = await Promise.all([
    readSummary(decoded),
    readMetricsJsonl(decoded),
    readBacktest(decoded),
    readGpuJsonl(decoded),
  ]);
  return NextResponse.json({
    summary,
    metrics,
    backtest,
    gpu: gpu.length > 0 ? gpu : null,
  });
}

function streamMetrics(
  id: string,
  initialOffset: number,
  signal: AbortSignal
): Response {
  const encoder = new TextEncoder();
  let offset = initialOffset;
  let interval: ReturnType<typeof setInterval> | null = null;

  const stream = new ReadableStream({
    start(controller) {
      const sendInit = () => {
        controller.enqueue(encoder.encode(`event: open\ndata: {}\n\n`));
      };
      sendInit();

      const tick = async () => {
        try {
          const { rows, newOffset } = await tailMetricsJsonl(id, offset);
          offset = newOffset;
          for (const row of rows) {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify(row)}\n\n`)
            );
          }
        } catch (err) {
          controller.enqueue(
            encoder.encode(
              `event: error\ndata: ${JSON.stringify({
                message: String(err),
              })}\n\n`
            )
          );
        }
      };

      interval = setInterval(tick, 2000);
      signal.addEventListener("abort", () => {
        if (interval) clearInterval(interval);
        try {
          controller.close();
        } catch {
          // already closed
        }
      });
    },
    cancel() {
      if (interval) clearInterval(interval);
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      "Connection": "keep-alive",
    },
  });
}
