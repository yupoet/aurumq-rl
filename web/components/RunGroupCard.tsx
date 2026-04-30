import Link from "next/link";
import type { RunGroup } from "@/lib/runs-shared";

const fmt = new Intl.DateTimeFormat("zh-CN", {
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
});

export function RunGroupCard({ group }: { group: RunGroup }) {
  return (
    <details className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 group/run">
      <summary className="cursor-pointer list-none">
        <div className="flex items-baseline justify-between gap-3">
          <h3 className="font-mono text-sm">
            <span className="text-zinc-400 mr-1">▸</span>
            {group.parent}{" "}
            <span className="text-zinc-500 text-xs">
              ({group.children.length} runs)
            </span>
          </h3>
          <time className="text-xs text-zinc-500">
            {fmt.format(group.modifiedAt)}
          </time>
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5 text-[10px]">
          {group.children.map((c) => (
            <span
              key={c.id}
              className="px-1.5 py-0.5 rounded bg-zinc-100 dark:bg-zinc-900 font-mono"
            >
              {c.summary?.reward_type ?? c.id.split("/").slice(-1)[0]}
            </span>
          ))}
        </div>
      </summary>

      <ul className="mt-3 space-y-2 border-t border-zinc-200 dark:border-zinc-800 pt-3">
        {group.children.map((c) => (
          <li key={c.id}>
            <Link
              href={`/runs/${encodeURIComponent(c.id)}`}
              className="flex items-baseline justify-between gap-2 text-sm hover:bg-zinc-50 dark:hover:bg-zinc-900 rounded px-2 py-1"
            >
              <span className="font-mono text-xs">
                {c.id.split("/").slice(1).join("/") || c.id}
              </span>
              <span className="text-xs text-zinc-500">
                {c.summary?.algorithm} · {c.summary?.total_timesteps?.toLocaleString()} ·{" "}
                {c.summary?.reward_type ?? "—"}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </details>
  );
}
