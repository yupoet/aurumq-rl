import Link from "next/link";
import { listRuns } from "@/lib/runs";
import HomeClient from "./HomeClient";

export const dynamic = "force-dynamic";

export default async function Page() {
  const runs = await listRuns();
  const algorithms = uniq(runs.map((r) => r.summary?.algorithm).filter(Boolean) as string[]);
  const rewardTypes = uniq(runs.map((r) => r.summary?.reward_type).filter(Boolean) as string[]);
  const universes = uniq(runs.map((r) => r.summary?.universe_filter).filter(Boolean) as string[]);

  return (
    <main className="mx-auto max-w-6xl px-6 py-8">
      <header className="flex items-baseline justify-between mb-6">
        <h1 className="text-2xl font-semibold">AurumQ-RL Dashboard</h1>
        <Link
          href="/compare"
          className="text-sm rounded-lg border border-zinc-300 dark:border-zinc-700 px-3 py-1.5 hover:border-zinc-500"
        >
          对比 →
        </Link>
      </header>
      <HomeClient
        runs={runs}
        algorithms={algorithms}
        rewardTypes={rewardTypes}
        universes={universes}
      />
    </main>
  );
}

function uniq<T>(arr: T[]): T[] {
  return Array.from(new Set(arr));
}
