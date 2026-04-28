import Link from "next/link";
import { listRuns } from "@/lib/runs";
import { RunCard } from "@/components/RunCard";

export const dynamic = "force-dynamic";

export default async function Page() {
  const runs = await listRuns();
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
      <p className="text-sm text-zinc-500 mb-4">
        发现 {runs.length} 次训练。最近修改在前。
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {runs.map((r) => (
          <RunCard key={r.id} run={r} />
        ))}
      </div>
    </main>
  );
}
