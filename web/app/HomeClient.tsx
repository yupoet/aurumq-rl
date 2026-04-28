"use client";

import { useSearchParams } from "next/navigation";
import { useMemo, Suspense } from "react";
import type { RunListEntry } from "@/lib/runs";
import { groupRuns } from "@/lib/runs";
import { RunCard } from "@/components/RunCard";
import { RunGroupCard } from "@/components/RunGroupCard";
import { FilterBar, applyFilters } from "@/components/FilterBar";

interface Props {
  runs: RunListEntry[];
  algorithms: string[];
  rewardTypes: string[];
  universes: string[];
}

function Inner({ runs, algorithms, rewardTypes, universes }: Props) {
  const sp = useSearchParams();
  const filters = useMemo(() => ({
    q: sp.get("q") ?? "",
    algorithm: sp.get("algo") ?? "",
    reward_type: sp.get("reward") ?? "",
    universe_filter: sp.get("universe") ?? "",
  }), [sp]);

  const visible = useMemo(() => applyFilters(runs, filters), [runs, filters]);
  const grouped = useMemo(() => groupRuns(visible), [visible]);

  return (
    <>
      <FilterBar
        algorithms={algorithms}
        rewardTypes={rewardTypes}
        universes={universes}
      />
      <p className="text-sm text-zinc-500 mb-4">
        {visible.length} / {runs.length} 次训练 · {grouped.length} entries
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {grouped.map((g) =>
          g.kind === "single" ? (
            <RunCard key={g.run.id} run={g.run} />
          ) : (
            <RunGroupCard key={g.group.parent} group={g.group} />
          )
        )}
      </div>
    </>
  );
}

export default function HomeClient(props: Props) {
  return (
    <Suspense fallback={<div className="text-sm text-zinc-500">Loading…</div>}>
      <Inner {...props} />
    </Suspense>
  );
}
