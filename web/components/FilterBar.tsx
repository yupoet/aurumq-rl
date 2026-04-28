"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

export interface FilterValues {
  q: string;
  algorithm: string;
  reward_type: string;
  universe_filter: string;
}

export function FilterBar({
  algorithms,
  rewardTypes,
  universes,
}: {
  algorithms: string[];
  rewardTypes: string[];
  universes: string[];
}) {
  const router = useRouter();
  const sp = useSearchParams();
  const [q, setQ] = useState(sp.get("q") ?? "");
  const [algo, setAlgo] = useState(sp.get("algo") ?? "");
  const [reward, setReward] = useState(sp.get("reward") ?? "");
  const [universe, setUniverse] = useState(sp.get("universe") ?? "");

  useEffect(() => {
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    if (algo) params.set("algo", algo);
    if (reward) params.set("reward", reward);
    if (universe) params.set("universe", universe);
    const qs = params.toString();
    router.replace(`/${qs ? `?${qs}` : ""}`);
  }, [q, algo, reward, universe, router]);

  return (
    <div className="flex flex-wrap items-center gap-2 mb-4">
      <input
        type="text"
        placeholder="Search id / reward…"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        className="text-sm rounded border border-zinc-300 dark:border-zinc-700 bg-transparent px-2 py-1 w-60"
      />
      <Chip label="algo" value={algo} options={algorithms} onChange={setAlgo} />
      <Chip label="reward" value={reward} options={rewardTypes} onChange={setReward} />
      <Chip
        label="universe"
        value={universe}
        options={universes}
        onChange={setUniverse}
      />
      {(q || algo || reward || universe) && (
        <button
          onClick={() => {
            setQ("");
            setAlgo("");
            setReward("");
            setUniverse("");
          }}
          className="text-xs text-zinc-500 hover:text-zinc-300 underline"
        >
          clear
        </button>
      )}
    </div>
  );
}

function Chip({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="text-xs rounded border border-zinc-300 dark:border-zinc-700 bg-transparent px-2 py-1"
    >
      <option value="">{label}: all</option>
      {options.map((o) => (
        <option key={o} value={o}>
          {label}={o}
        </option>
      ))}
    </select>
  );
}

export function applyFilters(
  runs: { id: string; summary: { algorithm?: string; reward_type?: string; universe_filter?: string } | null }[],
  filters: FilterValues
) {
  const q = filters.q.toLowerCase();
  return runs.filter((r) => {
    if (q && !r.id.toLowerCase().includes(q) && !(r.summary?.reward_type ?? "").toLowerCase().includes(q)) {
      return false;
    }
    if (filters.algorithm && r.summary?.algorithm !== filters.algorithm) return false;
    if (filters.reward_type && r.summary?.reward_type !== filters.reward_type) return false;
    if (filters.universe_filter && r.summary?.universe_filter !== filters.universe_filter) return false;
    return true;
  });
}
