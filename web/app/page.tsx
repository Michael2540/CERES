"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const FIXTURES = process.env.NEXT_PUBLIC_FIXTURES === "1";

type Health = { status: string; demo_mode: boolean; edge_mode: string };

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const url = FIXTURES ? "/fixtures/health.json" : `${API}/health`;
    fetch(url)
      .then((r) => r.json())
      .then(setHealth)
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <main className="mx-auto max-w-3xl p-8">
      <h1 className="text-3xl font-semibold">CERES</h1>
      <p className="mt-1 text-sm text-slate-400">
        Space-farming crop monitor. Plant images are real (ALED, CC BY 4.0); sensors and faults are simulated.
      </p>
      <section className="mt-8 rounded-lg border border-slate-700 bg-slate-900 p-4" data-testid="health">
        <h2 className="text-sm uppercase tracking-wide text-slate-400">
          Backend health {FIXTURES ? "(fixtures)" : ""}
        </h2>
        {health && (
          <p className="mt-2 font-mono" data-testid="health-status">
            status={health.status} demo_mode={String(health.demo_mode)} edge_mode={health.edge_mode}
          </p>
        )}
        {error && <p className="mt-2 text-red-400">Backend unreachable: {error}</p>}
        {!health && !error && <p className="mt-2 text-slate-500">Loading...</p>}
      </section>
    </main>
  );
}
