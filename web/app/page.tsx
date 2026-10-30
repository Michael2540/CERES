"use client";

import AlertFeed from "@/components/AlertFeed";
import TrayCard from "@/components/TrayCard";
import { ErrorNote, Loading } from "@/components/ui";
import { api } from "@/lib/api";
import { useAsync } from "@/lib/mission";

const ORDER = ["A", "B", "C", "L"];

export default function Home() {
  const trays = useAsync((a) => api.trays(a), []);
  const items = [...(trays.data?.items ?? [])].sort((a, b) => ORDER.indexOf(a.id) - ORDER.indexOf(b.id));
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
      <section aria-label="Trays">
        <h1 className="mb-1 text-sm font-semibold uppercase tracking-wider text-slate-400">Trays</h1>
        <p className="mb-3 text-xs text-slate-500">
          Real ALED plant images; sensors and faults are simulated, and every card says which.
        </p>
        {trays.error && <ErrorNote error={trays.error} retry={trays.reload} />}
        {!trays.data && trays.loading && <Loading what="Loading trays" />}
        {trays.data && items.length === 0 && <p className="text-sm text-slate-500">No trays configured.</p>}
        <div className="grid gap-3 sm:grid-cols-2" data-testid="tray-grid">
          {items.map((t) => (
            <TrayCard key={t.id} tray={t} />
          ))}
        </div>
      </section>
      <aside aria-label="Alerts">
        <AlertFeed />
      </aside>
    </div>
  );
}
