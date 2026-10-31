"use client";

import AlertFeed from "@/components/AlertFeed";
import { brainLabel, linkLabel } from "@/components/LinkStatus";
import { DrainBox, QueueView } from "@/components/Mars";
import { Panel } from "@/components/ui";
import { useMission } from "@/lib/mission";

export default function MarsPage() {
  const { link, linkError } = useMission();
  const brain = link ? brainLabel(link) : null;
  return (
    <div className="mx-auto max-w-5xl space-y-4">
      <h1 className="text-xl font-semibold">Mars mode</h1>
      <p className="text-sm text-slate-400">
        When the Earth link drops, the on-board edge model triages alerts and the work waits in a queue. When the link is back, you decide whether to
        send it to Earth, where a code-computed audit compares both answers.
      </p>
      <Panel title="Link">
        {linkError && <p className="text-sm text-red-400" role="alert">Link unavailable: {linkError}</p>}
        {link && brain && (
          <div className="space-y-1 text-sm text-slate-300" data-testid="mars-link">
            <p>
              <b data-testid="mars-link-state">{linkLabel(link)}</b> · simulated link · {brain.text}
              {brain.recorded && " · replayed from a recording, no model call"} · {link.queue_depth} waiting
            </p>
            <p className="text-xs text-slate-400" data-testid="mars-light-delay">{link.light_delay_note}</p>
            <p className="text-xs text-slate-500">Use Link control in the header to take the link down or add a delay.</p>
          </div>
        )}
      </Panel>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-4">
          <DrainBox />
          <QueueView />
        </div>
        <aside aria-label="Alerts">
          <AlertFeed title="Open alerts: queue for on-board triage" />
        </aside>
      </div>
    </div>
  );
}
