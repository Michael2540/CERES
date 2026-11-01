"use client";

import { useMission } from "@/lib/mission";

/** Global footer: model credit, image licence, and (in DEMO_MODE) a reminder that model results are recordings. */
export default function Footer() {
  const { link } = useMission();
  return (
    <footer className="border-t border-slate-800 px-3 py-3 text-xs text-slate-400 sm:px-4" data-testid="site-footer">
      <p data-testid="footer-built">Built with Muse Spark 1.3, SAM 3.1 (Meta Model API) · plant images: ALED, CC BY 4.0</p>
      {link?.demo_mode && (
        <p className="mt-0.5 text-violet-300" data-testid="footer-demo">
          DEMO MODE: no network, no key, no cost. Every model result is a RECORDED run, replayed and re-validated.
        </p>
      )}
    </footer>
  );
}
