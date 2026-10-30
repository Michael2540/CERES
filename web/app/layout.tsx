import type { Metadata } from "next";
import "./globals.css";
import AgentPanel from "@/components/AgentPanel";
import Header from "@/components/Header";
import { MissionProvider } from "@/lib/mission";

export const metadata: Metadata = {
  title: "CERES",
  description: "Space-farming crop monitor",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">
        <MissionProvider>
          <Header />
          <main className="px-3 py-4 sm:px-4">{children}</main>
          <AgentPanel />
        </MissionProvider>
      </body>
    </html>
  );
}
