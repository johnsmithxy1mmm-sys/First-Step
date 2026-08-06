import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Portfolio risk — Hyperliquid',
  description: 'Pre-trade liquidation risk for a book of perpetual positions.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-neutral-50 text-neutral-900 antialiased">{children}</body>
    </html>
  );
}
