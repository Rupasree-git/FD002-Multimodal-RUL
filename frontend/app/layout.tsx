// frontend/app/layout.tsx
import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "C-MAPSS Intelligence | Predictive Engine Health Platform",
  description:
    "Trust-aware multimodal Remaining Useful Life (RUL) prediction and contradiction diagnostics for NASA C-MAPSS FD002.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable} dark`}>
      <body className="bg-[#0B0D10] text-[#E6EDF3] font-sans antialiased selection:bg-[#38BDF8]/20 selection:text-[#7DD3FC]">
        {children}
      </body>
    </html>
  );
}