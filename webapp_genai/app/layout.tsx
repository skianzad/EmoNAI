import type { Metadata } from "next";
import type { ReactNode } from "react";
import { LangProvider } from "@/lib/i18n";
import "./globals.css";

export const metadata: Metadata = {
  title: "GenAI Image Editor",
  description: "Tap an island on the photo, then keep it or put it back.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <LangProvider>{children}</LangProvider>
      </body>
    </html>
  );
}
