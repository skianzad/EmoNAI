"use client";

import { useLang } from "@/lib/i18n";

export function RefsDemo() {
  const { t } = useLang();

  return (
    <div className="refs-empty">
      <h1>{t.refsTitle}</h1>
      <p>{t.refsBody}</p>
    </div>
  );
}
