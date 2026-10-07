"use client";

import { Editor } from "@/components/Editor";
import { Phone } from "@/components/Phone";
import { useLang } from "@/lib/i18n";

export default function Page() {
  const { t } = useLang();
  return (
    <main className="stage">
      <Phone>
        <Editor />
      </Phone>
      <p className="caption">
        {t.title}
        <span>{t.lab}</span>
      </p>
    </main>
  );
}
