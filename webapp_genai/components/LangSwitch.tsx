"use client";

import { useLang } from "@/lib/i18n";

export function LangSwitch() {
  const { lang, setLang, t } = useLang();

  return (
    <div className="lang-switch" role="group" aria-label={t.language} translate="no">
      <button type="button" className={lang === "zh" ? "on" : ""} onClick={() => setLang("zh")}>
        中文
      </button>
      <button type="button" className={lang === "en" ? "on" : ""} onClick={() => setLang("en")}>
        EN
      </button>
    </div>
  );
}
