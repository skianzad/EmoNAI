"use client";

import { useEffect, useState } from "react";
import { AiEditApp } from "@/components/ai-edit/AiEditApp";
import { Editor } from "@/components/Editor";
import { LangSwitch } from "@/components/LangSwitch";
import { Phone } from "@/components/Phone";
import { RefsDemo } from "@/components/RefsDemo";
import { useLang } from "@/lib/i18n";
import "@/app/color-map/color-map.css";

const TABS = [
  { id: "tree", key: "approachTree", blurb: "approachBlurbTree" },
  { id: "islands", key: "approachIslands", blurb: "approachBlurbIslands" },
  { id: "controls", key: "approachControls", blurb: "approachBlurbControls" },
  { id: "sketch", key: "approachSketch", blurb: "approachBlurbSketch" },
  { id: "refs", key: "approachRefs", blurb: "approachBlurbRefs" },
] as const;

type Tab = (typeof TABS)[number]["id"];

export function Approaches() {
  const { lang, t } = useLang();
  const [tab, setTab] = useState<Tab>("controls");
  const [sketchOn, setSketchOn] = useState(false);
  const current = TABS.find((item) => item.id === tab);

  function pick(id: Tab) {
    setTab(id);
    if (id === "sketch") setSketchOn(true);
  }

  useEffect(() => {
    const frame = document.querySelector<HTMLIFrameElement>(".sd-frame");
    frame?.contentWindow?.postMessage({ type: "lang", lang }, window.location.origin);
  }, [lang, sketchOn, tab]);

  return (
    <div className="approaches">
      <aside className="approach-rail">
        <div className="approach-rail-head">
          <p className="approach-kicker">{t.lab}</p>
          <h1>{t.approachesLabel}</h1>
          <LangSwitch />
        </div>
        <nav className="approach-tabs" aria-label={t.approachesLabel}>
          {TABS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={tab === item.id ? "on" : ""}
              aria-current={tab === item.id ? "page" : undefined}
              onClick={() => pick(item.id)}
            >
              {t[item.key]}
            </button>
          ))}
        </nav>
        {current ? <p className="approach-blurb">{t[current.blurb]}</p> : null}
      </aside>
      <Phone>
        <div className="approaches-stage">
          <section className={tab === "tree" ? "on" : ""} hidden={tab !== "tree"}>
            <Editor />
          </section>
          <section className={tab === "islands" ? "on" : ""} hidden={tab !== "islands"}>
            <AiEditApp pack="islands" />
          </section>
          <section className={tab === "controls" ? "on" : ""} hidden={tab !== "controls"}>
            <AiEditApp pack="controls" />
          </section>
          <section className={tab === "sketch" ? "on" : ""} hidden={tab !== "sketch"}>
            {sketchOn ? (
              <iframe
                className="sd-frame"
                title={t.approachSketch}
                src="/sd/index.html"
                onLoad={(event) => {
                  event.currentTarget.contentWindow?.postMessage({ type: "lang", lang }, window.location.origin);
                }}
              />
            ) : (
              <p className="approach-wait">{t.approachSketch}</p>
            )}
          </section>
          <section className={tab === "refs" ? "on" : ""} hidden={tab !== "refs"}>
            <RefsDemo />
          </section>
        </div>
      </Phone>
    </div>
  );
}
