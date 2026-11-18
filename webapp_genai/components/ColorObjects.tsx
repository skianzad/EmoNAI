"use client";

import { useEffect, useState } from "react";
import { handbag } from "@/data/edits";
import type { EditorId } from "@/data/editors";
import { objectsFromPaint, renderColorObjects, type ColorObject, type NamedColor } from "@/lib/colorAdjust";
import { fitImage, readDataUrl } from "@/lib/shrink";
import { splitIslands } from "@/lib/splitIslands";
import { useLang } from "@/lib/i18n";
import type { EditorChoice } from "./Menu";

export function ColorObjects() {
  const { t } = useLang();
  const [enabled, setEnabled] = useState(true);
  const [original, setOriginal] = useState(handbag.original);
  const [painted, setPainted] = useState<string | null>(null);
  const [showPainted, setShowPainted] = useState(false);
  const [objects, setObjects] = useState<ColorObject[]>([]);
  const [preview, setPreview] = useState<string | null>(null);
  const [editorId, setEditorId] = useState<EditorId>("nano_banana");
  const [editors, setEditors] = useState<EditorChoice[]>([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");

  useEffect(() => {
    fetch("/api/editors")
      .then((response) => response.json())
      .then((list: EditorChoice[]) => {
        if (!Array.isArray(list)) return;
        setEditors(list);
        setEditorId((current) => {
          if (list.some((item) => item.id === current && item.ready)) return current;
          return list.find((item) => item.ready)?.id ?? current;
        });
      })
      .catch(() => setStatus(t.couldNotLoadEditors));
  }, [t]);

  useEffect(() => {
    if (!enabled || objects.length === 0) {
      setPreview(null);
      return;
    }
    let cancel = false;
    renderColorObjects(original, objects)
      .then((url) => {
        if (!cancel) setPreview(url);
      })
      .catch((error) => {
        if (!cancel) setStatus(error instanceof Error ? error.message : t.drawFailed);
      });
    return () => {
      cancel = true;
    };
  }, [enabled, original, objects]);

  async function mapObjects() {
    setBusy(true);
      setStatus(t.coloring);
    setShowPainted(false);
    try {
      const blob = await fitImage(await fetch(original).then((response) => response.blob()));
      const dataUrl = await readDataUrl(blob);
      const [head, image] = dataUrl.split(",");
      const response = await fetch("/api/color-map", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          editor: editorId,
          image,
          mime: head.slice(5, head.indexOf(";")) || "image/jpeg",
        }),
      });
      const body = await response.json().catch(() => ({ error: t.editFailed }));
      if (!response.ok) throw new Error(body.error || t.editFailed);
      const nextPainted = String(body.image ?? "");
      const named = Array.isArray(body.items) ? (body.items as NamedColor[]) : [];
      const split = await splitIslands(dataUrl, nextPainted);
      const next = await objectsFromPaint(nextPainted, split.islands, named);
      setPainted(nextPainted);
      setObjects(next);
      setStatus(next.length === 0 ? t.noObjects : t.objectCount(next.length));
    } catch (error) {
      setStatus(error instanceof Error ? error.message : t.editFailed);
    } finally {
      setBusy(false);
    }
  }

  async function upload(file: File) {
    const blob = await fitImage(file);
    setOriginal(await readDataUrl(blob));
    setPainted(null);
    setObjects([]);
    setPreview(null);
    setShowPainted(false);
    setStatus("");
  }

  function patch(id: string, change: Partial<ColorObject>) {
    setShowPainted(false);
    setObjects((current) => current.map((object) => (object.id === id ? { ...object, ...change } : object)));
  }

  const photo = !enabled || showPainted ? (showPainted && painted ? painted : original) : preview || original;

  return (
    <div className="editor">
      <div className="title-block">
        <h1 className="title">{t.colorTitle}</h1>
        <p className="subtitle">{t.colorSubtitle}</p>
      </div>

      <label className="cmap-toggle">
        <span>{t.colorToggle}</span>
        <input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} />
      </label>

      {enabled ? (
        <label className="setting">
          <span className="setting-label">{t.editor}</span>
          <select
            className="editor-select"
            value={editorId}
            onChange={(event) => setEditorId(event.target.value as EditorId)}
          >
            {editors.map((editor) => (
              <option key={editor.id} value={editor.id} disabled={!editor.ready}>
                {editor.ready ? editor.name : t.noKey(editor.name)}
              </option>
            ))}
          </select>
        </label>
      ) : null}

      <div className="canvas">
        <img className="photo" src={photo} alt="" />
      </div>

      {enabled ? (
        <div className="sheet">
          <div className="upload-row">
            <button type="button" className="upload-btn" disabled={busy} onClick={() => void mapObjects()}>
              {busy ? t.coloring : t.colorObjects}
            </button>
          </div>
          <label className="upload-btn cmap-file">
            {t.upload}
            <input
              type="file"
              accept="image/*"
              hidden
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void upload(file);
                event.target.value = "";
              }}
            />
          </label>
          {painted ? (
            <button type="button" className="cmap-text" onClick={() => setShowPainted((on) => !on)}>
              {showPainted ? t.showAdjust : t.showMap}
            </button>
          ) : (
            <p className="hint">
              {t.colorHint}
            </p>
          )}
          <ul className="cmap-list">
            {objects.map((object) => (
              <li key={object.id} className="cmap-row">
                <span className="cmap-name">{object.name}</span>
                <label className="cmap-size">
                  {t.colorLabel}
                  <input
                    type="color"
                    value={toPicker(object.color)}
                    aria-label={`${object.name} color`}
                    onChange={(event) => patch(object.id, { color: event.target.value.toUpperCase() })}
                  />
                </label>
                <label className="cmap-size">
                  {t.sizeLabel}
                  <input
                    type="range"
                    min={40}
                    max={180}
                    step={5}
                    value={object.scale}
                    aria-label={`${object.name} size`}
                    onChange={(event) => patch(object.id, { scale: Number(event.target.value) })}
                  />
                  <span>{object.scale}%</span>
                </label>
              </li>
            ))}
          </ul>
          {status ? <p className="status">{status}</p> : null}
        </div>
      ) : (
        <p className="hint">{t.colorHintOff}</p>
      )}
    </div>
  );
}

function toPicker(color: string) {
  return /^#[0-9a-fA-F]{6}$/.test(color) ? color : "#888888";
}
