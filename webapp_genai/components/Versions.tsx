"use client";

import { useEffect, useRef, useState } from "react";
import type { DropPlace, Layer } from "@/data/versions";
import { useLang } from "@/lib/i18n";

type Props = {
  versions: Layer[];
  newPrompt: string;
  editorName: string;
  activeId: string;
  selectedIds: string[];
  minArea: number;
  busy: boolean;
  onNewPrompt: (text: string) => void;
  onAdd: () => void;
  onDelete: (id: string) => void;
  onReorder: (dragId: string, overId: string, place: DropPlace) => void;
  onPrompt: (id: string, text: string) => void;
  onPromptDone: () => void;
  drafts: Record<string, string>;
  onDraft: (id: string, text: string) => void;
  onRun: (id: string) => void;
  onSubprompt: (id: string) => Promise<boolean>;
  onActivate: (id: string) => void;
  onVisible: (id: string) => void;
  onSelect: (id: string) => void;
};

export function Versions(props: Props) {
  const { t } = useLang();
  const session = useRef<{ id: string; y: number; moved: boolean } | null>(null);
  const overRef = useRef<{ id: string; place: DropPlace } | null>(null);
  const [dragId, setDragId] = useState<string | null>(null);
  const [over, setOver] = useState<{ id: string; place: DropPlace } | null>(null);
  const [offset, setOffset] = useState(0);

  function pointerDown(id: string, _parentId: string | null, y: number) {
    session.current = { id, y, moved: false };
  }

  function pointerMove(y: number) {
    const current = session.current;
    if (!current) return;
    const delta = y - current.y;
    if (!current.moved && Math.abs(delta) < 8) return;
    current.moved = true;
    setDragId(current.id);
    setOffset(delta);
    let next: { id: string; place: DropPlace } | null = null;
    document.querySelectorAll<HTMLElement>("[data-version-id]").forEach((card) => {
      const id = card.dataset.versionId ?? "";
      if (!id || id === current.id || card.closest("[data-under-drag='true']")) return;
      const rect = card.getBoundingClientRect();
      if (y < rect.top || y > rect.bottom) return;
      const ratio = (y - rect.top) / Math.max(rect.height, 1);
      const place: DropPlace = ratio < 0.28 ? "before" : ratio > 0.72 ? "after" : "inside";
      next = { id, place };
    });
    overRef.current = next;
    setOver(next);
  }

  function pointerUp() {
    const current = session.current;
    const target = overRef.current;
    session.current = null;
    overRef.current = null;
    if (current?.moved && target) props.onReorder(current.id, target.id, target.place);
    setDragId(null);
    setOver(null);
    setOffset(0);
  }

  return (
    <section className="history">
      <h2 className="history-title">{t.history}</h2>
      <div className="composer">
        <textarea
          className="field"
          rows={2}
          placeholder={t.newPrompt}
          aria-label={t.newPrompt}
          value={props.newPrompt}
          onChange={(event) => props.onNewPrompt(event.target.value)}
        />
        <button
          type="button"
          className="plus"
          aria-label={t.addPrompt}
          disabled={props.busy || props.newPrompt.trim() === ""}
          onClick={props.onAdd}
        >
          +
        </button>
      </div>
      <ul className="tree tree-root">
        <Branch
          parentId={null}
          {...props}
          dragId={dragId}
          over={over}
          offset={offset}
          onPointerDownCard={pointerDown}
          onPointerMoveCard={pointerMove}
          onPointerUpCard={pointerUp}
        />
      </ul>
    </section>
  );
}

type DragProps = {
  dragId: string | null;
  over: { id: string; place: DropPlace } | null;
  offset: number;
  onPointerDownCard: (id: string, parentId: string | null, y: number) => void;
  onPointerMoveCard: (y: number) => void;
  onPointerUpCard: () => void;
};

function Branch({ parentId, versions, dragId, ...props }: Props & DragProps & { parentId: string | null }) {
  const nodes = versions.filter((version) => version.parentId === parentId).slice().reverse();
  if (nodes.length === 0) return null;
  return (
    <>
      {nodes.map((version) => (
        <li key={version.id} className="node" data-under-drag={dragId === version.id ? "true" : undefined}>
          <VersionCard version={version} versions={versions} dragId={dragId} {...props} />
          {versions.some((item) => item.parentId === version.id) ? (
            <ul className="tree">
              <Branch parentId={version.id} versions={versions} dragId={dragId} {...props} />
            </ul>
          ) : null}
        </li>
      ))}
    </>
  );
}

function VersionCard({
  version,
  versions,
  editorName,
  activeId,
  selectedIds,
  minArea,
  drafts,
  busy,
  onDelete,
  onPrompt,
  onPromptDone,
  onDraft,
  onRun,
  onSubprompt,
  onActivate,
  onVisible,
  onSelect,
  dragId,
  over,
  offset,
  onPointerDownCard,
  onPointerMoveCard,
  onPointerUpCard,
}: Props & DragProps & { version: Layer }) {
  const { t } = useLang();
  const draft = drafts[version.id] ?? "";
  const shown = version.islands.filter((island) => island.area >= minArea);
  const count = t.regions(shown.length);
  const active = version.id === activeId;
  const label = labelOf(versions, version.id);
  const [editing, setEditing] = useState(false);
  const [subpromptOpen, setSubpromptOpen] = useState(false);
  const fieldRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (editing) fieldRef.current?.focus();
  }, [editing]);

  const dragging = dragId === version.id;
  const parentKey = version.parentId ?? "";
  const place = over?.id === version.id ? over.place : null;

  return (
    <div
      className={["version", active ? "active" : "", version.visible ? "" : "hidden", dragging ? "dragging" : "", place ? `over-${place}` : ""].filter(Boolean).join(" ")}
      data-version-id={version.id}
      data-parent-id={parentKey}
      style={dragging ? { transform: `translateY(${offset}px)` } : undefined}
      onPointerDown={(event) => {
        const target = event.target as HTMLElement;
        if (target.closest("button, textarea, .prompt-text")) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        onPointerDownCard(version.id, version.parentId, event.clientY);
      }}
      onPointerMove={(event) => onPointerMoveCard(event.clientY)}
      onPointerUp={onPointerUpCard}
      onPointerCancel={onPointerUpCard}
    >
      <div className="version-head">
        <button
          type="button"
          className={active ? "mark on" : "mark"}
          aria-pressed={active}
          aria-label={active ? t.activeVersion : t.makeActive}
          onClick={() => onActivate(version.id)}
        />
        <button type="button" className="version-label" onClick={() => onActivate(version.id)}>
          {label}
        </button>
        <span className="editor-pill">{shortEditor(editorName)}</span>
        <button
          type="button"
          className={version.visible ? "eye on" : "eye"}
          aria-pressed={version.visible}
          aria-label={version.visible ? t.hideVersion : t.showVersion}
          onClick={() => onVisible(version.id)}
        >
          <Eye open={version.visible} />
        </button>
        <button type="button" className="delete" aria-label={t.deleteVersion(label)} onClick={() => onDelete(version.id)}>
          ×
        </button>
      </div>
      {editing ? (
        <textarea
          ref={fieldRef}
          className="field"
          rows={2}
          aria-label={t.promptFor(label)}
          value={version.prompt}
          onChange={(event) => onPrompt(version.id, event.target.value)}
          onBlur={() => {
            setEditing(false);
            onPromptDone();
          }}
        />
      ) : (
        <p
          className="prompt-text"
          onPointerDown={(event) => event.stopPropagation()}
          onDoubleClick={() => setEditing(true)}
        >
          {version.prompt}
        </p>
      )}
      <div className="action-row">
        <span className="region-count">{count}</span>
        {active ? (
          <div className="colors">
            {shown.map((island) => {
              const selected = selectedIds.includes(island.id);
              return (
                <button
                  key={island.id}
                  type="button"
                  className={selected ? "selected" : ""}
                  style={{ background: island.color }}
                  aria-label={t.changeRegion}
                  aria-pressed={selected}
                  onClick={() => onSelect(island.id)}
                />
              );
            })}
          </div>
        ) : null}
        <button
          type="button"
          className="update"
          disabled={busy || version.prompt.trim() === ""}
          onClick={() => onRun(version.id)}
        >
          {t.update}
        </button>
      </div>
      {subpromptOpen ? (
        <div className="follow">
          <textarea
            className="field"
            rows={2}
            placeholder={t.furtherChange}
            aria-label={t.furtherFor(label)}
            value={draft}
            onChange={(event) => onDraft(version.id, event.target.value)}
          />
          <div className="action-row">
            <span className="version-note">{t.sendsEdited}</span>
            <button
              type="button"
              className="subprompt"
              disabled={busy || draft.trim() === ""}
              onClick={() => {
                void onSubprompt(version.id).then((ok) => {
                  if (ok) setSubpromptOpen(false);
                });
              }}
            >
              <TurnArrow />
              {t.subprompt}
            </button>
          </div>
        </div>
      ) : (
        <button
          type="button"
          className="plus"
          aria-label={t.addSubprompt(label)}
          onClick={() => {
            onActivate(version.id);
            setSubpromptOpen(true);
          }}
        >
          +
        </button>
      )}
    </div>
  );
}

function TurnArrow() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
      <path
        d="M7 6v6.5a3 3 0 0 0 3 3H17"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <path
        d="M14 12.5 L17.5 15.5 L14 18.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function Eye({ open }: { open: boolean }) {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
      <path
        d="M2 12s3.8-6.2 10-6.2S22 12 22 12s-3.8 6.2-10 6.2S2 12 2 12z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
      />
      <circle cx="12" cy="12" r="2.3" fill="none" stroke="currentColor" strokeWidth="1.7" />
      {open ? null : <path d="M5 19 L19 5" fill="none" stroke="currentColor" strokeWidth="1.7" />}
    </svg>
  );
}

function shortEditor(name: string) {
  if (name.startsWith("GPT")) return "GPT";
  if (name.startsWith("Nano")) return "Nano";
  return name.split(" ")[0] || name;
}

function labelOf(versions: Layer[], id: string): string {
  const version = versions.find((item) => item.id === id);
  if (!version) return "";
  if (!version.parentId) {
    const roots = versions.filter((item) => item.parentId === null);
    return `v${roots.findIndex((item) => item.id === id) + 1}`;
  }
  const siblings = versions.filter((item) => item.parentId === version.parentId);
  return `${labelOf(versions, version.parentId)}.${siblings.findIndex((item) => item.id === id) + 1}`;
}
