"use client";

import type { EditorId } from "@/data/editors";
import { useLang, type Lang } from "@/lib/i18n";

export type SelectionMode = "tap" | "box" | "lasso" | "scissors" | "eraser";

export type EditorChoice = {
  id: EditorId;
  name: string;
  ready: boolean;
};

type Props = {
  open: boolean;
  onToggle: () => void;
  editors: EditorChoice[];
  editorId: EditorId;
  onEditor: (id: EditorId) => void;
  minArea: number;
  onMinArea: (value: number) => void;
  eraserSize: number;
  onEraserSize: (value: number) => void;
  selection: SelectionMode;
  onSelection: (mode: SelectionMode) => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  canDelete: boolean;
  onDelete: () => void;
};

export function Menu({
  open,
  onToggle,
  editors,
  editorId,
  onEditor,
  minArea,
  onMinArea,
  eraserSize,
  onEraserSize,
  selection,
  onSelection,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  canDelete,
  onDelete,
}: Props) {
  const { t, lang, setLang } = useLang();

  return (
    <div className="menu-root">
      <button
        type="button"
        className={open ? "hamburger open" : "hamburger"}
        aria-label={t.settings}
        aria-expanded={open}
        onClick={onToggle}
      >
        <span />
        <span />
        <span />
      </button>

      {open && (
        <div className="menu">
          <div className="undo-row">
            <button type="button" disabled={!canUndo} onClick={onUndo}>
              {t.undo}
            </button>
            <button type="button" disabled={!canRedo} onClick={onRedo}>
              {t.redo}
            </button>
            <button type="button" disabled={!canDelete} onClick={onDelete}>
              {t.delete}
            </button>
          </div>
          <label className="setting">
            <span className="setting-label">{t.language}</span>
            <select
              className="editor-select"
              value={lang}
              onChange={(event) => setLang(event.target.value as Lang)}
            >
              <option value="en">English</option>
              <option value="zh">中文</option>
            </select>
          </label>
          <label className="setting">
            <span className="setting-label">{t.editor}</span>
            <select
              className="editor-select"
              value={editorId}
              onChange={(event) => onEditor(event.target.value as EditorId)}
            >
              {editors.map((editor) => (
                <option key={editor.id} value={editor.id} disabled={!editor.ready}>
                  {editor.ready ? editor.name : t.noKey(editor.name)}
                </option>
              ))}
            </select>
          </label>

          <label className="setting">
            <span className="setting-label">
              {t.ignoreSmall}
              <span>{minArea === 0 ? t.off : t.underPx(minArea.toLocaleString())}</span>
            </span>
            <input
              type="range"
              min={0}
              max={30000}
              step={1000}
              value={minArea}
              onChange={(event) => onMinArea(Number(event.target.value))}
            />
          </label>

          <label className="setting">
            <span className="setting-label">
              {t.eraserSize}
              <span>{eraserSize} px</span>
            </span>
            <span className="eraser-row">
              <input
                type="range"
                min={4}
                max={40}
                step={2}
                value={eraserSize}
                onChange={(event) => onEraserSize(Number(event.target.value))}
              />
              <span className="eraser-preview" style={{ width: eraserSize, height: eraserSize }} />
            </span>
          </label>

          <div className="setting">
            <span className="setting-label">{t.selection}</span>
            <div className="picker" role="tablist">
              <button
                type="button"
                role="tab"
                aria-label={t.tap}
                title={t.tap}
                aria-selected={selection === "tap"}
                className={selection === "tap" ? "active" : ""}
                onClick={() => onSelection("tap")}
              >
                <TapIcon />
              </button>
              <button
                type="button"
                role="tab"
                aria-label={t.box}
                title={t.box}
                aria-selected={selection === "box"}
                className={selection === "box" ? "active" : ""}
                onClick={() => onSelection("box")}
              >
                <BoxIcon />
              </button>
              <button
                type="button"
                role="tab"
                aria-label={t.lasso}
                title={t.lasso}
                aria-selected={selection === "lasso"}
                className={selection === "lasso" ? "active" : ""}
                onClick={() => onSelection("lasso")}
              >
                <LassoIcon />
              </button>
              <button
                type="button"
                role="tab"
                aria-label={t.scissors}
                title={t.scissors}
                aria-selected={selection === "scissors"}
                className={selection === "scissors" ? "active" : ""}
                onClick={() => onSelection("scissors")}
              >
                <ScissorsIcon />
              </button>
              <button
                type="button"
                role="tab"
                aria-label={t.eraser}
                title={t.eraser}
                aria-selected={selection === "eraser"}
                className={selection === "eraser" ? "active" : ""}
                onClick={() => onSelection("eraser")}
              >
                <EraserIcon />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function TapIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <path
        d="M6.5 3.5 L6.5 16.2 L10 13.2 L13.2 19.2 L15.4 18.2 L12.2 12.2 L17 12.2 Z"
        fill="currentColor"
      />
    </svg>
  );
}

function BoxIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <rect
        x="4.5"
        y="5.5"
        width="15"
        height="13"
        rx="1.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeDasharray="3 2.2"
      />
    </svg>
  );
}

function LassoIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <path
        d="M9 15.2c-2.4-1.2-3.2-4.6-1.4-7 1.8-2.4 5.4-3 8-1.2 2.4 1.6 3 4.8 1.4 7.2-1.2 1.8-3.4 2.4-5.2 1.6"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <path
        d="M12 16.2c.6 1.6.4 3.2-.6 4.6"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}

function ScissorsIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <circle cx="6.5" cy="6.5" r="2.4" fill="none" stroke="currentColor" strokeWidth="1.7" />
      <circle cx="6.5" cy="17.5" r="2.4" fill="none" stroke="currentColor" strokeWidth="1.7" />
      <path
        d="M8.4 7.8 L19 17.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <path
        d="M8.4 16.2 L19 6.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}

function EraserIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <path
        d="M14.2 4.4a1.6 1.6 0 0 1 2.3 0l3.1 3.1a1.6 1.6 0 0 1 0 2.3L11.2 18.2H7.4L4.2 15a1.6 1.6 0 0 1 0-2.3Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinejoin="round"
      />
      <path d="M8.6 11.4 L12.8 15.6" fill="none" stroke="currentColor" strokeWidth="1.7" />
      <path d="M7 18.2 H16.5" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}
