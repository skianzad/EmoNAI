"use client";

import { useEffect, useRef, useState } from "react";
import { handbag } from "@/data/edits";
import { editorChoices, type EditorId } from "@/data/editors";
import { branchEndingAt, insertAfterBranch, moveVersion, removedIds, shownLayers, type DropPlace, type Layer } from "@/data/versions";
import { callEditor } from "@/lib/callEditor";
import { renderComposite } from "@/lib/composite";
import { eraseIslands } from "@/lib/erase";
import { keepInsideLasso } from "@/lib/lasso";
import { cutIslands } from "@/lib/cut";
import { downloadProject, packProject, unpackProject, type Snapshot } from "@/lib/project";
import { useLang } from "@/lib/i18n";
import { fitImage, readDataUrl } from "@/lib/shrink";
import { splitIslands } from "@/lib/splitIslands";
import { uid } from "@/lib/uid";
import { Menu, type EditorChoice, type SelectionMode } from "./Menu";
import { Photo } from "./Photo";
import { Versions } from "./Versions";

function handbagLayers(prompts: Record<string, string>): Layer[] {
  return handbag.layers.map((layer) => ({
    ...layer,
    prompt: prompts[layer.id] ?? layer.prompt,
    visible: true,
    edited: handbag.edited,
  }));
}

function handbagEnabled() {
  const flags: Record<string, boolean> = {};
  for (const layer of handbag.layers) {
    for (const island of layer.islands) flags[island.id] = true;
  }
  return flags;
}

export function Editor() {
  const { t, lang } = useLang();
  const fileRef = useRef<HTMLInputElement>(null);
  const photoRef = useRef<HTMLInputElement>(null);
  const [original, setOriginal] = useState<string | null>(handbag.original);
  const [photoName, setPhotoName] = useState(handbag.name);
  const [enabled, setEnabled] = useState<Record<string, boolean>>(handbagEnabled);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [layers, setLayers] = useState<Layer[]>(() =>
    handbagLayers({
      cafe: "Move the handbag from the chair to the table.",
      "cafe-people": "Remove the people in the back.",
    })
  );
  const [activeId, setActiveId] = useState(handbag.layers[0].id);
  const [aspect, setAspect] = useState(900 / 534);
  const [newPrompt, setNewPrompt] = useState("");
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [menuOpen, setMenuOpen] = useState(false);
  const [minArea, setMinArea] = useState(0);
  const [eraserSize, setEraserSize] = useState(14);
  const [selection, setSelection] = useState<SelectionMode>("tap");
  const [editorId, setEditorId] = useState<EditorId>("nano_banana");
  const [editors, setEditors] = useState<EditorChoice[]>(
    editorChoices.map((editor) => ({ ...editor, ready: false }))
  );
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [history, setHistory] = useState({ undo: false, redo: false });
  const undoStack = useRef<Snapshot[]>([]);
  const redoStack = useRef<Snapshot[]>([]);
  const promptMark = useRef<string | null>(null);

  const editorName = editors.find((item) => item.id === editorId)?.name ?? "Nano Banana";

  function shot(): Snapshot {
    return { original, name: photoName, layers, activeId, selectedIds: [...selectedIds] };
  }

  function markHistory() {
    setHistory({ undo: undoStack.current.length > 0, redo: redoStack.current.length > 0 });
  }

  function remember() {
    undoStack.current.push(shot());
    if (undoStack.current.length > 40) undoStack.current.shift();
    redoStack.current = [];
    markHistory();
  }

  function restore(entry: Snapshot) {
    setOriginal(entry.original);
    setPhotoName(entry.name);
    setLayers(entry.layers);
    setActiveId(entry.activeId);
    setSelectedIds(entry.selectedIds);
    promptMark.current = null;
  }

  function undo() {
    const entry = undoStack.current.pop();
    if (!entry) return;
    redoStack.current.push(shot());
    restore(entry);
    markHistory();
  }

  function redo() {
    const entry = redoStack.current.pop();
    if (!entry) return;
    undoStack.current.push(shot());
    restore(entry);
    markHistory();
  }

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const typing = event.target instanceof HTMLElement && event.target.closest("textarea, input");
      if (typing) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) redo();
        else undo();
        return;
      }
      if ((event.key === "Delete" || event.key === "Backspace") && selectedIds.length > 0 && !busy) {
        event.preventDefault();
        deleteSelected();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

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
    // Intentionally once on mount; status copy uses the language at that time.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (original !== handbag.original) return;
    const prompts: Record<string, string> = {
      cafe: t.demoPromptBag,
      "cafe-people": t.demoPromptPeople,
    };
    setLayers((current) =>
      current.map((layer) => (prompts[layer.id] ? { ...layer, prompt: prompts[layer.id] } : layer))
    );
  }, [lang, t, original]);

  function activate(layerId: string) {
    setActiveId(layerId);
  }

  function rememberIslands(islands: { id: string }[]) {
    setEnabled((current) => ({
      ...current,
      ...Object.fromEntries(islands.map((island) => [island.id, true])),
    }));
  }

  function selectIslands(ids: string[]) {
    const active = layers.find((layer) => layer.id === activeId);
    const allowed = new Set(active?.islands.filter((island) => island.area >= minArea).map((island) => island.id));
    setSelectedIds(ids.filter((id) => allowed.has(id)));
  }

  function chooseMinArea(value: number) {
    setMinArea(value);
    setSelectedIds((ids) =>
      ids.filter((id) =>
        layers.some((layer) => layer.islands.some((island) => island.id === id && island.area >= value))
      )
    );
  }

  function setPrompt(id: string, text: string) {
    if (promptMark.current !== id) {
      remember();
      promptMark.current = id;
    }
    setLayers((current) => current.map((layer) => (layer.id === id ? { ...layer, prompt: text } : layer)));
  }

  function deleteSelected() {
    if (!activeId || selectedIds.length === 0) return;
    remember();
    const remove = new Set(selectedIds);
    setLayers((current) =>
      current.map((layer) =>
        layer.id === activeId
          ? { ...layer, islands: layer.islands.filter((island) => !remove.has(island.id)) }
          : layer
      )
    );
    setSelectedIds([]);
    const count = remove.size;
    setStatus(t.removedRegions(count));
  }

  function deleteVersion(id: string) {
    remember();
    const remove = removedIds(layers, id);
    const next = layers.filter((layer) => !remove.has(layer.id));
    setLayers(next);
    setDrafts((current) => {
      const draftsNext = { ...current };
      for (const removed of remove) delete draftsNext[removed];
      return draftsNext;
    });
    setSelectedIds([]);
    if (remove.has(activeId)) activate(next[next.length - 1]?.id ?? "");
  }

  function reorder(dragId: string, overId: string, place: DropPlace) {
    remember();
    setLayers(moveVersion(layers, dragId, overId, place));
  }

  async function erase(points: { x: number; y: number }[]) {
    const layer = layers.find((item) => item.id === activeId);
    if (!layer || points.length === 0) return;
    const islands = await eraseIslands(layer.islands, points, eraserSize / 2, selectedIds);
    remember();
    const kept = new Set(islands.map((island) => island.id));
    setLayers((current) => current.map((item) => (item.id === layer.id ? { ...item, islands } : item)));
    setSelectedIds((ids) => ids.filter((id) => kept.has(id)));
  }

  async function keepLasso(points: { x: number; y: number }[]) {
    const layer = layers.find((item) => item.id === activeId);
    if (!layer || points.length < 3) return;
    const islands = await keepInsideLasso(layer.islands, points, selectedIds);
    if (!islands) {
      setStatus(t.drawLasso);
      return;
    }
    remember();
    const kept = new Set(islands.map((island) => island.id));
    setLayers((current) => current.map((item) => (item.id === layer.id ? { ...item, islands } : item)));
    setSelectedIds((ids) => ids.filter((id) => kept.has(id)));
    setStatus(t.keptLasso);
  }

  async function cutApart(points: { x: number; y: number }[]) {
    const layer = layers.find((item) => item.id === activeId);
    if (!layer || points.length < 2) return;
    const islands = await cutIslands(layer.islands, points, selectedIds);
    if (!islands) {
      setStatus(t.drawCut);
      return;
    }
    remember();
    rememberIslands(islands);
    const before = layer.islands.length;
    setLayers((current) => current.map((item) => (item.id === layer.id ? { ...item, islands } : item)));
    setSelectedIds([]);
    setStatus(t.cutInto(islands.length - before + 1));
  }

  function toggleVisible(id: string) {
    remember();
    setLayers((current) =>
      current.map((layer) => (layer.id === id ? { ...layer, visible: !layer.visible } : layer))
    );
  }

  async function replaceRoot(layer: Layer) {
    if (!original) {
      setStatus(t.uploadFirst);
      return;
    }
    setBusy(true);
    setStatus(t.running(editorName));
    try {
      const image = await callEditor(editorId, layer.prompt.trim(), original);
      const next = await splitIslands(original, image);
      remember();
      setLayers((current) =>
        current.map((item) =>
          item.id === layer.id ? { ...item, visible: true, edited: next.edited, islands: next.islands } : item
        )
      );
      rememberIslands(next.islands);
      activate(layer.id);
      setSelectedIds([]);
      setStatus("");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : t.editFailed);
    } finally {
      setBusy(false);
    }
  }

  async function followUp(parentId: string, prompt: string, replacing: string | null) {
    if (!original) return false;
    const parent = layers.find((layer) => layer.id === parentId);
    if (!parent) return false;

    setBusy(true);
    setStatus(replacing ? t.updating(editorName) : t.subprompting(editorName));
    try {
      const source = await renderComposite(original, branchEndingAt(layers, parentId), enabled, minArea);
      const image = await callEditor(editorId, prompt, source);
      const next = await splitIslands(source, image);

      const childId = replacing ?? uid();
      remember();
      setLayers((current) => {
        if (replacing) {
          return current.map((layer) =>
            layer.id === replacing
              ? { ...layer, prompt, parentId, visible: true, edited: next.edited, islands: next.islands }
              : layer
          );
        }
        const child: Layer = {
          id: childId,
          parentId,
          prompt,
          visible: true,
          edited: next.edited,
          islands: next.islands,
        };
        return insertAfterBranch(current, parentId, child);
      });
      rememberIslands(next.islands);
      activate(childId);
      setSelectedIds([]);
      setStatus("");
      return true;
    } catch (error) {
      setStatus(error instanceof Error ? error.message : t.editFailed);
      return false;
    } finally {
      setBusy(false);
    }
  }

  function runVersion(id: string) {
    const layer = layers.find((item) => item.id === id);
    const prompt = layer?.prompt.trim() ?? "";
    if (!layer || !prompt || busy) return;
    if (layer.parentId) void followUp(layer.parentId, prompt, layer.id);
    else void replaceRoot(layer);
  }

  async function addSubprompt(parentId: string) {
    const prompt = (drafts[parentId] ?? "").trim();
    if (!prompt || busy) return false;
    const ok = await followUp(parentId, prompt, null);
    if (ok) setDrafts((current) => ({ ...current, [parentId]: "" }));
    return ok;
  }

  async function addVersion() {
    const prompt = newPrompt.trim();
    if (!original) {
      setStatus(t.uploadFirst);
      return;
    }
    if (!prompt || busy) return;
    setBusy(true);
    setStatus(t.running(editorName));
    try {
      const image = await callEditor(editorId, prompt, original);
      const next = await splitIslands(original, image);
      const layer: Layer = {
        id: uid(),
        parentId: null,
        prompt,
        visible: true,
        edited: next.edited,
        islands: next.islands,
      };
      remember();
      setLayers((current) => [...current, layer]);
      rememberIslands(next.islands);
      activate(layer.id);
      setSelectedIds([]);
      setNewPrompt("");
      setStatus("");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : t.editFailed);
    } finally {
      setBusy(false);
    }
  }

  async function uploadPhoto(file: File) {
    try {
      const fitted = await fitImage(file);
      const url = await readDataUrl(fitted);
      remember();
      setOriginal(url);
      setPhotoName(file.name.replace(/\.[^.]+$/, "") || t.title);
      setLayers([]);
      setActiveId("");
      setSelectedIds([]);
      setNewPrompt("");
      setDrafts({});
      setStatus(fitted === file ? "" : t.compressed(Math.round(fitted.size / 1024)));
    } catch (error) {
      setStatus(error instanceof Error ? error.message : t.couldNotReadPhoto);
    }
  }

  async function saveProcess() {
    if (!original) {
      setStatus(t.uploadBeforeSave);
      return;
    }
    setBusy(true);
    setStatus(t.saving);
    try {
      const file = await packProject({
        name: photoName,
        original,
        layers,
        activeId,
        selectedIds,
        minArea,
        drafts: Object.fromEntries(layers.flatMap((layer) => (drafts[layer.id] ? [[layer.id, drafts[layer.id]]] : []))),
        newPrompt,
        undo: undoStack.current,
        redo: redoStack.current,
      });
      downloadProject(file);
      const count = file.layers.length;
      setStatus(t.savedVersions(count));
    } catch (error) {
      setStatus(error instanceof Error ? error.message : t.couldNotSave);
    } finally {
      setBusy(false);
    }
  }

  async function openProcess(file: File) {
    setBusy(true);
    setStatus(t.opening);
    try {
      const project = unpackProject(await file.text());
      const before = shot();
      setOriginal(project.original);
      setPhotoName(project.name || t.title);
      setLayers(project.layers);
      setActiveId(project.activeId || project.layers[project.layers.length - 1]?.id || "");
      setSelectedIds(project.selectedIds ?? []);
      setMinArea(project.minArea ?? 0);
      setDrafts(project.drafts ?? {});
      setNewPrompt(project.newPrompt ?? "");
      undoStack.current = [before, ...(project.undo ?? [])];
      redoStack.current = project.redo ?? [];
      markHistory();
      setEnabled((current) => ({
        ...current,
        ...Object.fromEntries(project.layers.flatMap((layer) => layer.islands.map((island) => [island.id, true]))),
      }));
      const count = project.layers.length;
      setStatus(t.openedVersions(count));
    } catch (error) {
      setStatus(error instanceof Error ? error.message : t.couldNotOpen);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <div className="editor">
      <div className="title-row">
        <Menu
          open={menuOpen}
          onToggle={() => setMenuOpen((open) => !open)}
          editors={editors}
          editorId={editorId}
          onEditor={setEditorId}
          minArea={minArea}
          onMinArea={chooseMinArea}
          eraserSize={eraserSize}
          onEraserSize={setEraserSize}
          selection={selection}
          onSelection={setSelection}
          canUndo={history.undo}
          canRedo={history.redo}
          onUndo={undo}
          onRedo={redo}
          canDelete={selectedIds.length > 0}
          onDelete={deleteSelected}
        />
        <div className="title-block">
          <h1 className="title">{t.title}</h1>
          <p className="subtitle">{t.lab}</p>
        </div>
        <div className="title-actions">
          <button type="button" className="icon-btn" aria-label={t.openProcess} onClick={() => fileRef.current?.click()}>
            <FolderIcon />
          </button>
          <button type="button" className="icon-btn" aria-label={t.saveProcess} disabled={busy} onClick={() => void saveProcess()}>
            <SaveIcon />
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="application/json,.json"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void openProcess(file);
            }}
          />
        </div>
      </div>

      <div className="upload-row">
        <button type="button" className="upload-btn" onClick={() => photoRef.current?.click()}>
          {t.upload}
        </button>
        <input
          ref={photoRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void uploadPhoto(file);
            event.target.value = "";
          }}
        />
      </div>

      <div className="canvas" style={{ aspectRatio: aspect }}>
        {original ? (
          <Photo
            original={original}
            layers={shownLayers(layers, activeId)}
            enabled={enabled}
            selectedIds={selectedIds}
            minArea={minArea}
            selection={selection}
            eraserSize={eraserSize}
            activeId={activeId}
            onSelect={selectIslands}
            onErase={(points) => void erase(points)}
            onLasso={(points) => void keepLasso(points)}
            onCut={(points) => void cutApart(points)}
            onSize={(width, height) => setAspect(width / height)}
          />
        ) : (
          <button type="button" className="upload-empty" onClick={() => photoRef.current?.click()}>
            {t.upload}
          </button>
        )}
      </div>

      <div className="sheet">
        <Versions
          versions={layers}
          newPrompt={newPrompt}
          editorName={editorName}
          activeId={activeId}
          selectedIds={selectedIds}
          minArea={minArea}
          busy={busy}
          onNewPrompt={setNewPrompt}
          onAdd={addVersion}
          onDelete={deleteVersion}
          onReorder={reorder}
          onPrompt={setPrompt}
          onPromptDone={() => {
            promptMark.current = null;
          }}
          drafts={drafts}
          onDraft={(id, text) => setDrafts((current) => ({ ...current, [id]: text }))}
          onRun={runVersion}
          onSubprompt={addSubprompt}
          onActivate={(id) => {
            if (id !== activeId) setSelectedIds([]);
            activate(id);
          }}
          onVisible={toggleVisible}
          onSelect={(id) => selectIslands(selectedIds.includes(id) ? [] : [id])}
        />
        {status ? <p className="status">{status}</p> : null}
      </div>
    </div>
  );
}

function SaveIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <path d="M9 2.5v8" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M5.5 8 9 11.5 12.5 8" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3.5 14.5h11" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

function FolderIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <path
        d="M2.5 5.2c0-.7.6-1.2 1.3-1.2h3l1.2 1.4h6.2c.7 0 1.3.6 1.3 1.3v6.6c0 .7-.6 1.2-1.3 1.2H3.8c-.7 0-1.3-.5-1.3-1.2V5.2Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}
