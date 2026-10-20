"use client";

import { useEffect, useRef, useState, type PointerEvent } from "react";
import { useLang } from "@/lib/i18n";
import type { EditorId } from "@/data/editors";
import { callEditor } from "@/lib/callEditor";
import { loadSegments, resetRoomEdits, saveRoomBackground, saveSegments } from "@/lib/faceSegments";
import {
  fitEdited,
  generateColorVariants,
  generateEmptyBackground,
  isBackdrop,
  isFurnitureRecolor,
  pickVariantColors,
  renderLocal,
  segmentPhoto,
  segmentsFromIslands,
  tightenSegments,
  VARIANT_SHADES,
  type SegmentObject,
} from "@/lib/localEdit";
import { fitImage, readDataUrl } from "@/lib/shrink";
import { diffOutside, splitIslands } from "@/lib/splitIslands";
import type { Island } from "@/data/edits";
import { uid } from "@/lib/uid";
import type { EditorChoice } from "../Menu";
import "./ai-edit.css";

const SCENE = "/scenes/living/armchair.jpg";
const PORTRAIT = "/scenes/portrait/woman.jpg";

type Shot = { id: string; src: string; label: string; zh: string; kind: "room" | "face" | "new" };

const STARTER_SHOTS: Shot[] = [
  { id: "room", src: SCENE, label: "Living room", zh: "客厅", kind: "room" },
  { id: "face", src: PORTRAIT, label: "Portrait", zh: "人像", kind: "face" },
];

function editStyle(items: SegmentObject[]): "room" | "face" {
  const names = items.map((object) => object.name.toLowerCase()).join(" ");
  const face = (names.match(/face|lip|cheek|eye|forehead|portrait|person|woman|man|skin|head/g) ?? []).length;
  const room = (names.match(/sofa|chair|table|lamp|plant|room|rug|shelf|book|window|cup|bag|phone/g) ?? []).length;
  return face > room ? "face" : "room";
}

type Mode = "photo" | "home" | "object" | "voice" | "review" | "history" | "cleanup" | "retouch";
type ChangeIsland = Island & { on: boolean };
type ChangeSet = { before: string; edited: string; islands: ChangeIsland[] };

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not read the edited photo"));
    image.src = src;
  });
}

async function composeChanges(pack: ChangeSet, tintOn: boolean) {
  const [base, edited, ...masks] = await Promise.all([
    loadImage(pack.before),
    loadImage(pack.edited),
    ...pack.islands.map((island) => loadImage(island.mask)),
  ]);
  const width = base.naturalWidth;
  const height = base.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return pack.edited;
  ctx.drawImage(base, 0, 0, width, height);
  pack.islands.forEach((island, index) => {
    if (!island.on) return;
    const cut = document.createElement("canvas");
    cut.width = width;
    cut.height = height;
    const cutCtx = cut.getContext("2d");
    if (!cutCtx) return;
    cutCtx.drawImage(edited, 0, 0, width, height);
    cutCtx.globalCompositeOperation = "destination-in";
    cutCtx.drawImage(masks[index], 0, 0, width, height);
    ctx.drawImage(cut, 0, 0);
  });
  if (tintOn) {
    pack.islands.forEach((island, index) => {
      if (!island.on) return;
      const color = document.createElement("canvas");
      color.width = width;
      color.height = height;
      const colorCtx = color.getContext("2d");
      if (!colorCtx) return;
      colorCtx.drawImage(masks[index], 0, 0, width, height);
      colorCtx.globalCompositeOperation = "source-in";
      colorCtx.fillStyle = island.color;
      colorCtx.fillRect(0, 0, width, height);
      ctx.save();
      ctx.globalAlpha = 0.5;
      ctx.drawImage(color, 0, 0);
      ctx.restore();
    });
  }
  return canvas.toDataURL("image/jpeg", 0.9);
}
type Step = { id: string; label: string; meta: string; objects: SegmentObject[]; warmth: number; shown: boolean };
type FaceStep = { id: string; label: string; meta: string; objects: SegmentObject[]; eyes: number; lines: number; blur: number; shown: boolean };
type Palette = "ai" | "warm" | "cool";

const PALETTES: Record<Palette, { name: string; hex: string }[]> = {
  ai: [
    { name: "Terracotta", hex: "#c46a4a" },
    { name: "Caramel", hex: "#c4894a" },
    { name: "Olive", hex: "#6b7a45" },
    { name: "Slate", hex: "#5a6570" },
    { name: "Sage", hex: "#8a9a8a" },
    { name: "Charcoal", hex: "#3a3a3e" },
  ],
  // warm: [
  //   { name: "Red", hex: "#e23b3b" },
  //   { name: "Coral", hex: "#f07a5a" },
  //   { name: "Mustard", hex: "#d4a017" },
  //   { name: "Rust", hex: "#a65a32" },
  //   { name: "Wine", hex: "#7a3048" },
  //   { name: "Blush", hex: "#e8b4b8" },
  // ],
  // cool: [
  //   { name: "Navy", hex: "#2a4a7a" },
  //   { name: "Teal", hex: "#2a7a72" },
  //   { name: "Forest", hex: "#2f5a40" },
  //   { name: "Lilac", hex: "#b0a0c8" },
  //   { name: "Sky", hex: "#7ec8e8" },
  //   { name: "Graphite", hex: "#4a4e56" },
  // ],
  warm: [],
  cool: [],
};

const SHADES = [
  { label: "Deepest", zh: "最深", value: -2 },
  { label: "Deeper", zh: "更深", value: -1 },
  { label: "As picked", zh: "原色", value: 0 },
  { label: "Lighter", zh: "更浅", value: 1 },
];

function shadeColor(hex: string, shade: number) {
  const raw = hex.replace("#", "");
  const safe = raw.length === 6 ? raw : "888888";
  const channels = [0, 2, 4].map((start) => parseInt(safe.slice(start, start + 2), 16));
  const mix = (channel: number) => {
    const next = shade > 0 ? channel + (255 - channel) * shade * 0.18 : channel * (1 + shade * 0.22);
    return Math.max(0, Math.min(255, Math.round(next)));
  };
  return `#${channels.map((channel) => mix(channel).toString(16).padStart(2, "0")).join("")}`;
}

const OBJECT_NAMES: Record<string, string> = {
  sofa: "沙发",
  "left chair": "左扶手椅",
  "right chair": "右扶手椅",
  "left plant": "左侧植物",
  "right plant": "右侧植物",
  table: "茶几",
  lamp: "落地灯",
  ladder: "梯子",
  rug: "地毯",
  paintings: "挂画",
  painting: "挂画",
  bookshelf: "书架",
  plant: "植物",
  cushion: "靠垫",
  "floor lamp": "落地灯",
  "side table": "边桌",
  vase: "花瓶",
  forehead: "额头",
  "left cheek": "左脸颊",
  "right cheek": "右脸颊",
  "under eyes": "眼下",
  lips: "嘴唇",
  background: "背景",
};

function objectName(name: string, zh: boolean) {
  if (!zh) return name;
  return OBJECT_NAMES[name.toLowerCase()] ?? name;
}

export function AiEditApp() {
  const original = SCENE;
  const cache = useRef(new Map<string, SegmentObject[]>());
  const bgCache = useRef(new Map<string, string>());
  const [mode, setMode] = useState<Mode>("photo");
  const [source, setSource] = useState(original);
  const [working, setWorking] = useState(original);
  const workingRef = useRef(original);
  const [photo, setPhoto] = useState(original);
  const [shots, setShots] = useState<Shot[]>(STARTER_SHOTS);
  const [shotId, setShotId] = useState("room");
  const [objects, setObjects] = useState<SegmentObject[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [warmth, setWarmth] = useState(0);
  const [view, setView] = useState<string | null>(null);
  const [changes, setChanges] = useState<ChangeSet | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [editorId, setEditorId] = useState<EditorId>("nano_banana");
  /** Bumps when the empty-background fill becomes available so previews refresh. */
  const [fillTick, setFillTick] = useState(0);
  /** Full Nano Banana for heavy scene work; Lite for small segment tweaks / regenerations. */
  function editorFor(kind: "heavy" | "lite"): EditorId {
    if (editorId === "nano_banana" || editorId === "nano_banana_lite") {
      return kind === "heavy" ? "nano_banana" : "nano_banana_lite";
    }
    return editorId;
  }
  const [ratio, setRatio] = useState(16 / 9);
  const [palette, setPalette] = useState<Palette>("ai");
  const [prompt, setPrompt] = useState("");
  const [steps, setSteps] = useState<Step[]>([]);
  const [at, setAt] = useState(0);
  const [said, setSaid] = useState("");
  const [hold, setHold] = useState(false);
  const [cleanOn, setCleanOn] = useState<string[]>(["passer"]);
  const [weathered, setWeathered] = useState(true);
  const faceCache = useRef<SegmentObject[] | null>(null);
  const [faceSource, setFaceSource] = useState(PORTRAIT);
  const [faceObjects, setFaceObjects] = useState<SegmentObject[]>([]);
  const [faceView, setFaceView] = useState<string | null>(null);
  const [faceSelected, setFaceSelected] = useState<string | null>(null);
  const [faceSteps, setFaceSteps] = useState<FaceStep[]>([]);
  const [faceAt, setFaceAt] = useState(0);
  const [hist, setHist] = useState<"edit" | "face">("edit");
  const [historyOpen, setHistoryOpen] = useState(false);
  const faceLive = useRef<SegmentObject[]>([]);
  const lookLive = useRef({ eyes: 35, lines: 25, blur: 20 });
  const [preset, setPreset] = useState("Natural");
  const [eyes, setEyes] = useState(35);
  const [lines, setLines] = useState(25);
  const [blur, setBlur] = useState(20);
  const [light, setLight] = useState("Even");
  const drag = useRef<{ id: string; face: boolean; mode: "move" | "resize"; x: number; y: number; dx: number; dy: number; scale: number; base: SegmentObject[]; latest: SegmentObject[] } | null>(null);
  const photoRef = useRef<HTMLInputElement>(null);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const maskBits = useRef(new Map<string, { width: number; height: number; alpha: Uint8ClampedArray }>());
  const segmentKind = useRef<"objects" | "face" | "room">("room");
  const zoomRef = useRef({ scale: 1, x: 0, y: 0 });
  const pointers = useRef(new Set<number>());
  const [zoom, setZoom] = useState({ scale: 1, x: 0, y: 0 });
  const [listening, setListening] = useState(false);
  const [voiceDraft, setVoiceDraft] = useState("");
  const { lang, setLang } = useLang();
  const zh = lang === "zh";
  const tx = (en: string, cn: string) => (zh ? cn : en);

  const editing = mode !== "photo";
  const current = objects.find((object) => object.id === selected && !isBackdrop(object.name)) ?? null;
  // Background / backdrop segments are never selectable or boxed.
  const visible = objects.filter((object) => !object.hidden && !isBackdrop(object.name));

  useEffect(() => {
    loadSegments("face")
      .then((saved) => {
        if (saved && !faceCache.current) faceCache.current = structuredClone(saved.objects);
      })
      .catch(() => undefined);
    loadSegments("room")
      .then((saved) => {
        if (!saved) return;
        if (saved.background) bgCache.current.set(SCENE, saved.background);
        // Always load the saved base masks (from room-segments.json). Session edits are not on disk.
        if (saved.objects.length) {
          cache.current.set(SCENE, resetRoomEdits(saved.objects));
        }
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    fetch("/api/editors")
      .then((response) => response.json())
      .then((list: EditorChoice[]) => {
        const ready = Array.isArray(list) ? list.find((item) => item.ready) : null;
        if (ready) setEditorId(ready.id);
      })
      .catch(() => setError("Could not load editors."));
  }, []);

  useEffect(() => {
    if (mode === "photo" || mode === "retouch" || mode === "review" || mode === "voice" || objects.length === 0) return;
    let cancel = false;
    // Present the original working photo; fill vacated masks from the saved empty background.
    renderLocal(working, objects, warmth, bgCache.current.get(source) || null)
      .then((url) => {
        if (!cancel) setView(url);
      })
      .catch((reason) => {
        if (!cancel) setError(reason instanceof Error ? reason.message : "Could not update the photo");
      });
    return () => {
      cancel = true;
    };
  }, [mode, working, objects, warmth, source, fillTick]);

  /** Append edit / suggestion lines to the prompt (deduped, latest at the bottom). */
  function appendPromptLines(...incoming: string[]) {
    const additions = incoming.map((line) => line.trim()).filter(Boolean);
    if (!additions.length) return;
    setPrompt((prev) => {
      const existing = prev.split("\n").map((line) => line.trim()).filter(Boolean);
      const next = [...existing];
      for (const line of additions) {
        if (!next.includes(line)) next.push(line);
      }
      return next.join("\n");
    });
    requestAnimationFrame(() => {
      const field = promptRef.current;
      if (field) field.scrollTop = field.scrollHeight;
    });
  }

  // Manual recolor / move / scale land in the prompt. Suggestion chips (Del / Add / All)
  // stay local — they never push lines meant for Nano Banana.
  useEffect(() => {
    if (mode === "photo" || mode === "review" || mode === "voice" || busy) return;
    const items = mode === "retouch" ? (faceLive.current.length ? faceLive.current : faceObjects) : objects;
    const nextLight = mode === "retouch" ? light : undefined;
    const lines = editLines(
      items.filter((object) => {
        if (object.hidden || object.suggested) return false;
        // Baked sofa/chair packs are local switches — don't re-queue for Nano Banana.
        if (object.variants?.length && object.cover === 0) return false;
        return true;
      }),
      0,
      nextLight
    );
    appendPromptLines(...lines);
  }, [mode, objects, faceObjects, light, eyes, lines, blur, busy]);

  useEffect(() => {
    if (faceObjects.length === 0) return;
    let cancel = false;
    renderLocal(faceSource, faceObjects, 0)
      .then((url) => {
        if (!cancel) setFaceView(url);
      })
      .catch((reason) => {
        if (!cancel) setError(reason instanceof Error ? reason.message : "Could not update the portrait");
      });
    return () => {
      cancel = true;
    };
  }, [faceSource, faceObjects]);

  useEffect(() => {
    const frame = frameRef.current;
    const stage = stageRef.current;
    if (!frame || !stage) return;

    function commit(next: { scale: number; x: number; y: number }) {
      if (!stage) return;
      const scale = Math.min(6, Math.max(1, next.scale));
      const fitted = scale <= 1;
      const minX = stage.offsetWidth * (1 - scale);
      const minY = stage.offsetHeight * (1 - scale);
      const view = fitted
        ? { scale: 1, x: 0, y: 0 }
        : { scale, x: Math.min(0, Math.max(minX, next.x)), y: Math.min(0, Math.max(minY, next.y)) };
      zoomRef.current = view;
      setZoom(view);
    }

    function layoutOrigin() {
      if (!frame || !stage) return { left: 0, top: 0 };
      return {
        left: frame.getBoundingClientRect().left + stage.offsetLeft,
        top: frame.getBoundingClientRect().top + stage.offsetTop,
      };
    }

    function applyZoom(clientX: number, clientY: number, nextScale: number) {
      const current = zoomRef.current;
      const scale = Math.min(6, Math.max(1, nextScale));
      const origin = layoutOrigin();
      const ratio = scale / current.scale;
      commit({
        scale,
        x: (clientX - origin.left) * (1 - ratio) + current.x * ratio,
        y: (clientY - origin.top) * (1 - ratio) + current.y * ratio,
      });
    }

    function pan(dx: number, dy: number) {
      const current = zoomRef.current;
      if (current.scale <= 1) return;
      commit({ scale: current.scale, x: current.x - dx, y: current.y - dy });
    }

    let pinchScale = 1;
    let gesturing = false;
    let touchGap = 0;
    const fingers = new Map<number, { x: number; y: number }>();

    function onWheel(event: WheelEvent) {
      if (event.ctrlKey) {
        event.preventDefault();
        if (gesturing) return;
        applyZoom(event.clientX, event.clientY, zoomRef.current.scale * Math.exp(-event.deltaY * 0.01));
        return;
      }
      if (zoomRef.current.scale <= 1) return;
      event.preventDefault();
      pan(event.deltaX, event.deltaY);
    }
    function onGestureStart(event: Event) {
      event.preventDefault();
      gesturing = true;
      pinchScale = zoomRef.current.scale;
    }
    function onGestureChange(event: Event) {
      event.preventDefault();
      const gesture = event as GestureLike;
      applyZoom(gesture.clientX, gesture.clientY, pinchScale * gesture.scale);
    }
    function onGestureEnd() {
      gesturing = false;
    }
    function onTouchStart(event: TouchEvent) {
      if (event.touches.length < 2) return;
      event.preventDefault();
      drag.current = null;
      const [first, second] = [event.touches[0], event.touches[1]];
      touchGap = Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY);
      fingers.clear();
      for (const touch of Array.from(event.touches)) fingers.set(touch.identifier, { x: touch.clientX, y: touch.clientY });
    }
    function onTouchMove(event: TouchEvent) {
      if (event.touches.length < 2 || gesturing) return;
      event.preventDefault();
      const first = event.touches[0];
      const second = event.touches[1];
      const gap = Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY);
      const midX = (first.clientX + second.clientX) / 2;
      const midY = (first.clientY + second.clientY) / 2;
      if (touchGap > 0) applyZoom(midX, midY, zoomRef.current.scale * (gap / touchGap));
      touchGap = gap;
      const prevFirst = fingers.get(first.identifier);
      const prevSecond = fingers.get(second.identifier);
      if (prevFirst && prevSecond) {
        pan(-((first.clientX - prevFirst.x) + (second.clientX - prevSecond.x)) / 2, -((first.clientY - prevFirst.y) + (second.clientY - prevSecond.y)) / 2);
      }
      fingers.set(first.identifier, { x: first.clientX, y: first.clientY });
      fingers.set(second.identifier, { x: second.clientX, y: second.clientY });
    }
    function onTouchEnd() {
      fingers.clear();
      touchGap = 0;
    }

    frame.addEventListener("wheel", onWheel, { passive: false });
    frame.addEventListener("gesturestart", onGestureStart);
    frame.addEventListener("gesturechange", onGestureChange);
    frame.addEventListener("gestureend", onGestureEnd);
    frame.addEventListener("touchstart", onTouchStart, { passive: false });
    frame.addEventListener("touchmove", onTouchMove, { passive: false });
    frame.addEventListener("touchend", onTouchEnd);
    frame.addEventListener("touchcancel", onTouchEnd);
    return () => {
      frame.removeEventListener("wheel", onWheel);
      frame.removeEventListener("gesturestart", onGestureStart);
      frame.removeEventListener("gesturechange", onGestureChange);
      frame.removeEventListener("gestureend", onGestureEnd);
      frame.removeEventListener("touchstart", onTouchStart);
      frame.removeEventListener("touchmove", onTouchMove);
      frame.removeEventListener("touchend", onTouchEnd);
      frame.removeEventListener("touchcancel", onTouchEnd);
    };
  }, []);

  function snapshot(next: SegmentObject[], nextWarmth: number, label: string, meta: string) {
    const step: Step = { id: uid(), label, meta, objects: structuredClone(next), warmth: nextWarmth, shown: true };
    const history = [...steps.slice(0, at + 1), step];
    setSteps(history);
    setAt(history.length - 1);
    setObjects(next);
    setWarmth(nextWarmth);
  }

  /** Ensure a saved empty background exists for fill-on-remove. Never replaces the on-screen original. */
  async function ensureEmptyBackground(base: string, segments: SegmentObject[]) {
    const cached = bgCache.current.get(base);
    if (cached) {
      setFillTick((tick) => tick + 1);
      return cached;
    }
    const disk = await loadSegments("room");
    if (disk?.background) {
      bgCache.current.set(base, disk.background);
      setFillTick((tick) => tick + 1);
      return disk.background;
    }
    setSaid(tx("Generating empty background…", "正在生成空背景…"));
    const background = await generateEmptyBackground(editorFor("heavy"), base, segments);
    bgCache.current.set(base, background);
    await saveRoomBackground(background);
    setSaid("");
    setFillTick((tick) => tick + 1);
    return background;
  }

  async function openEdit() {
    setError("");
    const shot = shots.find((item) => item.id === shotId);
    segmentKind.current = shot?.kind === "room" || source === SCENE ? "room" : "objects";
    setMode("home");
    // Last-push presentation: keep showing the original photo + segment boxes.
    workingRef.current = source;
    setWorking(source);
    setPhoto(source);

    const memory = cache.current.get(source);
    if (memory?.length) {
      setBusy(true);
      try {
        const next = source === SCENE ? await tightenSegments(memory) : structuredClone(memory);
        cache.current.set(source, structuredClone(next));
        setObjects(next);
        setSelected(null);
        if (source === SCENE) void ensureEmptyBackground(source, next);
        if (steps.length === 0) {
          const count = next.filter((object) => !object.hidden).length;
          setSteps([
            {
              id: uid(),
              label: tx("Opened AI Edit", "打开 AI 编辑"),
              meta: tx(`Segmented ${count} objects`, `已分出 ${count} 个对象`),
              objects: structuredClone(next),
              warmth,
              shown: true,
            },
          ]);
          setAt(0);
        }
      } finally {
        setBusy(false);
      }
      return;
    }

    setBusy(true);
    try {
      const disk = source === SCENE ? await loadSegments("room") : null;
      if (disk?.background) bgCache.current.set(source, disk.background);
      // Prefer the pushed room-segments.json — never re-run color-map when it exists.
      // Session edits (move/remove/tint) stay in memory only.
      let raw: SegmentObject[];
      if (source === SCENE && disk?.objects?.length) {
        raw = resetRoomEdits(disk.objects);
      } else {
        raw = await segmentPhoto(editorId, source, source === SCENE ? "room" : "objects");
        if (source === SCENE && raw.length > 0) {
          // Creates room-segments.json only if missing; never overwrites the pushed file.
          await saveSegments("room", raw, bgCache.current.get(source) || undefined);
        }
      }
      const next = source === SCENE ? await tightenSegments(raw) : raw;
      if (source === SCENE) await ensureEmptyBackground(source, next);
      cache.current.set(source, structuredClone(next));
      setObjects(next);
      setSelected(null);
      setSteps([
        {
          id: uid(),
          label: tx("Opened AI Edit", "打开 AI 编辑"),
          meta: tx(`Segmented ${next.length} objects`, `已分出 ${next.length} 个对象`),
          objects: structuredClone(next),
          warmth: 0,
          shown: true,
        },
      ]);
      setAt(0);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not find objects");
      setMode("photo");
    } finally {
      setBusy(false);
    }
  }

  function keepRoom(next: SegmentObject[]) {
    // Session only — do not write move/remove/tint edits to disk.
    cache.current.set(source, structuredClone(next));
  }

  async function revealAdded(segment: SegmentObject, label: string) {
    const present = objects.some((object) => object.id === segment.id);
    const next = present
      ? objects.map((object) => (object.id === segment.id ? { ...object, hidden: false, suggested: true } : object))
      : [...objects, { ...segment, hidden: false, suggested: true }];
    keepRoom(next);
    snapshot(next, warmth, `Added ${label}`, "New segment");
    setSelected(segment.id);
    setMode("object");
  }

  async function addRegion(name: string) {
    const label = name.trim();
    if (!label || busy) return;

    // 1) Already in session with a colored layer — just reveal.
    const existing = objects.find((object) => object.name.toLowerCase() === label.toLowerCase());
    if (existing?.layer) {
      await revealAdded(existing, label);
      return;
    }

    setBusy(true);
    setError("");
    try {
      // 2) Load the pre-baked suggestion (mask + colored layer) from disk.
      if (source === SCENE) {
        const disk = await loadSegments("room");
        const saved = disk?.objects?.find((object) => object.name.toLowerCase() === label.toLowerCase());
        if (saved?.layer || saved?.mask) {
          const [ready] = resetRoomEdits([saved]);
          await revealAdded({ ...ready, hidden: false, layer: ready.layer || saved.layer }, label);
          return;
        }
      }

      // 3) Last resort: segment once, then save for next time.
      const guide = { name: label, left: 0.34, top: 0.52, width: 0.28, height: 0.18 };
      const found = await segmentPhoto(editorId, source, source === SCENE ? "room" : "objects", [guide]);
      const added = pickAdded(
        objects.filter((object) => !object.hidden),
        found,
        label
      );
      if (!added.length) {
        setError(tx(`Could not add ${label}. Try again.`, `无法添加${objectName(label, true)}，请重试。`));
        return;
      }
      if (source === SCENE) {
        await saveSegments(
          "room",
          [{ ...added[0], suggested: true, hidden: true } as SegmentObject & { suggested?: boolean }],
          bgCache.current.get(source)
        );
      }
      await revealAdded(added[0], label);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "Could not add that segment";
      setError(message === "fetch failed" ? "Could not reach the editor. Try Add again." : message);
    } finally {
      setBusy(false);
    }
  }

  async function realize(base: string, items: SegmentObject[], nextWarmth: number, kind: "objects" | "face" | "room", light?: string, instruction?: string) {
    const changes = editLines(items, nextWarmth, light);
    const request = (instruction ?? changes.join(" ")).trim();
    if (!request) {
      setError(tx("Change a color or an edit, then rerun.", "先改一个颜色或编辑，再生成。"));
      return;
    }
    setBusy(true);
    setError("");
    setSaid(request);
    setMode("voice");
    try {
      const preview = await renderLocal(kind === "face" ? base : workingRef.current || base, items, nextWarmth);

      // Sofa / chair recolor: one contact-sheet generation → 3 colors × 4 shades to switch locally.
      const furniture = items.filter(
        (object) =>
          !object.hidden &&
          object.tint &&
          isFurnitureRecolor(object.name) &&
          changes.some((line) => line.toLowerCase().includes(object.name.toLowerCase()))
      );
      if (kind !== "face" && furniture.length === 1 && changes.length === 1) {
        const target = furniture[0];
        const colors = pickVariantColors(target.tint!, PALETTES.ai);
        const shades = [...VARIANT_SHADES];
        const clean = items.map((object) =>
          object.id === target.id ? { ...object, tint: null, shade: 0, cover: undefined, layer: object.baseLayer || object.layer } : object
        );
        const sheetBase = await renderLocal(workingRef.current || base, clean, nextWarmth);
        const variants = await generateColorVariants(editorFor("heavy"), sheetBase, target, colors, shades, shadeColor);
        if (!variants.length) throw new Error(tx("Could not build color options.", "无法生成颜色选项。"));
        const picked =
          variants.find((item) => item.hex.toLowerCase() === target.tint!.toLowerCase() && item.shade === (target.shade || 0)) ||
          variants.find((item) => item.hex.toLowerCase() === target.tint!.toLowerCase()) ||
          variants[0];
        const next = items.map((object) =>
          object.id === target.id
            ? {
                ...object,
                baseLayer: object.baseLayer || object.layer,
                variants,
                layer: picked.layer,
                tint: picked.hex,
                shade: picked.shade,
                cover: 0,
              }
            : object
        );
        snapshot(next, nextWarmth, tx(`Color options for ${target.name}`, `${objectName(target.name, true)}的颜色选项`), request);
        cache.current.set(base, structuredClone(next));
        setSelected(target.id);
        setMode("object");
        return;
      }

      const unchanged = items
        .filter((object) => !object.hidden && !changes.some((line) => line.toLowerCase().includes(object.name.toLowerCase())))
        .map((object) => object.name);
      const promptText = [
        "This is the same photograph. Change only the objects named below. Do not invent new objects and do not restyle the room.",
        request,
        unchanged.length ? `Leave these objects unchanged: ${unchanged.join(", ")}.` : "",
        "Do not change the walls, floor, windows, lighting, shadows, camera, or framing.",
        "A flat color painted on an object is only the target color for that object. Repaint that object photorealistically in that exact color, and leave every other pixel identical.",
      ].filter(Boolean).join(" ");
      const image = await callEditor(editorFor("lite"), promptText, preview);
      const hinted = items.filter((object) => changes.some((line) => line.toLowerCase().includes(object.name.toLowerCase())));
      const colors = ["#e15b4c", "#3d7ee8", "#2f9d62", "#e0a23a", "#8b6ad6"];
      const split = hinted.length
        ? {
            edited: await fitEdited(preview, image),
            islands: [
              ...hinted.map((object, index) => ({
                id: uid(),
                label: object.name,
                mask: object.mask,
                color: colors[index % colors.length],
                area: Math.round(object.box.w * object.box.h * 1_000_000),
              })),
              ...await diffOutside(preview, image, items.map((object) => object.mask).filter(Boolean)),
            ],
          }
        : await splitIslands(preview, image);
      const added = (await segmentsFromIslands(split.edited, split.islands, hinted.length ? hinted : items)).map((segment) => {
        const { layer: _layer, ...rest } = segment;
        return rest;
      });
      if (added.length === 0) throw new Error(tx("Could not find the new edit.", "没有找到新的改动。"));
      const cleared = items.map((object) => {
        const { layer: _layer, ...rest } = object;
        return hinted.some((item) => item.id === object.id) ? { ...rest, tint: null, shade: 0, dx: 0, dy: 0, scale: 100, hidden: false } : rest;
      });
      const next = [...cleared, ...added];
      const pack: ChangeSet = {
        before: preview,
        edited: split.edited,
        islands: split.islands.map((island) => ({
          ...island,
          on: true,
          label: added.find((segment) => segment.id === island.id)?.name ?? island.label,
        })),
      };
      if (kind === "face") {
        faceLive.current = next;
        setFaceObjects(next);
        setFaceSteps([{ id: uid(), label: tx("Rerun with AI", "重新生成"), meta: request, objects: structuredClone(next), eyes, lines, blur, shown: true }]);
        setFaceAt(0);
        setFaceSelected(added[0]?.id ?? null);
        setMode("retouch");
        return;
      }
      const baked = await composeChanges({ ...pack, islands: pack.islands.map((island) => ({ ...island, on: true })) }, false);
      workingRef.current = baked;
      setWorking(baked);
      setPhoto(baked);
      snapshot(next, 0, tx("Rerun with AI", "重新生成"), request);
      cache.current.set(base, structuredClone(next));
      setSelected(added[0]?.id ?? null);
      setChanges(pack);
      setView(await composeChanges(pack, true));
      setMode("review");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Edit failed");
      setMode(kind === "face" ? "retouch" : "home");
    } finally {
      setBusy(false);
    }
  }

  function patch(id: string, change: Partial<SegmentObject>, label?: string) {
    const next = objects.map((object) => (object.id === id ? { ...object, ...change } : object));
    if (label) snapshot(next, warmth, label, next.find((object) => object.id === id)?.name ?? "Edit");
    else setObjects(next);
  }

  /** Prefer a baked AI variant when present; otherwise local tint overlay. */
  function applyObjectColor(id: string, hex: string | null, shade: number, label: string) {
    const object = objects.find((item) => item.id === id);
    if (!object) return;
    if (!hex) {
      patch(id, { tint: null, shade: 0, layer: object.baseLayer || object.layer, cover: undefined }, label);
      return;
    }
    const baked = object.variants?.find(
      (item) => item.hex.toLowerCase() === hex.toLowerCase() && item.shade === shade
    );
    if (baked) {
      patch(id, { tint: hex, shade, layer: baked.layer, cover: 0 }, label);
      return;
    }
    patch(id, { tint: hex, shade, cover: object.variants ? 0.35 : undefined }, label);
  }

  function useStep(index: number, list = steps) {
    const step = list[index];
    if (!step) return;
    setAt(index);
    setObjects(structuredClone(step.objects));
    setWarmth(step.warmth);
    if (mode === "review" || mode === "voice") setMode("home");
  }

  function toggleStep(index: number) {
    const turningOn = !steps[index]?.shown;
    const next = steps.map((step, i) => (i === index ? { ...step, shown: turningOn } : step));
    setSteps(next);
    if (turningOn) {
      useStep(index, next);
      return;
    }
    const earlier = next.slice(0, index).findLastIndex((step) => step.shown);
    if (earlier >= 0) {
      useStep(earlier, next);
      return;
    }
    const saved = cache.current.get(source);
    if (saved) {
      setObjects(structuredClone(saved));
      setWarmth(0);
    }
  }

  function dropStep(index: number) {
    const next = steps.filter((_, i) => i !== index);
    setSteps(next);
    if (next.length === 0) {
      setAt(0);
      return;
    }
    const nextAt = Math.min(index, next.length - 1);
    const visible = next[nextAt].shown ? nextAt : next.findLastIndex((step) => step.shown);
    if (visible >= 0) useStep(visible, next);
  }

  function selectShot(shot: Shot) {
    setShotId(shot.id);
    setSource(shot.src);
    setPhoto(shot.src);
    workingRef.current = shot.src;
    setWorking(shot.src);
    setView(null);
    setSelected(null);
    setHold(false);
    setError("");
    setMode("photo");
  }

  async function addShot(file: File) {
    const url = await readDataUrl(await fitImage(file));
    const added = shots.filter((shot) => shot.kind === "new").length + 1;
    const shot: Shot = { id: uid(), src: url, label: added === 1 ? "Photo" : `Photo ${added}`, zh: added === 1 ? "照片" : `照片 ${added}`, kind: "new" };
    setShots((list) => [...list, shot]);
    selectShot(shot);
  }

  function showFace(next: SegmentObject[]) {
    faceLive.current = next;
    setFaceObjects(next);
    setFaceSteps([{ id: uid(), label: "Opened retouch", meta: `Segmented ${next.length} regions`, objects: structuredClone(next), eyes, lines, blur, shown: true }]);
    setFaceAt(0);
    setFaceSelected(next.find((object) => !/background|backdrop/i.test(object.name))?.id ?? null);
    if (next.length === 0) setError("Could not find face regions");
  }

  async function openActive() {
    const shot = shots.find((item) => item.id === shotId) ?? shots[0];
    if (shot.kind === "face") {
      await openRetouch(shot.src);
      return;
    }
    if (shot.kind === "room") {
      await openEdit();
      return;
    }
    setError("");
    setBusy(true);
    try {
      const segmented = await segmentPhoto(editorId, shot.src, "objects");
      const kind = editStyle(segmented);
      setShots((list) => list.map((item) => (item.id === shot.id ? { ...item, kind, label: kind === "face" ? "Portrait" : item.label, zh: kind === "face" ? "人像" : item.zh } : item)));
      if (kind === "face") {
        setBusy(false);
        await openRetouch(shot.src);
        return;
      }
      segmentKind.current = "objects";
      workingRef.current = shot.src;
      setWorking(shot.src);
      setPhoto(shot.src);
      cache.current.set(shot.src, structuredClone(segmented));
      setObjects(segmented);
      setSelected(null);
      setSteps([{ id: uid(), label: tx("Opened AI Edit", "打开 AI 编辑"), meta: tx(`Segmented ${segmented.length} objects`, `已分出 ${segmented.length} 个对象`), objects: structuredClone(segmented), warmth: 0, shown: true }]);
      setAt(0);
      setMode("home");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not find objects");
      setMode("photo");
    } finally {
      setBusy(false);
    }
  }

  async function openRetouch(base = PORTRAIT) {
    setError("");
    setMode("retouch");
    setHold(false);
    if (base === faceSource && faceObjects.length > 0) return;
    setFaceSource(base);
    setFaceObjects([]);
    setFaceView(null);
    setFaceSteps([]);
    if (base === PORTRAIT && faceCache.current) {
      showFace(structuredClone(faceCache.current));
      return;
    }
    setBusy(true);
    try {
      const stored = base === PORTRAIT ? await loadSegments("face") : null;
      const next = stored?.objects ?? (await segmentPhoto(editorId, base, "face"));
      if (!stored && base === PORTRAIT && next.length > 0) saveSegments("face", next);
      if (base === PORTRAIT) faceCache.current = structuredClone(next);
      showFace(next);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not find face regions");
    } finally {
      setBusy(false);
    }
  }

  function rememberFace(next: SegmentObject[], label: string, meta: string, look = lookLive.current) {
    const step: FaceStep = { id: uid(), label, meta, objects: structuredClone(next), eyes: look.eyes, lines: look.lines, blur: look.blur, shown: true };
    const history = [...faceSteps.slice(0, faceAt + 1), step];
    faceLive.current = next;
    lookLive.current = look;
    setFaceSteps(history);
    setFaceAt(history.length - 1);
    setFaceObjects(next);
    setEyes(look.eyes);
    setLines(look.lines);
    setBlur(look.blur);
  }

  function tuneFace(nextEyes = eyes, nextLines = lines, nextBlur = blur) {
    const look = { eyes: nextEyes, lines: nextLines, blur: nextBlur };
    lookLive.current = look;
    setEyes(nextEyes);
    setLines(nextLines);
    setBlur(nextBlur);
    setFaceObjects((items) => {
      const next = paintFace(items, nextEyes, nextLines, nextBlur);
      faceLive.current = next;
      return next;
    });
  }

  function tintFace(hex: string | null) {
    const next = faceLive.current.map((object) => (object.id === faceSelected ? { ...object, tint: hex, cover: hex ? (object.cover ?? 0.5) : undefined, shade: 0 } : object));
    const name = next.find((object) => object.id === faceSelected)?.name ?? "Region";
    rememberFace(next, hex ? `${name} recolored` : `${name} · original`, "Color");
  }

  function coverFace(value: number) {
    setFaceObjects((items) => {
      const next = items.map((object) => (object.id === faceSelected ? { ...object, cover: value } : object));
      faceLive.current = next;
      return next;
    });
  }

  function useFace(index: number, list = faceSteps) {
    const step = list[index];
    if (!step) return;
    setFaceAt(index);
    faceLive.current = structuredClone(step.objects);
    lookLive.current = { eyes: step.eyes, lines: step.lines, blur: step.blur };
    setFaceObjects(structuredClone(step.objects));
    setEyes(step.eyes);
    setLines(step.lines);
    setBlur(step.blur);
  }

  function toggleFace(index: number) {
    const turningOn = !faceSteps[index]?.shown;
    const next = faceSteps.map((step, i) => (i === index ? { ...step, shown: turningOn } : step));
    setFaceSteps(next);
    if (turningOn) {
      useFace(index, next);
      return;
    }
    const earlier = next.slice(0, index).findLastIndex((step) => step.shown);
    if (earlier >= 0) useFace(earlier, next);
  }

  function dropFace(index: number) {
    const next = faceSteps.filter((_, i) => i !== index);
    setFaceSteps(next);
    if (next.length === 0) {
      setFaceAt(0);
      return;
    }
    const nextAt = Math.min(index, next.length - 1);
    const visible = next[nextAt].shown ? nextAt : next.findLastIndex((step) => step.shown);
    if (visible >= 0) useFace(visible, next);
  }

  async function done() {
    if (mode === "retouch") {
      setMode("home");
      setHold(false);
      return;
    }
    if (mode === "review" && changes) {
      const kept = new Set(changes.islands.filter((island) => island.on).map((island) => island.id));
      const next = objects.filter((object) => !object.layer || kept.has(object.id));
      setObjects(next);
      cache.current.set(source, structuredClone(next));
      const baked = await composeChanges({ ...changes, islands: changes.islands.filter((island) => island.on) }, false);
      workingRef.current = baked;
      setWorking(baked);
      setPhoto(baked);
      setChanges(null);
      setView(null);
      setHold(false);
      setWarmth(0);
      setSelected(next.find((object) => object.layer)?.id ?? null);
      setMode("home");
      return;
    }
    if (view) setPhoto(view);
    cache.current.set(source, structuredClone(objects));
    setMode("photo");
    setHold(false);
  }

  async function toggleIsland(id: string) {
    if (!changes) return;
    const next = {
      ...changes,
      islands: changes.islands.map((island) => (island.id === id ? { ...island, on: !island.on } : island)),
    };
    setChanges(next);
    setView(await composeChanges(next, true));
  }

  async function send(text: string) {
    const trimmed = text.trim();
    if (!trimmed) return;
    setSaid(trimmed);
    setPrompt("");
    const bag = objects.find((object) => /bag/i.test(object.name));
    if (bag && /red/i.test(trimmed)) {
      patch(bag.id, { tint: "#e23b3b", shade: 0 }, `“${trimmed}”`);
      setSelected(bag.id);
      setPalette("warm");
      setMode("object");
      return;
    }
    if (/evening|warmer|warm up/i.test(trimmed)) {
      snapshot(objects, 0.28, `“${trimmed}”`, "Warmth");
      setMode("home");
      return;
    }
    if (/cooler|cool down/i.test(trimmed)) {
      snapshot(objects, -0.28, `“${trimmed}”`, "Warmth");
      setMode("home");
      return;
    }
    const localItems = mode === "retouch" ? (faceLive.current.length ? faceLive.current : faceObjects) : objects;
    const localLines = editLines(localItems, mode === "retouch" ? 0 : warmth, mode === "retouch" ? light : undefined);
    if (localLines.length) {
      const kind = mode === "retouch" ? "face" : segmentKind.current === "face" ? "objects" : segmentKind.current;
      const base = mode === "retouch" ? faceSource : source;
      await realize(base, localItems, mode === "retouch" ? 0 : warmth, kind, mode === "retouch" ? light : undefined, trimmed);
      return;
    }
    setMode("voice");
    setBusy(true);
    try {
      const image = await callEditor(editorFor("lite"), trimmed, view || photo);
      const before = source;
      const split = await splitIslands(before, image);
      const pack: ChangeSet = {
        before,
        edited: split.edited,
        islands: split.islands.map((island, index) => ({ ...island, label: `${index + 1}`, on: true })),
      };
      setChanges(pack);
      setView(await composeChanges(pack, true));
      setMode("review");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Edit failed");
      setMode("home");
    } finally {
      setBusy(false);
    }
  }

  const faceOn = mode === "retouch" || (mode === "history" && hist === "face");
  const picture = faceOn ? (hold ? faceSource : faceView || faceSource) : hold ? source : editing && view ? view : photo;
  const faceMarks = faceObjects.filter((object) => !object.hidden && !/background|backdrop/i.test(object.name));

  useEffect(() => {
    let cancel = false;
    for (const object of [...objects, ...faceObjects, ...(changes?.islands ?? [])]) {
      if (!object.mask || maskBits.current.has(object.mask)) continue;
      const image = new Image();
      image.onload = () => {
        if (cancel) return;
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) return;
        ctx.drawImage(image, 0, 0);
        const pixels = ctx.getImageData(0, 0, image.width, image.height).data;
        const alpha = new Uint8ClampedArray(image.width * image.height);
        for (let i = 0; i < alpha.length; i += 1) alpha[i] = pixels[i * 4 + 3];
        maskBits.current.set(object.mask, { width: image.width, height: image.height, alpha });
      };
      image.src = object.mask;
    }
    return () => { cancel = true; };
  }, [objects, faceObjects, changes]);

  function islandAt(nx: number, ny: number, islands: ChangeIsland[]) {
    for (let index = islands.length - 1; index >= 0; index -= 1) {
      const island = islands[index];
      if (!island.on) continue;
      const bits = maskBits.current.get(island.mask);
      if (!bits) continue;
      const px = Math.floor(nx * bits.width);
      const py = Math.floor(ny * bits.height);
      if (px < 0 || py < 0 || px >= bits.width || py >= bits.height) continue;
      if (bits.alpha[py * bits.width + px] > 24) return island;
    }
    return null;
  }

  function onSegment(object: SegmentObject, nx: number, ny: number) {
    const frame = boxFrame(object);
    if (nx < frame.left || ny < frame.top || nx > frame.left + frame.width || ny > frame.top + frame.height) return false;
    const bits = maskBits.current.get(object.mask);
    if (!bits) return true;
    const scale = object.scale / 100 || 1;
    const cx = object.box.x + object.box.w / 2;
    const cy = object.box.y + object.box.h / 2;
    const mx = (nx - object.dx - cx) / scale + cx;
    const my = (ny - object.dy - cy) / scale + cy;
    const px = Math.floor(mx * bits.width);
    const py = Math.floor(my * bits.height);
    if (px < 0 || py < 0 || px >= bits.width || py >= bits.height) return false;
    return bits.alpha[py * bits.width + px] > 24;
  }

  function segmentAt(nx: number, ny: number, items: SegmentObject[]) {
    for (let index = items.length - 1; index >= 0; index -= 1) {
      if (onSegment(items[index], nx, ny)) return items[index];
    }
    return null;
  }

  function clearSelection() {
    setSelected(null);
    setFaceSelected(null);
    if (mode === "object") setMode("home");
  }
  const faceCurrent = faceObjects.find((object) => object.id === faceSelected) ?? null;
  const ideas = suggestEdits(visible);
  const removeTargets = ideas.remove;
  const addIdea = ideas.add;
  const allIdea = { warmth: ideas.all.warmth, label: tx(ideas.all.en, ideas.all.zh) };

  function startBox(event: PointerEvent<HTMLElement>, object: SegmentObject, face: boolean, kind: "move" | "resize") {
    event.stopPropagation();
    event.preventDefault();
    if (!face && isBackdrop(object.name)) return;
    if (face && /background|backdrop/i.test(object.name)) return;
    if (face) setFaceSelected(object.id);
    else {
      setSelected(object.id);
      setMode("object");
    }
    const base = face ? faceLive.current : objects;
    drag.current = { id: object.id, face, mode: kind, x: event.clientX, y: event.clientY, dx: object.dx, dy: object.dy, scale: object.scale, base, latest: base };
    try { stageRef.current?.setPointerCapture(event.pointerId); } catch { /* the drag still tracks while the pointer stays on the photo */ }
  }

  function deleteFromPhoto(targets: SegmentObject[], face: boolean) {
    const names = targets.map((object) => object.name).filter(Boolean);
    if (!names.length || busy) return;
    drag.current = null;
    const ids = new Set(targets.map((object) => object.id));
    if (face) {
      const next = faceLive.current.map((object) => (ids.has(object.id) ? { ...object, hidden: true } : object));
      if (faceSource === PORTRAIT) faceCache.current = structuredClone(next);
      if (faceSelected && ids.has(faceSelected)) setFaceSelected(next.find((object) => !object.hidden && !/background|backdrop/i.test(object.name))?.id ?? null);
      rememberFace(next, `Deleted ${names.join(", ")}`, "Delete");
      return;
    }
    const next = objects.map((object) => (ids.has(object.id) ? { ...object, hidden: true } : object));
    if (selected && ids.has(selected)) {
      const nextId = next.find((object) => !object.hidden)?.id ?? null;
      setSelected(nextId);
      if (!nextId) setMode("home");
    }
    keepRoom(next);
    snapshot(next, warmth, tx(`Deleted ${names.join(", ")}`, `已去掉${names.map((name) => objectName(name, true)).join("、")}`), tx("Delete", "删除"));
  }

  const dateLabel = new Date().toLocaleDateString(zh ? "zh-CN" : "en-US", { month: "long", day: "numeric" });
  const canUndo = (faceOn ? faceAt : at) > 0;
  const requestText = busy ? said || tx("Generating a new photo.", "正在生成新照片。") : voiceDraft;

  function leaveEdit() {
    setHistoryOpen(false);
    if (mode === "history") {
      setMode(hist === "face" ? "retouch" : "home");
      return;
    }
    if (mode === "object") {
      setSelected(null);
      setMode("home");
      return;
    }
    if (mode === "review") {
      setChanges(null);
      setView(null);
      if (at > 0) {
        const prior = steps[at - 1];
        setSteps(steps.slice(0, at));
        setAt(at - 1);
        setObjects(structuredClone(prior.objects));
        setWarmth(prior.warmth);
      }
      setMode("home");
      return;
    }
    if (mode === "retouch" || mode === "cleanup" || mode === "voice") {
      setMode("home");
      return;
    }
    setListening(false);
    setVoiceDraft("");
    setMode("photo");
  }

  function undoOnce() {
    if (faceOn) {
      if (faceAt > 0) useFace(faceAt - 1);
      return;
    }
    if (at > 0) useStep(at - 1);
  }

  function submitEdit() {
    const text = (prompt.trim() || voiceDraft).trim();
    if (!text) return;
    setPrompt("");
    setVoiceDraft("");
    setListening(false);
    void send(text);
  }

  function onVoice() {
    if (busy) return;
    if (listening) {
      setListening(false);
      return;
    }
    setListening(true);
    speak(zh ? "zh-CN" : "en-US", (text) => {
      setListening(false);
      setPrompt(text);
      setVoiceDraft("");
      setSaid(text);
    }, () => setListening(false));
  }

  return (
    <div className="ae">
      <header className="ae-top">
        {editing ? (
          <button type="button" onClick={leaveEdit} aria-label={mode === "object" ? tx("Back", "返回") : tx("Cancel", "取消")}>
            {mode === "object" ? tx("Back", "返回") : tx("Cancel", "取消")}
          </button>
        ) : (
          <span className="ae-top-gap" />
        )}
        <div className="ae-title">
          <span suppressHydrationWarning>{dateLabel}</span>
          {editing ? <small>AI {tx("Edit", "编辑")}</small> : null}
        </div>
        {editing ? (
          <div className="ae-top-actions">
            <button type="button" className="ae-undo" aria-label={tx("Undo", "撤销")} disabled={!canUndo || busy} onClick={undoOnce}>
              <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M9 14 4 9l5-5" />
                <path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" />
              </svg>
            </button>
            <button type="button" className="ae-finish" onClick={done}>{tx("Done", "完成")}</button>
          </div>
        ) : (
          <span className="ae-top-gap" />
        )}
      </header>
      <div className="ae-lang" role="group" aria-label={tx("Language", "语言")} translate="no" suppressHydrationWarning>
        <button type="button" className={zh ? "on" : ""} onClick={() => setLang("zh")} suppressHydrationWarning>中文</button>
        <button type="button" className={!zh ? "on" : ""} onClick={() => setLang("en")} suppressHydrationWarning>EN</button>
      </div>

      <div
        className="ae-photo"
        ref={frameRef}
        onPointerDownCapture={(event) => {
          pointers.current.add(event.pointerId);
          if (pointers.current.size < 2 || !drag.current) return;
          const session = drag.current;
          drag.current = null;
          if (session.face) {
            faceLive.current = session.base;
            setFaceObjects(session.base);
          } else setObjects(session.base);
        }}
        onPointerUpCapture={(event) => pointers.current.delete(event.pointerId)}
        onPointerCancelCapture={(event) => pointers.current.delete(event.pointerId)}
      >
        {editing && mode === "review" ? <div className="ae-tip">{tx("Colored areas are the new edits on the original. Approve to keep them as segments", "彩色区域是叠在原图上的新改动，采用后会变成可编辑的分区")}</div> : editing && mode !== "object" ? <div className="ae-tip">{tx("Tap an object in the photo to edit it", "你可以选择图片中的对象直接编辑")}</div> : null}
        <div
          ref={stageRef}
          className="ae-stage"
          style={{
            ["--ratio" as string]: ratio,
            filter: warmth > 0.2 ? "sepia(0.25) saturate(1.15)" : undefined,
            transform: `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.scale})`,
            transformOrigin: "0 0",
          }}
          onPointerMove={(event) => {
            const session = drag.current;
            if (!session || pointers.current.size >= 2) return;
            const rect = event.currentTarget.getBoundingClientRect();
            const dx = (event.clientX - session.x) / rect.width;
            const dy = (event.clientY - session.y) / rect.height;
            const next = session.base.map((object) => {
              if (object.id !== session.id) return object;
              if (session.mode === "resize") return { ...object, scale: Math.max(40, Math.min(220, Math.round(session.scale + Math.max(dx, dy) * 220))) };
              return { ...object, dx: session.dx + dx, dy: session.dy + dy };
            });
            session.latest = next;
            if (session.face) {
              faceLive.current = next;
              setFaceObjects(next);
            } else setObjects(next);
          }}
          onPointerUp={() => {
            const session = drag.current;
            drag.current = null;
            if (!session) return;
            const name = session.base.find((object) => object.id === session.id)?.name ?? "Region";
            const shown = objectName(name, zh);
            if (session.face) rememberFace(session.latest, session.mode === "resize" ? tx(`Resized ${name}`, `缩放了${shown}`) : tx(`Moved ${name}`, `移动了${shown}`), tx("Box", "选框"));
            else snapshot(session.latest, warmth, session.mode === "resize" ? tx(`Resized ${name}`, `缩放了${shown}`) : tx(`Moved ${name}`, `移动了${shown}`), shown);
          }}
          onPointerDown={(event) => {
            if (mode === "review" && changes) {
              if ((event.target as HTMLElement).closest("button")) return;
              const rect = event.currentTarget.getBoundingClientRect();
              const hit = islandAt((event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height, changes.islands);
              if (hit) void toggleIsland(hit.id);
              return;
            }
            if (event.target !== event.currentTarget && !(event.target instanceof HTMLImageElement)) return;
            const rect = event.currentTarget.getBoundingClientRect();
            const hit = segmentAt((event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height, faceOn ? faceMarks : visible);
            if (hit) startBox(event, hit, faceOn, "move");
            else clearSelection();
          }}
        >
          {editing ? (
            <div className="ae-photo-actions">
              <span className="ae-badge">{hold ? tx("Original", "原图") : tx("Preview", "效果预览")}</span>
            </div>
          ) : null}
          <img
            src={picture}
            alt=""
            onLoad={(event) => {
              const image = event.currentTarget;
              if (image.naturalWidth) setRatio(image.naturalWidth / image.naturalHeight);
            }}
          />
          {faceOn || (editing && mode !== "voice" && mode !== "review")
            ? (faceOn ? faceMarks : visible).map((object) => {
                const frame = boxFrame(object);
                const on = faceOn ? object.id === faceSelected : object.id === selected && mode === "object";
                const shown = objectName(object.name, zh);
                return (
                  <div
                    key={object.id}
                    className={on ? "ae-box on" : "ae-box"}
                    style={{ left: `${frame.left * 100}%`, top: `${frame.top * 100}%`, width: `${frame.width * 100}%`, height: `${frame.height * 100}%` }}
                    onPointerDown={(event) => {
                      const target = event.target as HTMLElement;
                      if (target.closest(".ae-resize, .ae-x")) return;
                      const stage = stageRef.current?.getBoundingClientRect();
                      if (!stage) return;
                      const hit = segmentAt((event.clientX - stage.left) / stage.width, (event.clientY - stage.top) / stage.height, faceOn ? faceMarks : visible);
                      if (hit) startBox(event, hit, faceOn, "move");
                      else clearSelection();
                    }}
                  >
                    {on ? <b>{shown} · {tx("Drag to move", "拖动移动")}</b> : null}
                    {on ? (
                      <button
                        type="button"
                        className="ae-x"
                        aria-label={tx(`Delete ${object.name}`, `删除${shown}`)}
                        onPointerDown={(event) => {
                          event.stopPropagation();
                          event.preventDefault();
                        }}
                        onPointerUp={(event) => {
                          event.stopPropagation();
                          event.preventDefault();
                          void deleteFromPhoto([object], faceOn);
                        }}
                      >
                        ×
                      </button>
                    ) : null}
                    {on ? (
                      <button
                        type="button"
                        className="ae-resize"
                        aria-label={tx(`Resize ${object.name}`, `调整${shown}大小`)}
                        onPointerDown={(event) => startBox(event, object, faceOn, "resize")}
                      />
                    ) : null}
                  </div>
                );
              })
            : null}
        </div>
      </div>
      <div className="ae-bottom">
      <div className="ae-under">
        {editing ? (
          <button type="button" className={historyOpen ? "ae-hist-arrow on" : "ae-hist-arrow"} aria-label={tx("History", "历史")} aria-expanded={historyOpen} onClick={() => setHistoryOpen((open) => !open)}>
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d={historyOpen ? "m6 10 6 6 6-6" : "m6 14 6-6 6 6"} />
            </svg>
          </button>
        ) : <span />}
        {editing ? (
          <button type="button" className="ae-compare" onPointerDown={(event) => { event.preventDefault(); setHold(true); }} onPointerUp={() => setHold(false)} onPointerLeave={() => setHold(false)} onPointerCancel={() => setHold(false)}>
            {tx("Hold for original", "按住看原图")}
          </button>
        ) : null}
      </div>
      {historyOpen && editing ? (
        <div className="ae-hist-pop" role="dialog" aria-label={faceOn ? tx("Retouch history", "精修历史") : tx("Edit history", "编辑历史")}>
          {faceOn ? [...faceSteps].reverse().map((step, index) => {
            const real = faceSteps.length - 1 - index;
            return (
              <div key={step.id} className={real === faceAt ? "ae-step on" : "ae-step"} style={{ opacity: step.shown ? 1 : 0.45 }}>
                <button type="button" className="ae-step-main" onClick={() => { if (!step.shown) return; useFace(real); }}>
                  <b>{real + 1}</b>
                  <span><b>{step.label}</b><br /><span className="ae-muted">{step.meta}</span></span>
                </button>
                <button type="button" className="ae-icon" aria-label={step.shown ? tx("Hide step", "隐藏此步") : tx("Show step", "显示此步")} onClick={() => toggleFace(real)}><Eye open={step.shown} /></button>
                <button type="button" className="ae-icon" aria-label={tx(`Delete ${step.label}`, `删除${step.label}`)} onClick={() => dropFace(real)}>×</button>
              </div>
            );
          }) : [...steps].reverse().map((step, index) => {
            const real = steps.length - 1 - index;
            return (
              <div key={step.id} className={real === at ? "ae-step on" : "ae-step"} style={{ opacity: step.shown ? 1 : 0.45 }}>
                <button type="button" className="ae-step-main" onClick={() => { if (!step.shown) return; setObjects(structuredClone(step.objects)); setWarmth(step.warmth); setAt(real); }}>
                  <b>{real + 1}</b>
                  <span><b>{step.label}</b><br /><span className="ae-muted">{step.meta}</span></span>
                </button>
                <button type="button" className="ae-icon" aria-label={step.shown ? tx("Hide step", "隐藏此步") : tx("Show step", "显示此步")} onClick={() => toggleStep(real)}><Eye open={step.shown} /></button>
                <button type="button" className="ae-icon" aria-label={tx(`Delete ${step.label}`, `删除${step.label}`)} onClick={() => dropStep(real)}>×</button>
              </div>
            );
          })}
        </div>
      ) : null}

      {mode === "photo" ? (
        <section className="ae-browse">
          <div className="ae-caption">
            <span>{Math.max(1, shots.findIndex((shot) => shot.id === shotId) + 1)} / {shots.length}</span>
            <span>{(() => { const shot = shots.find((item) => item.id === shotId); return shot ? tx(shot.label, shot.zh) : ""; })()}</span>
          </div>
          <div className="ae-thumbs">
            {shots.map((shot) => (
              <button key={shot.id} type="button" className={shot.id === shotId ? "ae-thumb active" : "ae-thumb"} aria-label={tx(shot.label, shot.zh)} onClick={() => selectShot(shot)}>
                <img src={shot.src} alt="" />
              </button>
            ))}
            <button type="button" className="ae-add" aria-label={tx("Add a photo", "添加照片")} onClick={() => photoRef.current?.click()}>+</button>
          </div>
          <div className="ae-tools">
            <button type="button" className="ae-ai" disabled={busy} onClick={() => void openActive()}>✧ &nbsp; {tx("AI Edit", "AI 编辑")}</button>
          </div>
          <input
            ref={photoRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void addShot(file);
              event.target.value = "";
            }}
          />
        </section>
      ) : mode === "cleanup" ? (
        <Cleanup cleanOn={cleanOn} weathered={weathered} busy={busy} onToggle={(id) => setCleanOn((items) => items.includes(id) ? items.filter((item) => item !== id) : [...items, id])} onWeather={setWeathered} onErase={() => void send(cleanupPrompt(cleanOn, weathered))} onHold={setHold} />
      ) : mode === "retouch" ? (
        <Retouch
          preset={preset}
          eyes={eyes}
          lines={lines}
          blur={blur}
          light={light}
          busy={busy}
          error={error}
          regions={faceMarks}
          selected={faceCurrent}
          onPreset={(name) => {
            setPreset(name);
            const look = name === "Studio" ? [70, 35, 18] : name === "LinkedIn" ? [45, 28, 10] : [15, 8, 0];
            const painted = paintFace(faceLive.current, look[0], look[1], look[2]).map((object) => {
              if (name !== "Studio") return object;
              if (/lip/i.test(object.name)) return { ...object, tint: "#d46a78", cover: 0.5, shade: 0 };
              if (/cheek/i.test(object.name)) return { ...object, tint: "#e09a84", cover: 0.35, shade: 0 };
              return object;
            });
            rememberFace(painted, name, tx("Preset", "预设"), { eyes: look[0], lines: look[1], blur: look[2] });
          }}
          onEyes={(value) => tuneFace(value, lines, blur)}
          onLines={(value) => tuneFace(eyes, value, blur)}
          onBlur={(value) => tuneFace(eyes, lines, value)}
          onCommit={(label) => rememberFace(faceLive.current, label, tx("Retouch", "精修"))}
          onLight={setLight}
          onSelect={setFaceSelected}
          onTint={tintFace}
          onCover={coverFace}
        />
      ) : (
        <section className="ae-editor">
          {mode === "object" && current ? (
            <div className="ae-examples">
              <div className="ae-object-options">
                <p className="ae-option-title"><span>{tx("Change color", "修改颜色")}</span></p>
                <div className="ae-option-row ae-colors">
                  <button type="button" aria-label={tx("Original", "原色")} className={!current.tint ? "ae-color original on" : "ae-color original"} onClick={() => applyObjectColor(current.id, null, 0, tx(`${current.name} · original color`, `${objectName(current.name, true)} · 原色`))} />
                  {(current.variants?.length
                    ? [...new Map(current.variants.map((item) => [item.hex.toLowerCase(), item.hex])).values()].map((hex) => {
                        const named = PALETTES[palette].find((swatch) => swatch.hex.toLowerCase() === hex.toLowerCase());
                        return { name: named?.name || hex, hex };
                      })
                    : PALETTES[palette].slice(0, 3)
                  ).map((swatch) => (
                    <button key={swatch.hex} type="button" aria-label={objectName(swatch.name, zh)} className={current.tint === swatch.hex ? "ae-color on" : "ae-color"} style={{ background: swatch.hex }} onClick={() => applyObjectColor(current.id, swatch.hex, 0, tx(`${current.name} · ${swatch.name}`, `${objectName(current.name, true)} · ${swatch.name}`))} />
                  ))}
                  <label className="ae-color rainbow" aria-label={tx("More colors", "更多颜色")}>
                    <input type="color" value={current.tint || "#c46a4a"} onChange={(event) => applyObjectColor(current.id, event.target.value, 0, tx(`${current.name} recolored`, `${objectName(current.name, true)}改色`))} />
                  </label>
                </div>
                <div className="ae-option-row ae-shades">
                  {SHADES.map((shade) => {
                    const base = current.tint || PALETTES[palette][0].hex;
                    return (
                      <button
                        key={shade.label}
                        type="button"
                        aria-label={zh ? shade.zh : shade.label}
                        className={current.tint && current.shade === shade.value ? "ae-color on" : "ae-color"}
                        style={{ background: shadeColor(base, shade.value) }}
                        onClick={() => applyObjectColor(current.id, base, shade.value, tx(`${current.name} · ${shade.label}`, `${objectName(current.name, true)} · ${shade.zh}`))}
                      />
                    );
                  })}
                </div>
                {/* Suggested / Warm / Cool tabs hidden — show suggested swatches + picker only.
                <div className="ae-picks">
                  {(["ai", "warm", "cool"] as Palette[]).map((item) => (
                    <button key={item} type="button" className={palette === item ? "ae-variant on" : "ae-variant"} onClick={() => setPalette(item)}>
                      {item === "ai" ? tx("Suggested", "推荐色") : item === "warm" ? tx("Warm", "暖色") : tx("Cool", "冷色")}
                    </button>
                  ))}
                </div>
                */}
              </div>
            </div>
          ) : mode === "review" ? (
            <div className="ae-examples">
              <p className="ae-recommend">{tx("Changes", "改动")}</p>
              {said ? <p className="ae-applied">{said}</p> : null}
              {(changes?.islands ?? []).map((island, index) => (
                <div key={island.id} className={island.on ? "ae-change" : "ae-change off"}>
                  <i className="ae-swatch" style={{ background: island.color }} />
                  <span>{objectName(island.label || `Change ${index + 1}`, zh)}</span>
                  <button type="button" onClick={() => void toggleIsland(island.id)}>
                    {island.on ? tx("Remove", "去掉") : tx("Keep", "保留")}
                  </button>
                </div>
              ))}
            </div>
          ) : listening || busy ? (
            <div className="ae-examples" />
          ) : (
            <div className="ae-examples">
              <p className="ae-recommend">{tx("Suggestions", "推荐")}</p>
              {error ? <p className="ae-error">{error}</p> : null}
              {removeTargets.length ? (
                <button
                  type="button"
                  className="ae-chip"
                  disabled={busy}
                  onClick={() => {
                    void deleteFromPhoto(removeTargets, false);
                  }}
                >
                  <span>{tx("Del", "删")}</span>
                  {tx(`Remove ${removeTargets.map((object) => object.name).join(", ")}`, `去掉${removeTargets.map((object) => objectName(object.name, true)).join("、")}`)}
                </button>
              ) : null}
              {addIdea ? (
                <button
                  type="button"
                  className="ae-chip"
                  disabled={busy}
                  onClick={() => {
                    void addRegion(addIdea);
                  }}
                >
                  <span>{tx("Add", "增")}</span>
                  {tx(`Add the ${addIdea}`, `加上${objectName(addIdea, true)}`)}
                </button>
              ) : null}
              <button
                type="button"
                className="ae-chip"
                disabled={busy}
                onClick={() => {
                  snapshot(objects, warmth > 0.2 ? 0 : allIdea.warmth, allIdea.label, tx("Whole photo", "整体"));
                }}
              >
                <span>{tx("All", "整体")}</span>
                {allIdea.label}
              </button>
            </div>
          )}
          {mode === "review" ? (
            <button type="button" className="ae-approve" onClick={done}>{tx("Approve", "采用")}</button>
          ) : null}
          <div className="ae-composer">
            {requestText ? (
              <div className="ae-request" role="status">
                <span>{requestText}</span>
                {busy ? <span className="ae-spinner" aria-label={tx("Generating", "正在生成")} /> : null}
              </div>
            ) : null}
            <div className="ae-voicebox">
              <button type="button" className={listening ? "ae-voice recording" : "ae-voice"} aria-label={tx("Voice input", "语音输入")} onClick={onVoice}>
                <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"><rect x="9" y="3" width="6" height="12" rx="3" /><path d="M5 11v1a7 7 0 0 0 14 0v-1M12 19v3M9 22h6" /></svg>
              </button>
              <textarea
                ref={promptRef}
                className="ae-prompt"
                value={prompt}
                disabled={busy}
                rows={Math.min(6, Math.max(1, prompt.split("\n").length))}
                placeholder={listening ? tx("Listening…", "正在听…") : tx("Describe the edit", "描述要修改的内容")}
                aria-label={tx("Edit prompt", "编辑提示")}
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    submitEdit();
                  }
                }}
              />
              <button type="button" className="ae-voice-send" aria-label={tx("Send", "发送")} disabled={busy} onClick={submitEdit}>↑</button>
            </div>
          </div>
        </section>
      )}
      </div>

      {mode === "history" ? (
        <section className="ae-history">
          <div className="ae-history-head">
            <strong>{hist === "face" ? tx("Retouch history", "精修历史") : tx("Edit history", "编辑历史")}</strong>
            <button type="button" onClick={() => setMode(hist === "face" ? "retouch" : "home")}>{tx("Close", "关闭")}</button>
          </div>
          <p className="ae-muted">{tx("Every step is kept. Tap one to go back. Hide or delete a step on the right.", "每一步都会保留。点一条即可回到那一步，右侧可以隐藏或删除。")}</p>
          {hist === "face" ? [...faceSteps].reverse().map((step, index) => {
            const real = faceSteps.length - 1 - index;
            return (
              <div key={step.id} className={real === faceAt ? "ae-step on" : "ae-step"} style={{ opacity: step.shown ? 1 : 0.45 }}>
                <button type="button" className="ae-step-main" onClick={() => { if (!step.shown) return; useFace(real); }}>
                  <b>{real + 1}</b>
                  <span><b>{step.label}</b><br /><span className="ae-muted">{step.meta}</span></span>
                </button>
                <button type="button" className="ae-icon" aria-label={step.shown ? tx("Hide step", "隐藏此步") : tx("Show step", "显示此步")} onClick={() => toggleFace(real)}>
                  <Eye open={step.shown} />
                </button>
                <button type="button" className="ae-icon" aria-label={tx(`Delete ${step.label}`, `删除${step.label}`)} onClick={() => dropFace(real)}>×</button>
              </div>
            );
          }) : [...steps].reverse().map((step, index) => {
            const real = steps.length - 1 - index;
            return (
              <div key={step.id} className={real === at ? "ae-step on" : "ae-step"} style={{ opacity: step.shown ? 1 : 0.45 }}>
                <button type="button" className="ae-step-main" onClick={() => { if (!step.shown) return; setObjects(structuredClone(step.objects)); setWarmth(step.warmth); setAt(real); }}>
                  <b>{real + 1}</b>
                  <span><b>{step.label}</b><br /><span className="ae-muted">{step.meta}</span></span>
                </button>
                <button type="button" className="ae-icon" aria-label={step.shown ? tx("Hide step", "隐藏此步") : tx("Show step", "显示此步")} onClick={() => toggleStep(real)}>
                  <Eye open={step.shown} />
                </button>
                <button type="button" className="ae-icon" aria-label={tx(`Delete ${step.label}`, `删除${step.label}`)} onClick={() => dropStep(real)}>×</button>
              </div>
            );
          })}
        </section>
      ) : null}
    </div>
  );
}
function Cleanup(props: { cleanOn: string[]; weathered: boolean; busy: boolean; onToggle: (id: string) => void; onWeather: (keep: boolean) => void; onErase: () => void; onHold: (hold: boolean) => void }) {
  const { lang } = useLang();
  const zh = lang === "zh";
  const items = [
    { id: "passer", title: zh ? "左侧路人" : "Passer-by on the left", note: zh ? "补上后面的墙和路沿" : "Rebuilds the wall and curb behind" },
    { id: "rust", title: zh ? "锈迹" : "Rust streaks", note: zh ? "属于画面气质，先保留" : "Part of the look, so left on" },
    { id: "patch", title: zh ? "灰色补丁" : "Grey patch", note: zh ? "属于画面气质，先保留" : "Part of the look, so left on" },
  ];
  return (
    <>
      <section className="ae-sheet">
        <h2>{zh ? "清理这张照片" : "Clean up this shot"}</h2>
        <p className="ae-muted">{zh ? "找到 1 个路人和 2 处墙面痕迹。你和指示牌会保留。" : "1 passer-by and 2 wall marks found. You and the signs stay."}</p>
        {items.map((item) => (
          <button key={item.id} type="button" className={props.cleanOn.includes(item.id) ? "ae-version on" : "ae-version"} onClick={() => props.onToggle(item.id)}>
            <span><b>{item.title}</b><br /><span className="ae-muted">{props.cleanOn.includes(item.id) ? item.note : (zh ? "保留" : "Left on")}</span></span>
          </button>
        ))}
        <div className="ae-tabs3">
          <button type="button" className={props.weathered ? "on" : ""} onClick={() => props.onWeather(true)}>{zh ? "保留旧痕" : "Keep it weathered"}</button>
          <button type="button" className={!props.weathered ? "on" : ""} onClick={() => props.onWeather(false)}>{zh ? "更干净" : "Cleaner"}</button>
        </div>
        <div className="ae-row">
          <button type="button" className="ae-link" onPointerDown={() => props.onHold(true)} onPointerUp={() => props.onHold(false)}>{zh ? "按住对比" : "Hold to compare"}</button>
          <button type="button" className="ae-done" disabled={props.busy} onClick={props.onErase}>{zh ? `擦除 ${props.cleanOn.length}` : `Erase ${props.cleanOn.length}`}</button>
        </div>
      </section>
    </>
  );
}

function colorName(hex: string) {
  const found = [...Object.values(PALETTES).flat(), ...MAKEUP].find((swatch) => swatch.hex.toLowerCase() === hex.toLowerCase());
  if (found) return found.name.toLowerCase();
  if (hex.toLowerCase() === "#fff1e6") return "a lighter skin tone";
  if (hex.toLowerCase() === "#fff6ee") return "a softer skin tone";
  return "the color already painted on it";
}

function suggestEdits(items: SegmentObject[]) {
  const visible = items.filter((object) => !object.hidden && !/background|backdrop|wall|floor|ceiling|sky/i.test(object.name));
  const allNames = items.map((object) => object.name.toLowerCase()).join(" | ");
  const joined = visible.map((object) => object.name.toLowerCase()).join(" | ");
  const has = (pattern: RegExp) => pattern.test(joined);
  const hasSaved = (pattern: RegExp) => pattern.test(allNames);
  const missing = (pattern: RegExp) => !has(pattern);
  // Prefer suggesting a saved-but-hidden extra (curtain) before inventing a new add.
  const savedAdd = items.find((object) => object.hidden && object.suggested);

  const rank = (name: string) => {
    if (/phone|bag|cup|bottle|remote|cable|trash/i.test(name)) return 4;
    if (/plant|flower|vase/i.test(name)) return 3;
    if (/lamp|light|pillow|cushion|rug|carpet/i.test(name)) return 2;
    if (/chair|table|ottoman|stool/i.test(name)) return 1;
    return 0;
  };
  const remove = [...visible]
    .filter((object) => rank(object.name) > 0)
    .sort((a, b) => rank(b.name) - rank(a.name) || b.box.w * b.box.h - a.box.w * a.box.h)
    .slice(0, 2);

  const addChoices: { when: boolean; name: string }[] = [
    { when: Boolean(savedAdd), name: savedAdd?.name ?? "" },
    { when: has(/sofa|couch|chair/) && missing(/plant|flower/) && !hasSaved(/plant|flower/), name: "plant" },
    // Curtains by the window — sofa already has a blanket, so don't suggest cushion.
    { when: has(/sofa|couch|window|lamp/) && missing(/curtain|drape/) && !hasSaved(/curtain|drape/), name: "curtain" },
    { when: has(/sofa|couch|chair|table/) && missing(/rug|carpet/), name: "rug" },
    { when: missing(/lamp|light/), name: "floor lamp" },
    { when: has(/wall|sofa|chair/) && missing(/paint|art|picture|frame/), name: "painting" },
    { when: missing(/shelf|book/), name: "bookshelf" },
    { when: has(/table/) && missing(/vase|flower/), name: "vase" },
    { when: missing(/table/), name: "side table" },
  ];
  const add = addChoices.find((choice) => choice.when && choice.name)?.name ?? "";

  const all = has(/plant|window|daylight/)
    ? { en: "Make the room feel brighter and airier", zh: "让房间更明亮通透", warmth: -0.12 }
    : has(/sofa|chair|lamp/)
      ? { en: "Make the room feel like a warm afternoon", zh: "营造温暖的午后氛围", warmth: 0.28 }
      : { en: "Make the whole photo a little warmer", zh: "让整张照片更温暖一点", warmth: 0.22 };

  return { remove, add, all };
}

function editLines(items: SegmentObject[], warmth: number, light?: string) {
  const lines: string[] = [];
  for (const object of items) {
    if (object.hidden) {
      lines.push(`Remove the ${object.name}.`);
      continue;
    }
    const parts: string[] = [];
    if (object.tint) {
      const shade = SHADES.find((item) => item.value === object.shade);
      const tone = shade && shade.value !== 0 ? `, ${shade.label.toLowerCase()}` : "";
      const hex = shadeColor(object.tint, object.shade || 0).toUpperCase();
      const strength = object.cover != null ? ` at about ${Math.round(object.cover * 100)}% strength` : "";
      parts.push(`recolor only the ${object.name} to ${colorName(object.tint)}${tone} (${hex})${strength}`);
    }
    if (Math.abs(object.scale - 100) >= 5) parts.push(`${object.scale > 100 ? "enlarge" : "shrink"} the ${object.name}`);
    if (Math.abs(object.dx) >= 0.02 || Math.abs(object.dy) >= 0.02) {
      const way = [object.dy <= -0.02 ? "up" : object.dy >= 0.02 ? "down" : "", object.dx <= -0.02 ? "left" : object.dx >= 0.02 ? "right" : ""].filter(Boolean).join(" and ");
      parts.push(`move the ${object.name} ${way}`);
    }
    if (object.blur && object.blur > 1) parts.push(`blur the ${object.name}`);
    if (parts.length) lines.push(`${parts[0][0].toUpperCase()}${parts[0].slice(1)}${parts.length > 1 ? `, and ${parts.slice(1).join(", and ")}` : ""}.`);
  }
  if (warmth > 0.05) lines.push("Make the whole photo warmer.");
  if (warmth < -0.05) lines.push("Make the whole photo cooler.");
  if (light && light !== "Even") lines.push(`Light the subject from the ${light.toLowerCase()}.`);
  return lines;
}

type GestureLike = Event & { scale: number; clientX: number; clientY: number };

function overlapScore(a: { left: number; top: number; width: number; height: number }, b: { x: number; y: number; w: number; h: number }) {
  const width = Math.max(0, Math.min(a.left + a.width, b.x + b.w) - Math.max(a.left, b.x));
  const height = Math.max(0, Math.min(a.top + a.height, b.y + b.h) - Math.max(a.top, b.y));
  const area = Math.min(a.width * a.height, b.w * b.h);
  return area > 0 ? (width * height) / area : 0;
}

function pickAdded(kept: SegmentObject[], found: SegmentObject[], name: string) {
  const want = name.trim().toLowerCase();
  // Prefer an exact name match from color-map — do not reject for sofa overlap.
  const named = found
    .filter((object) => object.name.toLowerCase() === want)
    .sort((a, b) => b.box.w * b.box.h - a.box.w * a.box.h);
  if (named.length) {
    return [{ ...named[0], name }];
  }
  const unused = found.filter(
    (object) =>
      !kept.some(
        (prior) =>
          prior.name.toLowerCase() === object.name.toLowerCase() ||
          overlapScore(boxFrame(prior), object.box) > 0.45
      )
  );
  const chosen = unused.length ? [unused.sort((a, b) => b.box.w * b.box.h - a.box.w * a.box.h)[0]] : [];
  return chosen.map((object) => ({
    ...object,
    name: object.name === "Object" || object.name === "Region" ? name : object.name,
  }));
}

function boxFrame(object: SegmentObject) {
  const scale = object.scale / 100;
  const width = object.box.w * scale;
  const height = object.box.h * scale;
  return {
    left: object.box.x + object.dx + (object.box.w - width) / 2,
    top: object.box.y + object.dy + (object.box.h - height) / 2,
    width,
    height,
  };
}

function paintFace(items: SegmentObject[], nextEyes: number, nextLines: number, nextBlur: number) {
  return items.map((object) => {
    if (/under/i.test(object.name)) return { ...object, tint: nextEyes ? "#fff1e6" : null, cover: nextEyes ? nextEyes / 180 : undefined, shade: 0 };
    if (/forehead/i.test(object.name)) return { ...object, tint: nextLines ? "#fff6ee" : null, cover: nextLines ? nextLines / 200 : undefined, shade: 0 };
    if (/background|backdrop/i.test(object.name)) return { ...object, blur: nextBlur / 4 };
    return object;
  });
}

const MAKEUP = [
  { name: "Nude", hex: "#c48a78" },
  { name: "Peach", hex: "#e09a84" },
  { name: "Rose", hex: "#d46a78" },
  { name: "Berry", hex: "#a33d55" },
  { name: "Coral", hex: "#e07a62" },
  { name: "Soft", hex: "#f3e4d8" },
];

function Retouch(props: {
  preset: string;
  eyes: number;
  lines: number;
  blur: number;
  light: string;
  busy: boolean;
  error: string;
  regions: SegmentObject[];
  selected: SegmentObject | null;
  onPreset: (value: string) => void;
  onEyes: (value: number) => void;
  onLines: (value: number) => void;
  onBlur: (value: number) => void;
  onCommit: (label: string) => void;
  onLight: (value: string) => void;
  onSelect: (id: string) => void;
  onTint: (hex: string | null) => void;
  onCover: (value: number) => void;
}) {
  const { lang } = useLang();
  const zh = lang === "zh";
  const presets = zh
    ? [["Natural", "自然"], ["LinkedIn", "职业"], ["Studio", "影棚"]]
    : [["Natural", "Natural"], ["LinkedIn", "LinkedIn"], ["Studio", "Studio"]];
  const lights = zh
    ? [["Left", "左侧"], ["Even", "均匀"], ["Right", "右侧"]]
    : [["Left", "Left"], ["Even", "Even"], ["Right", "Right"]];
  return (
    <>
      <section className="ae-sheet">
        <div className="ae-row">
          <h2>{props.busy ? (zh ? "正在找面部区域" : "Finding face regions") : (zh ? `${props.regions.length} 个区域 · ${presets.find((item) => item[0] === props.preset)?.[1] ?? props.preset}` : `${props.regions.length} regions · ${props.preset}`)}</h2>
          <span className="ae-muted">{zh ? "点一个区域改颜色" : "Tap one to recolor"}</span>
        </div>
        {props.error ? <p className="ae-error">{props.error}</p> : null}
        <div className="ae-tabs3">
          {presets.map(([name, label]) => <button key={name} type="button" className={props.preset === name ? "on" : ""} onClick={() => props.onPreset(name)}>{label}</button>)}
        </div>
        <div className="ae-regions">
          {props.regions.map((region) => (
            <button key={region.id} type="button" className={region.id === props.selected?.id ? "on" : ""} onClick={() => props.onSelect(region.id)}>
              {objectName(region.name, zh)}
            </button>
          ))}
        </div>
        {props.selected ? (
          <>
            <div className="ae-row"><span>{objectName(props.selected.name, zh)}</span><span className="ae-muted">{zh ? "颜色只作用在这个区域" : "Color stays on this region"}</span></div>
            <div className="ae-swatches">
              <button type="button" aria-label={zh ? "原色" : "Original"} className={!props.selected.tint ? "ae-original on" : "ae-original"} onClick={() => props.onTint(null)} />
              {MAKEUP.map((swatch) => (
                <button key={swatch.hex} type="button" aria-label={swatch.name} className={props.selected?.tint === swatch.hex ? "on" : ""} style={{ background: swatch.hex }} onClick={() => props.onTint(swatch.hex)} />
              ))}
            </div>
            <div className="ae-row"><span>{zh ? "浓淡" : "Opacity"}</span><span className="ae-muted">{Math.round((props.selected.cover ?? 0.5) * 100)}%</span></div>
            <input type="range" min={0} max={100} value={Math.round((props.selected.cover ?? 0.5) * 100)} disabled={!props.selected.tint} onChange={(event) => props.onCover(Number(event.target.value) / 100)} onPointerUp={() => props.onCommit(`${props.selected?.name ?? "Region"} · opacity`)} />
          </>
        ) : null}
        <div className="ae-row"><span>{zh ? "眼下阴影" : "Under-eye shadows"}</span><span className="ae-muted">{zh ? "变浅" : "Lighter"}</span></div>
        <input type="range" min={0} max={100} value={props.eyes} onChange={(event) => props.onEyes(Number(event.target.value))} onPointerUp={() => props.onCommit("Under-eye shadows")} />
        <div className="ae-row"><span>{zh ? "额头纹" : "Forehead lines"}</span><span className="ae-muted">{zh ? "稍微柔化" : "Soften a little"}</span></div>
        <input type="range" min={0} max={100} value={props.lines} onChange={(event) => props.onLines(Number(event.target.value))} onPointerUp={() => props.onCommit("Forehead lines")} />
        <div className="ae-row"><span>{zh ? "杂乱背景" : "Busy background"}</span><span className="ae-muted">{zh ? "轻微虚化" : "Soft blur"}</span></div>
        <input type="range" min={0} max={100} value={props.blur} onChange={(event) => props.onBlur(Number(event.target.value))} onPointerUp={() => props.onCommit("Background blur")} />
        <div className="ae-tabs3">
          {lights.map(([name, label]) => <button key={name} type="button" className={props.light === name ? "on" : ""} onClick={() => props.onLight(name)}>{label}</button>)}
        </div>
      </section>
    </>
  );
}

function cleanupPrompt(ids: string[], weathered: boolean) {
  const names = [
    ids.includes("passer") ? "the passer-by on the left" : "",
    ids.includes("rust") ? "the rust streaks" : "",
    ids.includes("patch") ? "the grey patch" : "",
  ].filter(Boolean);
  return `Remove ${names.join(" and ")}. ${weathered ? "Keep the wall weathered." : "Make the wall cleaner."} Keep the person and the signs.`;
}

function speak(language: string, onText: (text: string) => void, onFail: () => void) {
  const host = window as Window & { SpeechRecognition?: SpeechCtor; webkitSpeechRecognition?: SpeechCtor };
  const Recognition = host.SpeechRecognition || host.webkitSpeechRecognition;
  if (!Recognition) {
    onFail();
    return;
  }
  const recognition = new Recognition();
  recognition.lang = language;
  recognition.onresult = (event) => onText(event.results[0][0].transcript);
  recognition.onerror = onFail;
  recognition.start();
}

type SpeechCtor = new () => {
  lang: string;
  start: () => void;
  onresult: ((event: { results: { [index: number]: { [index: number]: { transcript: string } } } }) => void) | null;
  onerror: (() => void) | null;
};

function Eye({ open }: { open: boolean }) {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
      <path d="M2 12s3.8-6.2 10-6.2S22 12 22 12s-3.8 6.2-10 6.2S2 12 2 12z" fill="none" stroke="currentColor" strokeWidth="1.7" />
      <circle cx="12" cy="12" r="2.3" fill="none" stroke="currentColor" strokeWidth="1.7" />
      {open ? null : <path d="M5 19 L19 5" fill="none" stroke="currentColor" strokeWidth="1.7" />}
    </svg>
  );
}

