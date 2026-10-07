"use client";

import { useEffect, useRef, useState, type MouseEvent, type PointerEvent } from "react";
import type { Island } from "@/data/edits";
import type { SelectionMode } from "./Menu";

export type PaintLayer = {
  id: string;
  visible: boolean;
  edited: string;
  islands: Island[];
};

type Props = {
  original: string;
  layers: PaintLayer[];
  enabled: Record<string, boolean>;
  selectedIds: string[];
  minArea: number;
  selection: SelectionMode;
  eraserSize: number;
  activeId: string;
  onSelect: (ids: string[]) => void;
  onErase: (points: { x: number; y: number }[]) => void;
  onLasso: (points: { x: number; y: number }[]) => void;
  onCut: (points: { x: number; y: number }[]) => void;
  onSize: (width: number, height: number) => void;
};

type Box = { x: number; y: number; x2: number; y2: number };

type LayerPack = {
  id: string;
  cutouts: HTMLCanvasElement[];
  tints: HTMLCanvasElement[];
  alpha: Uint8ClampedArray[];
};

type GestureLike = Event & { scale: number; clientX: number; clientY: number };

type Pack = {
  width: number;
  height: number;
  base: HTMLImageElement;
  layers: LayerPack[];
};

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${src}`));
    image.src = src;
  });
}

function cutout(edited: HTMLImageElement, mask: HTMLImageElement, width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;
  ctx.drawImage(edited, 0, 0, width, height);
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(mask, 0, 0, width, height);
  return canvas;
}

function tint(mask: HTMLImageElement, color: string, width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;
  ctx.drawImage(mask, 0, 0, width, height);
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, width, height);
  return canvas;
}

function alphaOf(mask: HTMLImageElement, width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return new Uint8ClampedArray();
  ctx.drawImage(mask, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height).data;
}

function imagePoint(
  event: { clientX: number; clientY: number },
  canvas: HTMLCanvasElement,
  pack: Pack
) {
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / pack.width, rect.height / pack.height);
  const drawnWidth = pack.width * scale;
  const drawnHeight = pack.height * scale;
  return {
    x: Math.floor((event.clientX - rect.left - (rect.width - drawnWidth) / 2) / scale),
    y: Math.floor((event.clientY - rect.top - (rect.height - drawnHeight) / 2) / scale),
  };
}

function packedLayer(pack: Pack, layer: PaintLayer) {
  return pack.layers.find((item) => item.id === layer.id);
}

function islandsInBox(pack: Pack, layers: PaintLayer[], minArea: number, activeId: string, box: Box) {
  const left = Math.max(0, Math.min(box.x, box.x2));
  const right = Math.min(pack.width - 1, Math.max(box.x, box.x2));
  const top = Math.max(0, Math.min(box.y, box.y2));
  const bottom = Math.min(pack.height - 1, Math.max(box.y, box.y2));
  const hits: string[] = [];

  layers.forEach((layer) => {
    if (layer.id !== activeId || !layer.visible) return;
    const packed = packedLayer(pack, layer);
    if (!packed) return;
    layer.islands.forEach((island, index) => {
      if (island.area < minArea) return;
      for (let y = top; y <= bottom; y += 4) {
        const row = y * pack.width;
        for (let x = left; x <= right; x += 4) {
          if ((packed.alpha[index]?.[(row + x) * 4 + 3] ?? 0) > 40) {
            hits.push(island.id);
            return;
          }
        }
      }
    });
  });

  return hits;
}

function hitIsland(pack: Pack, layers: PaintLayer[], minArea: number, activeId: string, x: number, y: number) {
  const pixel = (y * pack.width + x) * 4 + 3;
  for (let index = layers.length - 1; index >= 0; index -= 1) {
    const layer = layers[index];
    if (layer.id !== activeId || !layer.visible) continue;
    const packed = packedLayer(pack, layer);
    if (!packed) continue;
    for (let islandIndex = 0; islandIndex < layer.islands.length; islandIndex += 1) {
      const island = layer.islands[islandIndex];
      if (island.area < minArea) continue;
      if ((packed.alpha[islandIndex]?.[pixel] ?? 0) > 40) return island.id;
    }
  }
  return null;
}

type View = { scale: number; x: number; y: number };

const fitted: View = { scale: 1, x: 0, y: 0 };

export function Photo({
  original,
  layers,
  enabled,
  selectedIds,
  minArea,
  selection,
  eraserSize,
  activeId,
  onSelect,
  onErase,
  onLasso,
  onCut,
  onSize,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const packRef = useRef<Pack | null>(null);
  const dragRef = useRef<Box | null>(null);
  const strokeRef = useRef<{ x: number; y: number }[] | null>(null);
  const viewRef = useRef<View>(fitted);
  const [ready, setReady] = useState(0);
  const [box, setBox] = useState<Box | null>(null);
  const [lasso, setLasso] = useState<{ x: number; y: number }[] | null>(null);
  const [cut, setCut] = useState<{ x: number; y: number }[] | null>(null);
  const [ring, setRing] = useState<{ x: number; y: number } | null>(null);
  const [view, setView] = useState<View>(fitted);
  const signature = layers
    .map((layer) => `${layer.id}:${layer.edited}:${layer.islands.map((island) => island.mask).join(",")}`)
    .join("|");

  useEffect(() => {
    let cancel = false;
    packRef.current = null;
    const snapshot = layers;

    Promise.all(
      snapshot.map(async (layer) => {
        const [edited, ...masks] = await Promise.all([
          loadImage(layer.edited),
          ...layer.islands.map((island) => loadImage(island.mask)),
        ]);
        return { layer, edited, masks };
      })
    )
      .then(async (loaded) => {
        const base = await loadImage(original);
        if (cancel) return;
        const width = base.naturalWidth;
        const height = base.naturalHeight;
        onSize(width, height);
        packRef.current = {
          width,
          height,
          base,
          layers: loaded.map(({ layer, edited, masks }) => ({
            id: layer.id,
            cutouts: masks.map((mask) => cutout(edited, mask, width, height)),
            tints: layer.islands.map((island, index) => tint(masks[index], island.color, width, height)),
            alpha: masks.map((mask) => alphaOf(mask, width, height)),
          })),
        };
        setReady((value) => value + 1);
      })
      .catch(() => {
        if (!cancel) packRef.current = null;
      });

    return () => {
      cancel = true;
    };
    // signature covers the images this pack depends on
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [original, signature]);

  useEffect(() => {
    const pack = packRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!pack || !canvas || !ctx) return;

    canvas.width = pack.width;
    canvas.height = pack.height;
    ctx.drawImage(pack.base, 0, 0);

    layers.forEach((layer) => {
      if (!layer.visible) return;
      const packed = packedLayer(pack, layer);
      if (!packed) return;
      layer.islands.forEach((island, index) => {
        if (island.area < minArea || enabled[island.id] === false) return;
        ctx.drawImage(packed.cutouts[index], 0, 0);
      });
    });

    layers.forEach((layer) => {
      if (layer.id !== activeId || !layer.visible) return;
      const packed = packedLayer(pack, layer);
      if (!packed) return;
      layer.islands.forEach((island, index) => {
        if (island.area < minArea || !selectedIds.includes(island.id)) return;
        ctx.save();
        ctx.globalAlpha = 0.45;
        ctx.drawImage(packed.tints[index], 0, 0);
        ctx.restore();
      });
    });

    if (box) {
      const left = Math.min(box.x, box.x2);
      const top = Math.min(box.y, box.y2);
      ctx.save();
      ctx.strokeStyle = "#3d7ee8";
      ctx.lineWidth = 3;
      ctx.strokeRect(left, top, Math.abs(box.x2 - box.x), Math.abs(box.y2 - box.y));
      ctx.restore();
    }

    if (lasso && lasso.length > 1) {
      ctx.beginPath();
      ctx.moveTo(lasso[0].x, lasso[0].y);
      for (const point of lasso.slice(1)) ctx.lineTo(point.x, point.y);
      ctx.closePath();
      ctx.fillStyle = "rgba(61, 126, 232, 0.2)";
      ctx.fill();
      ctx.strokeStyle = "#3d7ee8";
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    if (cut && cut.length > 1) {
      ctx.beginPath();
      ctx.moveTo(cut[0].x, cut[0].y);
      for (const point of cut.slice(1)) ctx.lineTo(point.x, point.y);
      ctx.strokeStyle = "#af52de";
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (ring && selection === "eraser") {
      ctx.beginPath();
      ctx.arc(ring.x, ring.y, Math.max(eraserSize / 2, 1), 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 149, 0, 0.18)";
      ctx.fill();
      ctx.strokeStyle = "#ff9500";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [ready, layers, enabled, selectedIds, minArea, activeId, box, lasso, cut, ring, selection, eraserSize]);

  useEffect(() => {
    viewRef.current = fitted;
    setView(fitted);
  }, [original]);

  useEffect(() => {
    const element = canvasRef.current;
    if (!element) return;
    const canvas: HTMLCanvasElement = element;

    function applyZoom(clientX: number, clientY: number, nextScale: number) {
      const current = viewRef.current;
      const scale = Math.min(8, Math.max(1, nextScale));
      if (scale === 1) {
        viewRef.current = fitted;
        setView(fitted);
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const ratio = scale / current.scale;
      const next = {
        scale,
        x: current.x + (clientX - rect.left) * (1 - ratio),
        y: current.y + (clientY - rect.top) * (1 - ratio),
      };
      viewRef.current = next;
      setView(next);
    }

    function pan(dx: number, dy: number) {
      const current = viewRef.current;
      if (current.scale <= 1) return;
      const minX = canvas.offsetWidth * (1 - current.scale);
      const minY = canvas.offsetHeight * (1 - current.scale);
      const next = {
        scale: current.scale,
        x: Math.min(0, Math.max(minX, current.x - dx)),
        y: Math.min(0, Math.max(minY, current.y - dy)),
      };
      viewRef.current = next;
      setView(next);
    }

    let pinchScale = 1;
    let gesturing = false;
    const fingers = new Map<number, { x: number; y: number }>();

    function onWheel(event: WheelEvent) {
      if (event.ctrlKey) {
        event.preventDefault();
        if (gesturing) return;
        const next = viewRef.current.scale * Math.exp(-event.deltaY * 0.01);
        applyZoom(event.clientX, event.clientY, next);
        return;
      }
      if (viewRef.current.scale <= 1) return;
      event.preventDefault();
      pan(event.deltaX, event.deltaY);
    }
    function onGestureStart(event: Event) {
      event.preventDefault();
      gesturing = true;
      pinchScale = viewRef.current.scale;
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
      fingers.clear();
      for (const touch of Array.from(event.touches)) {
        fingers.set(touch.identifier, { x: touch.clientX, y: touch.clientY });
      }
    }
    function onTouchMove(event: TouchEvent) {
      if (event.touches.length < 2) return;
      event.preventDefault();
      const first = event.touches[0];
      const second = event.touches[1];
      const prevFirst = fingers.get(first.identifier);
      const prevSecond = fingers.get(second.identifier);
      if (prevFirst && prevSecond) {
        const dx = ((first.clientX - prevFirst.x) + (second.clientX - prevSecond.x)) / 2;
        const dy = ((first.clientY - prevFirst.y) + (second.clientY - prevSecond.y)) / 2;
        pan(-dx, -dy);
      }
      fingers.set(first.identifier, { x: first.clientX, y: first.clientY });
      fingers.set(second.identifier, { x: second.clientX, y: second.clientY });
    }
    function onTouchEnd() {
      fingers.clear();
    }

    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("gesturestart", onGestureStart);
    canvas.addEventListener("gesturechange", onGestureChange);
    canvas.addEventListener("gestureend", onGestureEnd);
    canvas.addEventListener("touchstart", onTouchStart, { passive: false });
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });
    canvas.addEventListener("touchend", onTouchEnd);
    canvas.addEventListener("touchcancel", onTouchEnd);
    return () => {
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("gesturestart", onGestureStart);
      canvas.removeEventListener("gesturechange", onGestureChange);
      canvas.removeEventListener("gestureend", onGestureEnd);
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
      canvas.removeEventListener("touchend", onTouchEnd);
      canvas.removeEventListener("touchcancel", onTouchEnd);
    };
  }, []);

  function pick(event: MouseEvent<HTMLCanvasElement>) {
    if (selection !== "tap") return;
    const pack = packRef.current;
    const canvas = canvasRef.current;
    if (!pack || !canvas) return;

    const { x, y } = imagePoint(event, canvas, pack);
    if (x < 0 || y < 0 || x >= pack.width || y >= pack.height) {
      onSelect([]);
      return;
    }
    const hit = hitIsland(pack, layers, minArea, activeId, x, y);
    onSelect(hit ? [hit] : []);
  }

  function startPointer(event: PointerEvent<HTMLCanvasElement>) {
    const pack = packRef.current;
    const canvas = canvasRef.current;
    if (!pack || !canvas || !event.isPrimary) return;
    const point = imagePoint(event, canvas, pack);
    if (selection === "box") {
      const next = { x: point.x, y: point.y, x2: point.x, y2: point.y };
      dragRef.current = next;
      setBox(next);
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }
    if (selection === "lasso") {
      strokeRef.current = [point];
      setLasso([point]);
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }
    if (selection === "scissors") {
      strokeRef.current = [point];
      setCut([point]);
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }
    if (selection !== "eraser") return;
    strokeRef.current = [point];
    setRing(point);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function movePointer(event: PointerEvent<HTMLCanvasElement>) {
    const pack = packRef.current;
    const canvas = canvasRef.current;
    if (!pack || !canvas) return;
    const point = imagePoint(event, canvas, pack);
    if (selection === "eraser") setRing(point);
    const drag = dragRef.current;
    if (drag && selection === "box") {
      const next = { ...drag, x2: point.x, y2: point.y };
      dragRef.current = next;
      setBox(next);
    }
    if (strokeRef.current && (selection === "eraser" || selection === "lasso" || selection === "scissors")) {
      const last = strokeRef.current[strokeRef.current.length - 1];
      if (Math.hypot(point.x - last.x, point.y - last.y) >= 2) {
        strokeRef.current.push(point);
        if (selection === "lasso") setLasso([...strokeRef.current]);
        if (selection === "scissors") setCut([...strokeRef.current]);
      }
    }
  }

  function endPointer() {
    const pack = packRef.current;
    const drag = dragRef.current;
    const stroke = strokeRef.current;
    dragRef.current = null;
    strokeRef.current = null;
    setBox(null);
    setLasso(null);
    setCut(null);
    if (drag && pack && selection === "box") onSelect(islandsInBox(pack, layers, minArea, activeId, drag));
    if (stroke && stroke.length && selection === "eraser") onErase(stroke);
    if (stroke && stroke.length >= 3 && selection === "lasso") onLasso(stroke);
    if (stroke && stroke.length >= 2 && selection === "scissors") onCut(stroke);
  }

  return (
    <canvas
      ref={canvasRef}
      className="photo"
      style={{
        transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
        transformOrigin: "0 0",
      }}
      onClick={pick}
      onPointerDown={startPointer}
      onPointerMove={movePointer}
      onPointerUp={endPointer}
      onPointerLeave={() => {
        if (!strokeRef.current) setRing(null);
      }}
    />
  );
}
