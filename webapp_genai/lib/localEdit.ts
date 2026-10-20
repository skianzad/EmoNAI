import type { Island } from "@/data/edits";
import { objectsFromPaint, type NamedColor } from "@/lib/colorAdjust";
import { COLOR_MAP_PALETTE, guidePrompt } from "@/lib/colorMapPrompt";
import { fitImage, readDataUrl } from "@/lib/shrink";
import { splitIslands } from "@/lib/splitIslands";
import { uid } from "@/lib/uid";

export type Box = { x: number; y: number; w: number; h: number };

export type SegmentObject = {
  id: string;
  name: string;
  keyColor: string;
  tint: string | null;
  shade: number;
  scale: number;
  hidden: boolean;
  dx: number;
  dy: number;
  mask: string;
  box: Box;
  /** Tint strength from 0 to 1. Omitted uses the default half-strength overlay. */
  cover?: number;
  /** Edited pixels for this segment, masked, in the same frame as the original photo. */
  layer?: string;
  /** Blur radius in pixels, used for a background region. */
  blur?: number;
};

const TINT = 0.5;

async function imageBlob(src: string) {
  if (src.startsWith("data:")) {
    const comma = src.indexOf(",");
    const mime = src.slice(5, src.indexOf(";")) || "image/jpeg";
    const binary = atob(src.slice(comma + 1));
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    return new Blob([bytes], { type: mime });
  }
  const response = await fetch(src);
  if (!response.ok) throw new Error("Could not read the photo");
  return response.blob();
}

export async function segmentPhoto(
  editorId: string,
  src: string,
  kind: "objects" | "face" | "room" = "objects",
  guide?: { name: string; left: number; top: number; width: number; height: number }[]
) {
  const blob = await fitImage(await imageBlob(src));
  const dataUrl = await readDataUrl(blob);
  const [head, image] = dataUrl.split(",");
  const response = await fetch("/api/color-map", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      editor: editorId,
      image,
      mime: head.slice(5, head.indexOf(";")) || "image/jpeg",
      kind,
      prompt: guide && guide.length > 0 ? guidePrompt(guide) : undefined,
    }),
  });
  const body = await response.json().catch(() => ({ error: "Edit failed" }));
  if (!response.ok) throw new Error(body.error || "Edit failed");
  const named = Array.isArray(body.items) ? (body.items as NamedColor[]) : [];
  const paintedSrc = String(body.image ?? "");
  const byColor = await segmentsFromColors(
    dataUrl,
    paintedSrc,
    named,
    kind === "objects" ? 400 : 80,
    kind !== "objects",
    kind === "room" ? 110 : 58
  );
  if (byColor.length > 0) return byColor;
  const split = await splitIslands(dataUrl, paintedSrc);
  const painted = await objectsFromPaint(paintedSrc, split.islands, named);
  return Promise.all(
    painted.map(async (object) => ({
      id: object.id,
      name: object.name,
      keyColor: object.color,
      tint: null,
      shade: 0,
      scale: 100,
      hidden: false,
      dx: 0,
      dy: 0,
      mask: object.mask,
      box: await maskBox(object.mask),
    }))
  );
}

/** Cover the whole object with a blur wide enough that its center smears into the surroundings. */
function eraseObject(ctx: CanvasRenderingContext2D, mask: HTMLCanvasElement, width: number, height: number) {
  const probe = document.createElement("canvas");
  probe.width = width;
  probe.height = height;
  const probeCtx = probe.getContext("2d");
  const layer = document.createElement("canvas");
  layer.width = width;
  layer.height = height;
  const layerCtx = layer.getContext("2d");
  if (!probeCtx || !layerCtx) return;
  probeCtx.drawImage(mask, 0, 0);
  const pixels = probeCtx.getImageData(0, 0, width, height).data;
  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;
  for (let i = 3; i < pixels.length; i += 16) {
    if (pixels[i] < 20) continue;
    const pixel = (i - 3) / 4;
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  const radius = Math.max(36, Math.round(Math.max(maxX - minX, maxY - minY) * 0.45));
  layerCtx.filter = `blur(${radius}px)`;
  layerCtx.drawImage(ctx.canvas, 0, 0);
  layerCtx.filter = "none";
  layerCtx.globalCompositeOperation = "destination-in";
  layerCtx.drawImage(mask, 0, 0);
  ctx.drawImage(layer, 0, 0);
}

/** Local color, size, move, hide, and warmth. No editor call. */
export async function renderLocal(originalSrc: string, objects: SegmentObject[], warmth: number) {
  const original = await loadImage(originalSrc);
  const width = original.naturalWidth;
  const height = original.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return originalSrc;
  ctx.drawImage(original, 0, 0);

  for (const object of objects) {
    const cut = await cutout(original, object.mask, width, height);
    if (!cut) continue;
    const scale = object.scale / 100;
    if (object.layer) {
      if (object.hidden) continue;
      const sprite = await loadImage(object.layer);
      const moved = scale !== 1 || object.dx !== 0 || object.dy !== 0;
      if (moved && cut) {
        ctx.save();
        ctx.globalCompositeOperation = "destination-out";
        ctx.drawImage(cut.canvas, 0, 0);
        ctx.restore();
      }
      ctx.save();
      if (cut) {
        ctx.translate(cut.cx + object.dx * width, cut.cy + object.dy * height);
        ctx.scale(scale, scale);
        ctx.translate(-cut.cx, -cut.cy);
      }
      ctx.drawImage(sprite, 0, 0, width, height);
      if (object.tint) {
        ctx.globalAlpha = object.cover ?? TINT;
        ctx.drawImage(await flatMask(object.mask, shadeHex(object.tint, object.shade || 0), width, height), 0, 0);
      }
      ctx.restore();
      continue;
    }
    if (object.hidden) {
      eraseObject(ctx, cut.canvas, width, height);
      continue;
    }
    const moved = scale !== 1 || object.dx !== 0 || object.dy !== 0;
    if (moved) {
      ctx.save();
      ctx.globalCompositeOperation = "destination-out";
      ctx.drawImage(cut.canvas, 0, 0);
      ctx.restore();
      ctx.save();
      ctx.filter = "blur(18px)";
      ctx.globalCompositeOperation = "destination-over";
      ctx.drawImage(original, 0, 0, width, height);
      ctx.restore();
    }
    if (object.hidden) continue;
    ctx.save();
    ctx.translate(cut.cx + object.dx * width, cut.cy + object.dy * height);
    ctx.scale(scale, scale);
    ctx.translate(-cut.cx, -cut.cy);
    ctx.drawImage(cut.canvas, 0, 0);
    if (object.blur && object.blur > 0) {
      const layer = document.createElement("canvas");
      layer.width = width;
      layer.height = height;
      const layerCtx = layer.getContext("2d");
      if (layerCtx) {
        layerCtx.filter = `blur(${object.blur}px)`;
        layerCtx.drawImage(original, 0, 0, width, height);
        layerCtx.filter = "none";
        layerCtx.globalCompositeOperation = "destination-in";
        layerCtx.drawImage(cut.canvas, 0, 0);
        ctx.drawImage(layer, 0, 0);
      }
    }
    if (object.tint) {
      ctx.globalAlpha = object.cover ?? TINT;
      ctx.drawImage(await flatMask(object.mask, shadeHex(object.tint, object.shade || 0), width, height), 0, 0);
    }
    ctx.restore();
  }

  if (warmth !== 0) {
    ctx.fillStyle = warmth > 0 ? `rgba(255, 154, 72, ${warmth})` : `rgba(86, 140, 255, ${-warmth})`;
    ctx.fillRect(0, 0, width, height);
  }
  return canvas.toDataURL("image/jpeg", 0.9);
}

async function segmentsFromColors(originalSrc: string, paintedSrc: string, named: NamedColor[], minArea = 400, keepAll = false, match = 58) {
  const [original, painted] = await Promise.all([loadImage(originalSrc), loadImage(paintedSrc)]);
  const width = original.naturalWidth;
  const height = original.naturalHeight;
  const before = imagePixels(original, width, height);
  const after = imagePixels(painted, width, height);
  const targets = [...named.map((item) => item.color), ...COLOR_MAP_PALETTE].filter(
    (color, index, all) => all.indexOf(color) === index
  );
  const owner = new Int16Array(width * height).fill(-1);
  for (let i = 0; i < owner.length; i++) {
    const offset = i * 4;
    const changed = Math.max(
      Math.abs(before[offset] - after[offset]),
      Math.abs(before[offset + 1] - after[offset + 1]),
      Math.abs(before[offset + 2] - after[offset + 2])
    );
    if (changed < 24) continue;
    let best = -1;
    let bestDist = match * match;
    targets.forEach((color, index) => {
      const [r, g, b] = hexRgb(color);
      const dist = (after[offset] - r) ** 2 + (after[offset + 1] - g) ** 2 + (after[offset + 2] - b) ** 2;
      if (dist < bestDist) {
        best = index;
        bestDist = dist;
      }
    });
    owner[i] = best;
  }

  const found = [];
  for (let index = 0; index < targets.length; index++) {
    const component = keepAll ? allPixels(owner, index) : largestBlob(owner, index, width, height);
    if (component.area < minArea) continue;
    const color = targets[index];
    const listed = named.find((item) => colorDistance(item.color, color) < 70 * 70);
    found.push({
      id: uid(),
      name: listed?.name ?? "Object",
      keyColor: listed?.color ?? color,
      tint: null,
      shade: 0,
      scale: 100,
      hidden: false,
      dx: 0,
      dy: 0,
      mask: maskUrl(component.pixels, width, height),
      box: boxOf(component.pixels, width, height),
      area: component.area,
    });
  }
  return found
    .sort((a, b) => b.area - a.area)
    .slice(0, 8)
    .map(({ area: _area, ...object }) => object);
}

function allPixels(owner: Int16Array, colorIndex: number) {
  const pixels = new Uint8Array(owner.length);
  let area = 0;
  for (let i = 0; i < owner.length; i++) {
    if (owner[i] !== colorIndex) continue;
    pixels[i] = 1;
    area += 1;
  }
  return { area, pixels };
}

function largestBlob(owner: Int16Array, colorIndex: number, width: number, height: number) {
  const seen = new Uint8Array(owner.length);
  let bestArea = 0;
  let best: Uint8Array | null = null;
  const stack: number[] = [];
  for (let start = 0; start < owner.length; start++) {
    if (owner[start] !== colorIndex || seen[start]) continue;
    const pixels = new Uint8Array(owner.length);
    let area = 0;
    stack.push(start);
    seen[start] = 1;
    while (stack.length) {
      const point = stack.pop() as number;
      pixels[point] = 1;
      area += 1;
      const x = point % width;
      const neighbors = [point - 1, point + 1, point - width, point + width];
      if (x === 0) neighbors[0] = -1;
      if (x === width - 1) neighbors[1] = -1;
      for (const neighbor of neighbors) {
        if (neighbor < 0 || neighbor >= owner.length || seen[neighbor] || owner[neighbor] !== colorIndex) continue;
        seen[neighbor] = 1;
        stack.push(neighbor);
      }
    }
    if (area > bestArea) {
      bestArea = area;
      best = pixels;
    }
  }
  return { area: bestArea, pixels: best ?? new Uint8Array(owner.length) };
}

function maskUrl(pixels: Uint8Array, width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  const image = ctx.createImageData(width, height);
  for (let i = 0; i < pixels.length; i++) {
    if (pixels[i]) image.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas.toDataURL("image/png");
}

function boxOf(pixels: Uint8Array, width: number, height: number): Box {
  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;
  for (let i = 0; i < pixels.length; i++) {
    if (!pixels[i]) continue;
    const x = i % width;
    const y = Math.floor(i / width);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  if (maxX <= minX) return { x: 0.3, y: 0.3, w: 0.4, h: 0.4 };
  return {
    x: minX / width,
    y: minY / height,
    w: Math.max(0.04, (maxX - minX) / width),
    h: Math.max(0.04, (maxY - minY) / height),
  };
}

function imagePixels(image: HTMLImageElement, width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return new Uint8ClampedArray(width * height * 4);
  ctx.drawImage(image, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height).data;
}

function colorDistance(a: string, b: string) {
  const [ar, ag, ab] = hexRgb(a);
  const [br, bg, bb] = hexRgb(b);
  return (ar - br) ** 2 + (ag - bg) ** 2 + (ab - bb) ** 2;
}

export async function fitEdited(baseSrc: string, editedSrc: string) {
  const [base, edited] = await Promise.all([loadImage(baseSrc), loadImage(editedSrc)]);
  const canvas = document.createElement("canvas");
  canvas.width = base.naturalWidth;
  canvas.height = base.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return editedSrc;
  ctx.drawImage(edited, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.9);
}

export async function segmentsFromIslands(editedSrc: string, islands: Island[], hints: SegmentObject[]) {
  const edited = await loadImage(editedSrc);
  const width = edited.naturalWidth;
  const height = edited.naturalHeight;
  const used = new Set<string>();
  const segments: SegmentObject[] = [];
  for (const [index, island] of islands.entries()) {
    const mask = await loadImage(island.mask);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) continue;
    ctx.drawImage(edited, 0, 0, width, height);
    ctx.globalCompositeOperation = "destination-in";
    ctx.drawImage(mask, 0, 0, width, height);
    const box = await maskBox(island.mask);
    const cx = box.x + box.w / 2;
    const cy = box.y + box.h / 2;
    const hint = hints.find((object) => {
      if (used.has(object.id)) return false;
      return cx >= object.box.x && cy >= object.box.y && cx <= object.box.x + object.box.w && cy <= object.box.y + object.box.h;
    });
    if (hint) used.add(hint.id);
    segments.push({
      id: island.id,
      name: hint?.name ?? (island.label.startsWith("Outside") ? island.label : `Change ${index + 1}`),
      keyColor: island.color,
      tint: null,
      shade: 0,
      scale: 100,
      hidden: false,
      dx: 0,
      dy: 0,
      mask: island.mask,
      box,
      layer: canvas.toDataURL("image/png"),
    });
  }
  return segments;
}

async function maskBox(maskSrc: string): Promise<Box> {
  const mask = await loadImage(maskSrc);
  const canvas = document.createElement("canvas");
  canvas.width = mask.naturalWidth;
  canvas.height = mask.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx || canvas.width === 0) return { x: 0.3, y: 0.3, w: 0.4, h: 0.4 };
  ctx.drawImage(mask, 0, 0);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  let minX = canvas.width;
  let minY = canvas.height;
  let maxX = 0;
  let maxY = 0;
  let count = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i + 3] < 128) continue;
    const pixel = i / 4;
    const x = pixel % canvas.width;
    const y = Math.floor(pixel / canvas.width);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    count += 1;
  }
  if (count === 0) return { x: 0.3, y: 0.3, w: 0.4, h: 0.4 };
  return {
    x: minX / canvas.width,
    y: minY / canvas.height,
    w: Math.max(0.04, (maxX - minX) / canvas.width),
    h: Math.max(0.04, (maxY - minY) / canvas.height),
  };
}

async function cutout(original: HTMLImageElement, maskSrc: string, width: number, height: number) {
  const mask = await loadImage(maskSrc);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(original, 0, 0, width, height);
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(mask, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height).data;
  let sx = 0;
  let sy = 0;
  let count = 0;
  for (let i = 0; i < pixels.length; i += 16) {
    if (pixels[i + 3] < 128) continue;
    const pixel = i / 4;
    sx += pixel % width;
    sy += Math.floor(pixel / width);
    count += 1;
  }
  if (count === 0) return null;
  return { canvas, cx: sx / count, cy: sy / count };
}

async function flatMask(maskSrc: string, color: string, width: number, height: number) {
  const mask = await loadImage(maskSrc);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;
  ctx.drawImage(mask, 0, 0, width, height);
  const image = ctx.getImageData(0, 0, width, height);
  const [r, g, b] = hexRgb(color);
  for (let i = 0; i < image.data.length; i += 4) {
    if (image.data[i + 3] < 128) {
      image.data[i + 3] = 0;
      continue;
    }
    image.data[i] = r;
    image.data[i + 1] = g;
    image.data[i + 2] = b;
    image.data[i + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

function shadeHex(value: string, shade: number) {
  const [r, g, b] = hexRgb(value);
  const mix = (channel: number) => {
    if (shade > 0) return Math.round(channel + (255 - channel) * shade * 0.18);
    return Math.round(channel * (1 + shade * 0.22));
  };
  return `#${[mix(r), mix(g), mix(b)].map((channel) => Math.max(0, Math.min(255, channel)).toString(16).padStart(2, "0")).join("")}`;
}

function hexRgb(value: string): [number, number, number] {
  const hex = value.replace("#", "");
  const safe = hex.length === 6 ? hex : "888888";
  return [parseInt(safe.slice(0, 2), 16), parseInt(safe.slice(2, 4), 16), parseInt(safe.slice(4, 6), 16)];
}

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not read the photo"));
    image.src = src;
  });
}
