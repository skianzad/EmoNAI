import type { Island } from "@/data/edits";

export type NamedColor = { name: string; color: string };

export type ColorObject = {
  id: string;
  name: string;
  color: string;
  scale: number;
  mask: string;
};

export async function objectsFromPaint(
  paintedSrc: string,
  islands: Island[],
  named: NamedColor[]
): Promise<ColorObject[]> {
  const painted = await loadImage(paintedSrc);
  const canvas = document.createElement("canvas");
  canvas.width = painted.naturalWidth;
  canvas.height = painted.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return [];
  ctx.drawImage(painted, 0, 0);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const used = new Set<number>();

  const objects: ColorObject[] = [];
  for (const [index, island] of islands.entries()) {
    const sampled = await sampleColor(island.mask, pixels, canvas.width, canvas.height);
    const match = closestName(sampled, named, used);
    objects.push({
      id: island.id,
      name: match?.name ?? `Object ${index + 1}`,
      color: match?.color ?? sampled ?? island.color,
      scale: 100,
      mask: island.mask,
    });
  }
  return objects;
}

/** Flat-color objects on the original photo. Scale is a percent around each object's center. */
export async function renderColorObjects(originalSrc: string, objects: ColorObject[]) {
  const original = await loadImage(originalSrc);
  const canvas = document.createElement("canvas");
  canvas.width = original.naturalWidth;
  canvas.height = original.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return originalSrc;
  ctx.drawImage(original, 0, 0);

  for (const object of objects) {
    const sprite = await coloredSprite(object.mask, object.color, canvas.width, canvas.height);
    if (!sprite) continue;
    const point = sprite.center;
    const scale = object.scale / 100;
    ctx.save();
    ctx.translate(point.x, point.y);
    ctx.scale(scale, scale);
    ctx.translate(-point.x, -point.y);
    ctx.drawImage(sprite.canvas, 0, 0);
    ctx.restore();
  }
  return canvas.toDataURL("image/jpeg", 0.9);
}

function closestName(sampled: string | null, named: NamedColor[], used: Set<number>) {
  if (!sampled || named.length === 0) return null;
  let best = -1;
  let bestDist = 80 * 80;
  named.forEach((item, index) => {
    if (used.has(index)) return;
    const dist = colorDistance(sampled, item.color);
    if (dist < bestDist) {
      best = index;
      bestDist = dist;
    }
  });
  if (best < 0) return null;
  used.add(best);
  return named[best];
}

async function sampleColor(maskSrc: string, pixels: Uint8ClampedArray, width: number, height: number) {
  const mask = await loadImage(maskSrc);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(mask, 0, 0, width, height);
  const alpha = ctx.getImageData(0, 0, width, height).data;
  let r = 0;
  let g = 0;
  let b = 0;
  let count = 0;
  for (let i = 0; i < alpha.length; i += 16) {
    if (alpha[i + 3] < 128) continue;
    r += pixels[i];
    g += pixels[i + 1];
    b += pixels[i + 2];
    count += 1;
  }
  if (count === 0) return null;
  return rgbHex(Math.round(r / count), Math.round(g / count), Math.round(b / count));
}

async function coloredSprite(maskSrc: string, color: string, width: number, height: number) {
  const mask = await loadImage(maskSrc);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(mask, 0, 0, width, height);
  const image = ctx.getImageData(0, 0, width, height);
  const [r, g, b] = hexRgb(color);
  let sx = 0;
  let sy = 0;
  let count = 0;
  for (let i = 0; i < image.data.length; i += 4) {
    if (image.data[i + 3] < 128) {
      image.data[i + 3] = 0;
      continue;
    }
    const pixel = i / 4;
    sx += pixel % width;
    sy += Math.floor(pixel / width);
    count += 1;
    image.data[i] = r;
    image.data[i + 1] = g;
    image.data[i + 2] = b;
    image.data[i + 3] = 255;
  }
  if (count === 0) return null;
  ctx.putImageData(image, 0, 0);
  return { canvas, center: { x: sx / count, y: sy / count } };
}

function colorDistance(a: string, b: string) {
  const [ar, ag, ab] = hexRgb(a);
  const [br, bg, bb] = hexRgb(b);
  return (ar - br) ** 2 + (ag - bg) ** 2 + (ab - bb) ** 2;
}

function hexRgb(value: string): [number, number, number] {
  const hex = value.replace("#", "");
  const safe = hex.length === 6 ? hex : "888888";
  return [parseInt(safe.slice(0, 2), 16), parseInt(safe.slice(2, 4), 16), parseInt(safe.slice(4, 6), 16)];
}

function rgbHex(r: number, g: number, b: number) {
  return `#${[r, g, b].map((channel) => channel.toString(16).padStart(2, "0")).join("").toUpperCase()}`;
}

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not read the photo"));
    image.src = src;
  });
}
