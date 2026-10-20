import type { Island } from "@/data/edits";
import type { Layer } from "@/data/versions";
import { renderComposite } from "@/lib/composite";

export async function previewEdit(
  source: string,
  edited: string,
  islands: Island[],
  kept: Record<string, boolean>,
  strength: number,
  shade: boolean,
  tint: string | null
) {
  const layer: Layer = {
    id: "draft",
    parentId: null,
    prompt: "",
    visible: true,
    edited,
    islands,
  };
  const composited = await renderComposite(source, [layer], kept, 0);
  if (strength >= 100 && !shade && !tint) return composited;
  return grade(source, composited, strength, shade, tint);
}

export async function maskAnchor(mask: string) {
  const image = await loadImage(mask);
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx || canvas.width === 0) return { x: 0.5, y: 0.5 };
  ctx.drawImage(image, 0, 0);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  let sx = 0;
  let sy = 0;
  let count = 0;
  for (let i = 0; i < pixels.length; i += 16) {
    if (pixels[i + 3] < 128) continue;
    const pixel = i / 4;
    sx += pixel % canvas.width;
    sy += Math.floor(pixel / canvas.width);
    count += 1;
  }
  if (count === 0) return { x: 0.5, y: 0.5 };
  return { x: sx / count / canvas.width, y: sy / count / canvas.height };
}

async function grade(source: string, edited: string, strength: number, shade: boolean, tint: string | null) {
  const [baseImage, editImage] = await Promise.all([loadImage(source), loadImage(edited)]);
  const width = baseImage.naturalWidth;
  const height = baseImage.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return edited;
  ctx.drawImage(baseImage, 0, 0, width, height);
  const base = ctx.getImageData(0, 0, width, height);
  ctx.drawImage(editImage, 0, 0, width, height);
  const over = ctx.getImageData(0, 0, width, height);
  const mix = Math.min(1, Math.max(0, strength / 100));
  const tintRgb = tint ? hexRgb(tint) : null;

  for (let i = 0; i < base.data.length; i += 4) {
    let r = base.data[i] + (over.data[i] - base.data[i]) * mix;
    let g = base.data[i + 1] + (over.data[i + 1] - base.data[i + 1]) * mix;
    let b = base.data[i + 2] + (over.data[i + 2] - base.data[i + 2]) * mix;
    const changed =
      Math.max(
        Math.abs(base.data[i] - over.data[i]),
        Math.abs(base.data[i + 1] - over.data[i + 1]),
        Math.abs(base.data[i + 2] - over.data[i + 2])
      ) > 28;
    if (changed && shade) {
      const avg = (r + g + b) / 3;
      r += (avg - r) * 0.55;
      g += (avg - g) * 0.55;
      b += (avg - b) * 0.55;
    }
    if (changed && tintRgb) {
      r = r * 0.42 + tintRgb[0] * 0.58;
      g = g * 0.42 + tintRgb[1] * 0.58;
      b = b * 0.42 + tintRgb[2] * 0.58;
    }
    over.data[i] = r;
    over.data[i + 1] = g;
    over.data[i + 2] = b;
  }
  ctx.putImageData(over, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.86);
}

function hexRgb(value: string): [number, number, number] {
  const hex = value.replace("#", "");
  return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)];
}

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not read the photo"));
    image.src = src;
  });
}
