import type { Island } from "@/data/edits";
import { uid } from "./uid";

const colors = ["#e15b4c", "#3d7ee8", "#2f9d62", "#e0a23a", "#8b6ad6"];

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not read the edited photo"));
    image.src = src;
  });
}

function solidify(changed: Uint8Array, width: number, height: number) {
  const step = 4;
  const smallWidth = Math.ceil(width / step);
  const smallHeight = Math.ceil(height / step);
  const small = new Uint8Array(smallWidth * smallHeight);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (changed[y * width + x]) small[Math.floor(y / step) * smallWidth + Math.floor(x / step)] = 1;
    }
  }
  const grown = dilate(small, smallWidth, smallHeight, 2);
  fillHoles(grown, smallWidth, smallHeight);
  const shrunk = erode(grown, smallWidth, smallHeight, 1);
  changed.fill(0);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (shrunk[Math.floor(y / step) * smallWidth + Math.floor(x / step)]) changed[y * width + x] = 1;
    }
  }
}

function dilate(mask: Uint8Array, width: number, height: number, radius: number) {
  const next = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let on = 0;
      for (let dy = -radius; dy <= radius && !on; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          if (mask[ny * width + nx]) {
            on = 1;
            break;
          }
        }
      }
      next[y * width + x] = on;
    }
  }
  return next;
}

function erode(mask: Uint8Array, width: number, height: number, radius: number) {
  const next = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let on = 1;
      for (let dy = -radius; dy <= radius && on; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height || !mask[ny * width + nx]) {
            on = 0;
            break;
          }
        }
      }
      next[y * width + x] = on;
    }
  }
  return next;
}

function fillHoles(mask: Uint8Array, width: number, height: number) {
  const seen = new Uint8Array(mask.length);
  const stack: number[] = [];
  const push = (index: number) => {
    if (index < 0 || index >= mask.length || mask[index] || seen[index]) return;
    seen[index] = 1;
    stack.push(index);
  };
  for (let x = 0; x < width; x += 1) {
    push(x);
    push((height - 1) * width + x);
  }
  for (let y = 0; y < height; y += 1) {
    push(y * width);
    push(y * width + width - 1);
  }
  while (stack.length) {
    const point = stack.pop() as number;
    const x = point % width;
    if (x > 0) push(point - 1);
    if (x < width - 1) push(point + 1);
    push(point - width);
    push(point + width);
  }
  for (let index = 0; index < mask.length; index += 1) {
    if (!mask[index] && !seen[index]) mask[index] = 1;
  }
}

function maskUrl(labels: Int32Array, id: number, width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  const image = ctx.createImageData(width, height);
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === id) image.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas.toDataURL("image/png");
}

export async function splitIslands(
  originalSrc: string,
  editedSrc: string
): Promise<{ edited: string; islands: Island[] }> {
  const [original, editedImage] = await Promise.all([loadImage(originalSrc), loadImage(editedSrc)]);
  const width = original.naturalWidth;
  const height = original.naturalHeight;
  const beforeCanvas = document.createElement("canvas");
  beforeCanvas.width = width;
  beforeCanvas.height = height;
  const beforeCtx = beforeCanvas.getContext("2d");
  const afterCanvas = document.createElement("canvas");
  afterCanvas.width = width;
  afterCanvas.height = height;
  const afterCtx = afterCanvas.getContext("2d");
  if (!beforeCtx || !afterCtx) throw new Error("Could not read the edited photo");
  beforeCtx.drawImage(original, 0, 0, width, height);
  afterCtx.drawImage(editedImage, 0, 0, width, height);
  const before = beforeCtx.getImageData(0, 0, width, height).data;
  const after = afterCtx.getImageData(0, 0, width, height).data;
  const changed = new Uint8Array(width * height);

  for (let i = 0; i < changed.length; i++) {
    const offset = i * 4;
    const delta = Math.max(
      Math.abs(before[offset] - after[offset]),
      Math.abs(before[offset + 1] - after[offset + 1]),
      Math.abs(before[offset + 2] - after[offset + 2])
    );
    if (delta > 28) changed[i] = 1;
  }
  solidify(changed, width, height);

  const labels = new Int32Array(changed.length);
  const areas: number[] = [];
  let next = 0;
  const stack: number[] = [];

  for (let start = 0; start < changed.length; start++) {
    if (!changed[start] || labels[start]) continue;
    next += 1;
    let area = 0;
    labels[start] = next;
    stack.push(start);
    while (stack.length) {
      const point = stack.pop() as number;
      area += 1;
      const x = point % width;
      const neighbors = [point - 1, point + 1, point - width, point + width];
      if (x === 0) neighbors[0] = -1;
      if (x === width - 1) neighbors[1] = -1;
      for (const neighbor of neighbors) {
        if (neighbor < 0 || neighbor >= changed.length || !changed[neighbor] || labels[neighbor]) continue;
        labels[neighbor] = next;
        stack.push(neighbor);
      }
    }
    areas[next] = area;
  }

  const kept = areas
    .map((area, id) => ({ id, area }))
    .filter((item) => item.area >= 400)
    .sort((a, b) => b.area - a.area)
    .slice(0, 5);

  const aligned = afterCanvas.toDataURL("image/jpeg", 0.86);

  if (kept.length === 0) {
    const full = new Int32Array(changed.length).fill(1);
    return {
      edited: aligned,
      islands: [
        {
          id: uid(),
          label: "Island 1",
          mask: maskUrl(full, 1, width, height),
          color: colors[0],
          area: changed.length,
        },
      ],
    };
  }

  return {
    edited: aligned,
    islands: kept.map((item, index) => ({
      id: uid(),
      label: `Island ${index + 1}`,
      mask: maskUrl(labels, item.id, width, height),
      color: colors[index % colors.length],
      area: item.area,
    })),
  };
}

/** Pixel changes that sit outside the removed objects and the other segments. */
export async function diffOutside(originalSrc: string, editedSrc: string, covered: string[]) {
  const [original, editedImage] = await Promise.all([loadImage(originalSrc), loadImage(editedSrc)]);
  const width = original.naturalWidth;
  const height = original.naturalHeight;
  const beforeCanvas = document.createElement("canvas");
  beforeCanvas.width = width;
  beforeCanvas.height = height;
  const afterCanvas = document.createElement("canvas");
  afterCanvas.width = width;
  afterCanvas.height = height;
  const beforeCtx = beforeCanvas.getContext("2d");
  const afterCtx = afterCanvas.getContext("2d");
  if (!beforeCtx || !afterCtx) return [];
  beforeCtx.drawImage(original, 0, 0, width, height);
  afterCtx.drawImage(editedImage, 0, 0, width, height);
  const before = beforeCtx.getImageData(0, 0, width, height).data;
  const after = afterCtx.getImageData(0, 0, width, height).data;
  const changed = new Uint8Array(width * height);
  for (let i = 0; i < changed.length; i += 1) {
    const offset = i * 4;
    const delta = Math.max(
      Math.abs(before[offset] - after[offset]),
      Math.abs(before[offset + 1] - after[offset + 1]),
      Math.abs(before[offset + 2] - after[offset + 2])
    );
    if (delta > 28) changed[i] = 1;
  }
  for (const mask of covered) {
    const alpha = await maskAlpha(mask, width, height);
    for (let i = 0; i < changed.length; i += 1) {
      if (alpha[i]) changed[i] = 0;
    }
  }
  // Same cleanup as splitIslands so the new-object mask is a solid region.
  solidify(changed, width, height);

  const labels = new Int32Array(changed.length);
  const areas: number[] = [];
  let next = 0;
  const stack: number[] = [];
  for (let start = 0; start < changed.length; start += 1) {
    if (!changed[start] || labels[start]) continue;
    next += 1;
    let area = 0;
    labels[start] = next;
    stack.push(start);
    while (stack.length) {
      const point = stack.pop() as number;
      area += 1;
      const x = point % width;
      const neighbors = [point - 1, point + 1, point - width, point + width];
      if (x === 0) neighbors[0] = -1;
      if (x === width - 1) neighbors[1] = -1;
      for (const neighbor of neighbors) {
        if (neighbor < 0 || neighbor >= changed.length || !changed[neighbor] || labels[neighbor]) continue;
        labels[neighbor] = next;
        stack.push(neighbor);
      }
    }
    areas[next] = area;
  }

  return areas
    .map((area, id) => ({ id, area }))
    .filter((item) => item.area >= 24)
    .sort((a, b) => b.area - a.area)
    .slice(0, 8)
    .map((item, index) => ({
      id: uid(),
      label: `Outside ${index + 1}`,
      mask: maskUrl(labels, item.id, width, height),
      color: "#d7c15a",
      area: item.area,
    }));
}

async function maskAlpha(src: string, width: number, height: number) {
  const image = await loadImage(src);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const alpha = new Uint8Array(width * height);
  if (!ctx) return alpha;
  ctx.drawImage(image, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height).data;
  for (let i = 0; i < alpha.length; i += 1) alpha[i] = pixels[i * 4 + 3] > 24 ? 1 : 0;
  return alpha;
}
