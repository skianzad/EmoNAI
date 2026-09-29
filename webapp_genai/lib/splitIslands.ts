import type { Island } from "@/data/edits";

const colors = ["#e15b4c", "#3d7ee8", "#2f9d62", "#e0a23a", "#8b6ad6"];

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not read the edited photo"));
    image.src = src;
  });
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
          id: crypto.randomUUID(),
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
      id: crypto.randomUUID(),
      label: `Island ${index + 1}`,
      mask: maskUrl(labels, item.id, width, height),
      color: colors[index % colors.length],
      area: item.area,
    })),
  };
}
