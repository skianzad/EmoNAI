import type { Island } from "@/data/edits";

const colors = ["#e15b4c", "#3d7ee8", "#2f9d62", "#e0a23a", "#8b6ad6"];

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not read the region"));
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

function paintBarrier(width: number, height: number, points: { x: number; y: number }[]) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return new Uint8Array(width * height);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  ctx.stroke();
  const pixels = ctx.getImageData(0, 0, width, height).data;
  const barrier = new Uint8Array(width * height);
  for (let i = 0; i < barrier.length; i++) {
    if (pixels[i * 4 + 3] > 40) barrier[i] = 1;
  }
  return barrier;
}

function strokeHits(data: Uint8ClampedArray, width: number, height: number, barrier: Uint8Array) {
  for (let i = 0; i < barrier.length; i++) {
    if (barrier[i] && data[i * 4 + 3] > 40) return true;
  }
  return false;
}

/** Split one island along a drawn cut. Every pixel stays; nothing is erased. */
export async function cutIslands(islands: Island[], points: { x: number; y: number }[], selectedIds: string[]) {
  if (points.length < 2) return null;
  const drawn = await Promise.all(
    islands.map(async (island) => {
      const image = await loadImage(island.mask);
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(image, 0, 0);
      const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
      return { island, canvas, pixels };
    })
  );
  const ready = drawn.filter((item): item is NonNullable<typeof item> => item !== null);
  if (ready.length === 0) return null;

  const width = ready[0].canvas.width;
  const height = ready[0].canvas.height;
  const barrier = paintBarrier(width, height, points);

  const chosen = new Set(selectedIds);
  if (chosen.size === 0) {
    let bestId = "";
    let best = 0;
    for (const item of ready) {
      if (!strokeHits(item.pixels.data, width, height, barrier)) continue;
      if (item.island.area > best) {
        best = item.island.area;
        bestId = item.island.id;
      }
    }
    if (!bestId) return null;
    chosen.add(bestId);
  }

  const colorStart = islands.length;
  const next: Island[] = [];
  let changed = false;
  let piece = 0;

  for (const item of ready) {
    if (!chosen.has(item.island.id) || !strokeHits(item.pixels.data, width, height, barrier)) {
      next.push(item.island);
      continue;
    }

    const alpha = item.pixels.data;
    const labels = new Int32Array(width * height);
    const sizes: number[] = [0];
    let nextId = 1;

    for (let i = 0; i < labels.length; i++) {
      if (alpha[i * 4 + 3] <= 40 || barrier[i] || labels[i]) continue;
      const id = nextId++;
      sizes[id] = 0;
      const stack = [i];
      labels[i] = id;
      while (stack.length) {
        const cur = stack.pop()!;
        sizes[id] += 1;
        const cx = cur % width;
        const cy = (cur / width) | 0;
        for (const [dx, dy] of [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1],
        ] as const) {
          const nx = cx + dx;
          const ny = cy + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const ni = ny * width + nx;
          if (alpha[ni * 4 + 3] > 40 && !barrier[ni] && !labels[ni]) {
            labels[ni] = id;
            stack.push(ni);
          }
        }
      }
    }

    const parts = sizes
      .map((area, id) => ({ id, area }))
      .filter((part) => part.id > 0 && part.area > 0)
      .sort((a, b) => b.area - a.area);

    if (parts.length < 2) {
      next.push(item.island);
      continue;
    }

    // Put the cut pixels back by giving each one to the nearest piece.
    let growing = true;
    while (growing) {
      growing = false;
      for (let i = 0; i < labels.length; i++) {
        if (alpha[i * 4 + 3] <= 40 || labels[i]) continue;
        const cx = i % width;
        const cy = (i / width) | 0;
        let best = 0;
        for (const [dx, dy] of [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1],
        ] as const) {
          const nx = cx + dx;
          const ny = cy + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const label = labels[ny * width + nx];
          if (label && (!best || sizes[label] > sizes[best])) best = label;
        }
        if (best) {
          labels[i] = best;
          sizes[best] += 1;
          growing = true;
        }
      }
    }

    changed = true;
    parts.forEach((part, index) => {
      let area = 0;
      for (let i = 0; i < labels.length; i++) {
        if (labels[i] === part.id) area += 1;
      }
      next.push({
        id: crypto.randomUUID(),
        label: parts.length === 2 ? (index === 0 ? "Cut A" : "Cut B") : `Cut ${index + 1}`,
        mask: maskUrl(labels, part.id, width, height),
        color: colors[(colorStart + piece++) % colors.length],
        area,
      });
    });
  }

  if (!changed) return null;
  return next;
}
