import type { Island } from "@/data/edits";

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not read the region"));
    image.src = src;
  });
}

function insidePolygon(x: number, y: number, points: { x: number; y: number }[]) {
  let hit = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const a = points[i];
    const b = points[j];
    if ((a.y > y) !== (b.y > y) && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) hit = !hit;
  }
  return hit;
}

/** Keep the lassoed part of the island and drop the rest of it. */
export async function keepInsideLasso(
  islands: Island[],
  points: { x: number; y: number }[],
  selectedIds: string[]
) {
  if (points.length < 3) return null;
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
      return { island, canvas, ctx, pixels };
    })
  );
  const ready = drawn.filter((item): item is NonNullable<typeof item> => item !== null);

  const chosen = new Set(selectedIds);
  if (chosen.size === 0) {
    let bestId = "";
    let best = 0;
    for (const item of ready) {
      const count = overlapCount(item.pixels.data, item.canvas.width, item.canvas.height, points);
      if (count > best) {
        best = count;
        bestId = item.island.id;
      }
    }
    if (!bestId) return null;
    chosen.add(bestId);
  }

  let changed = false;
  const next = ready.map((item) => {
    if (!chosen.has(item.island.id)) return item.island;
    if (overlapCount(item.pixels.data, item.canvas.width, item.canvas.height, points) === 0) return item.island;
    const { ctx, canvas } = item;
    ctx.globalCompositeOperation = "destination-in";
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
    ctx.closePath();
    ctx.fill();
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let area = 0;
    for (let i = 3; i < pixels.length; i += 4) {
      if (pixels[i] > 40) area += 1;
    }
    changed = true;
    return { ...item.island, mask: canvas.toDataURL("image/png"), area };
  });
  if (!changed) return null;
  const kept = new Map(next.map((island) => [island.id, island]));
  return islands.map((island) => kept.get(island.id) ?? island).filter((island) => island.area > 0);
}

function overlapCount(data: Uint8ClampedArray, width: number, height: number, points: { x: number; y: number }[]) {
  const minX = Math.max(0, Math.floor(Math.min(...points.map((point) => point.x))));
  const maxX = Math.min(width - 1, Math.ceil(Math.max(...points.map((point) => point.x))));
  const minY = Math.max(0, Math.floor(Math.min(...points.map((point) => point.y))));
  const maxY = Math.min(height - 1, Math.ceil(Math.max(...points.map((point) => point.y))));
  let count = 0;
  for (let y = minY; y <= maxY; y += 2) {
    for (let x = minX; x <= maxX; x += 2) {
      if (data[(y * width + x) * 4 + 3] > 40 && insidePolygon(x, y, points)) count += 1;
    }
  }
  return count;
}
