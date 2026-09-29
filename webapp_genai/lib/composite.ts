import type { Island } from "@/data/edits";
import type { Layer } from "@/data/versions";

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not read the photo"));
    image.src = src;
  });
}

function paintIsland(edited: HTMLImageElement, mask: HTMLImageElement, width: number, height: number) {
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

function keptIslands(islands: Island[], enabled: Record<string, boolean>, minArea: number) {
  return islands.filter((island) => island.area >= minArea && enabled[island.id] !== false);
}

/** Original, then each visible layer painted only inside its islands. */
export async function renderComposite(
  originalSrc: string,
  layers: Layer[],
  enabled: Record<string, boolean>,
  minArea: number
) {
  const base = await loadImage(originalSrc);
  const width = base.naturalWidth;
  const height = base.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not build the photo");
  ctx.drawImage(base, 0, 0);

  for (const layer of layers) {
    if (!layer.visible) continue;
    const islands = keptIslands(layer.islands, enabled, minArea);
    if (islands.length === 0) continue;
    const edited = await loadImage(layer.edited);
    for (const island of islands) {
      const mask = await loadImage(island.mask);
      ctx.drawImage(paintIsland(edited, mask, width, height), 0, 0);
    }
  }

  return canvas.toDataURL("image/jpeg", 0.9);
}
