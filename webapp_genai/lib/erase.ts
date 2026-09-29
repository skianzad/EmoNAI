import type { Island } from "@/data/edits";

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not read the region"));
    image.src = src;
  });
}

/** Punch the stroke out of these regions. Selected regions are the only ones touched. */
export async function eraseIslands(
  islands: Island[],
  points: { x: number; y: number }[],
  radius: number,
  selectedIds: string[]
) {
  const targets = new Set(selectedIds.length ? selectedIds : islands.map((island) => island.id));
  const next = await Promise.all(
    islands.map(async (island) => {
      if (!targets.has(island.id) || points.length === 0) return island;
      const image = await loadImage(island.mask);
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return island;
      ctx.drawImage(image, 0, 0);
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = "#000";
      for (const point of points) {
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
      const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let area = 0;
      for (let i = 3; i < pixels.length; i += 4) {
        if (pixels[i] > 40) area += 1;
      }
      return { ...island, mask: canvas.toDataURL("image/png"), area };
    })
  );
  return next.filter((island) => island.area > 0);
}
