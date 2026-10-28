import { uid } from "@/lib/uid";
import type { SegmentObject } from "@/lib/localEdit";

export const DESERT = "/scenes/desert/original.jpg";
export const SKY_MASK = "/scenes/desert/sky-mask.png?v=2";

export type SkyPreset = {
  id: string;
  src: string;
  label: string;
  zh: string;
  group: string;
  groupZh: string;
};

export const SKY_PRESETS: SkyPreset[] = [
  { id: "original", src: "", label: "Original", zh: "原天空", group: "Original", groupZh: "原图" },
  { id: "golden-a", src: "/scenes/desert/skies/golden-a.jpg", label: "Golden hour", zh: "黄金时刻", group: "Golden hour", groupZh: "黄金时刻" },
  { id: "golden-b", src: "/scenes/desert/skies/golden-b.jpg", label: "Sunburst", zh: "霞光", group: "Golden hour", groupZh: "黄金时刻" },
  { id: "dawn-a", src: "/scenes/desert/skies/dawn-a.jpg", label: "Dawn pink", zh: "晨粉", group: "Dawn", groupZh: "黎明" },
  { id: "dawn-b", src: "/scenes/desert/skies/dawn-b.jpg", label: "Sunrise", zh: "日出", group: "Dawn", groupZh: "黎明" },
  { id: "blue", src: "/scenes/desert/skies/blue.jpg", label: "Blue clouds", zh: "蓝天白云", group: "Day", groupZh: "白天" },
  { id: "overcast", src: "/scenes/desert/skies/overcast.jpg", label: "Overcast", zh: "阴天", group: "Cloudy", groupZh: "多云" },
  { id: "storm", src: "/scenes/desert/skies/storm.jpg", label: "Storm", zh: "暴风雨", group: "Cloudy", groupZh: "多云" },
  { id: "night", src: "/scenes/desert/skies/night.jpg", label: "Night", zh: "夜空", group: "Night", groupZh: "夜晚" },
];

function loadImage(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not read the sky"));
    image.src = src;
  });
}

/** Build a sky segment from the baked mask so the rest of AI Edit can keep using objects. */
export function skyObject(maskSrc = SKY_MASK): SegmentObject {
  return {
    id: uid(),
    name: "sky",
    keyColor: "#7ec8e8",
    tint: null,
    shade: 0,
    scale: 100,
    hidden: false,
    dx: 0,
    dy: 0,
    mask: maskSrc,
    box: { x: 0, y: 0, w: 1, h: 0.55 },
  };
}

/**
 * Composite a replacement sky into the sky mask.
 * `zoom` 1 = cover the sky box; larger values make cloud detail bigger (zoom in).
 */
export async function replaceSky(photoSrc: string, maskSrc: string, skySrc: string | null, zoom = 1) {
  const photo = await loadImage(photoSrc);
  const width = photo.naturalWidth;
  const height = photo.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return photoSrc;
  ctx.drawImage(photo, 0, 0, width, height);
  if (!skySrc) return canvas.toDataURL("image/jpeg", 0.92);

  const [mask, sky] = await Promise.all([loadImage(maskSrc), loadImage(skySrc)]);
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskCtx = maskCanvas.getContext("2d");
  if (!maskCtx) return canvas.toDataURL("image/jpeg", 0.92);
  maskCtx.drawImage(mask, 0, 0, width, height);
  const hard = maskCtx.getImageData(0, 0, width, height);
  const hardPx = hard.data;
  for (let i = 0; i < hardPx.length; i += 4) {
    const luma = (hardPx[i] + hardPx[i + 1] + hardPx[i + 2]) / 3;
    hardPx[i + 3] = Math.round((luma * hardPx[i + 3]) / 255);
    hardPx[i] = 255;
    hardPx[i + 1] = 255;
    hardPx[i + 2] = 255;
  }
  maskCtx.putImageData(hard, 0, 0);
  // Soften only inside the sky so the ridge does not glow onto rock.
  maskCtx.filter = "blur(1.4px)";
  maskCtx.drawImage(maskCanvas, 0, 0);
  maskCtx.filter = "none";
  const maskPixels = maskCtx.getImageData(0, 0, width, height);
  const pixels = maskPixels.data;
  for (let i = 3; i < pixels.length; i += 4) {
    if (hardPx[i] < 12) pixels[i] = 0;
  }
  maskCtx.putImageData(maskPixels, 0, 0);
  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;
  for (let y = 0; y < height; y += 3) {
    for (let x = 0; x < width; x += 3) {
      if (pixels[(y * width + x) * 4 + 3] < 24) continue;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }
  if (maxX <= minX || maxY <= minY) {
    minX = 0;
    minY = 0;
    maxX = width;
    maxY = Math.round(height * 0.55);
  }
  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  const scale = Math.max(boxW / sky.naturalWidth, boxH / sky.naturalHeight) * Math.max(1, zoom);
  const drawW = sky.naturalWidth * scale;
  const drawH = sky.naturalHeight * scale;
  const cx = minX + boxW / 2;
  const cy = minY + boxH / 2;

  const layer = document.createElement("canvas");
  layer.width = width;
  layer.height = height;
  const layerCtx = layer.getContext("2d");
  if (!layerCtx) return canvas.toDataURL("image/jpeg", 0.92);
  layerCtx.drawImage(sky, cx - drawW / 2, cy - drawH / 2, drawW, drawH);
  layerCtx.globalCompositeOperation = "destination-in";
  layerCtx.drawImage(maskCanvas, 0, 0);
  ctx.drawImage(layer, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.92);
}
