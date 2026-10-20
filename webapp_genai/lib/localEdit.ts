import type { Island } from "@/data/edits";
import { objectsFromPaint, type NamedColor } from "@/lib/colorAdjust";
import { callEditor } from "@/lib/callEditor";
import { COLOR_MAP_PALETTE, guidePrompt } from "@/lib/colorMapPrompt";
import { fitImage, readDataUrl } from "@/lib/shrink";
import { splitIslands } from "@/lib/splitIslands";
import { uid } from "@/lib/uid";

const BACKDROP = /background|backdrop|wall|floor|ceiling|sky|ground|carpet|rug/i;

export function isBackdrop(name: string) {
  return BACKDROP.test(name);
}

export type Box = { x: number; y: number; w: number; h: number };

/** One cell from a 3×4 AI color pack (sofa / chair). */
export type ColorVariant = {
  hex: string;
  shade: number;
  layer: string;
};

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
  /** Layer before an AI color pack (restore via Original). */
  baseLayer?: string;
  /** Pre-baked 3 colors × 4 shades from one contact-sheet generation. */
  variants?: ColorVariant[];
  /** Blur radius in pixels, used for a background region. */
  blur?: number;
  /** Pre-baked add-on (e.g. curtain). Not in the original photo; hide = skip draw. */
  suggested?: boolean;
  /** AI-replaced object: clear old footprint with empty bg, then draw `layer` on top. */
  replaced?: boolean;
  /**
   * Layer is already in the working photo (approved AI add). Skip redraw until the
   * user moves / recolors / hides it — otherwise empty-bg punch wipes the object.
   */
  baked?: boolean;
  /** Louvre crowd: subject = couple (always kept), person = background tourist. */
  crowdRole?: "subject" | "person";
};

export function isCrowdPerson(object: SegmentObject) {
  if (object.crowdRole === "person") return true;
  if (object.crowdRole === "subject") return false;
  return /person|people|tourist|visitor|crowd|passer/i.test(object.name);
}

export function isCrowdSubject(object: SegmentObject) {
  if (object.crowdRole === "subject") return true;
  if (object.crowdRole === "person") return false;
  return /^(man|woman|couple|subject)\b/i.test(object.name);
}

/** Shade steps used in the sofa/chair color pack (4 shades). */
export const VARIANT_SHADES = [-2, -1, 0, 1] as const;

export function isFurnitureRecolor(name: string) {
  return /sofa|couch|chair|armchair|loveseat|ottoman/i.test(name);
}

/** Pick 3 hexes for the pack: primary first, then fill from the palette. */
export function pickVariantColors(primary: string, palette: { hex: string }[]) {
  const colors: string[] = [];
  const add = (hex: string) => {
    const next = hex.toLowerCase();
    if (!colors.some((item) => item.toLowerCase() === next)) colors.push(hex);
  };
  add(primary);
  for (const swatch of palette) {
    if (colors.length >= 3) break;
    add(swatch.hex);
  }
  while (colors.length < 3) add(palette[colors.length % Math.max(1, palette.length)]?.hex || primary);
  return colors.slice(0, 3);
}

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

export type GuideBox = { name: string; left: number; top: number; width: number; height: number };

/** Local segment from a guide box — used when color-map is unavailable. */
export async function segmentFromGuide(src: string, guide: GuideBox): Promise<SegmentObject> {
  const original = await loadImage(src);
  const width = original.naturalWidth;
  const height = original.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not create a segment");
  const x = Math.round(guide.left * width);
  const y = Math.round(guide.top * height);
  const w = Math.max(8, Math.round(guide.width * width));
  const h = Math.max(8, Math.round(guide.height * height));
  ctx.fillStyle = "#fff";
  // Soft rounded region so edges are not a hard rectangle.
  const radius = Math.min(24, Math.round(Math.min(w, h) * 0.2));
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
  ctx.fill();
  const soft = softenMask(canvas, width, height, Math.max(2, Math.round(Math.min(w, h) * 0.04)));
  return {
    id: uid(),
    name: guide.name,
    keyColor: "#7CFC00",
    tint: null,
    shade: 0,
    scale: 100,
    hidden: false,
    dx: 0,
    dy: 0,
    mask: soft.toDataURL("image/png"),
    box: {
      x: guide.left,
      y: guide.top,
      w: guide.width,
      h: guide.height,
    },
  };
}

export async function segmentPhoto(
  editorId: string,
  src: string,
  kind: "objects" | "face" | "room" | "crowd" = "objects",
  guide?: GuideBox[]
) {
  const blob = await fitImage(await imageBlob(src));
  const dataUrl = await readDataUrl(blob);
  const [head, image] = dataUrl.split(",");
  let response: Response;
  try {
    response = await fetch("/api/color-map", {
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
  } catch {
    if (guide?.length) return Promise.all(guide.map((item) => segmentFromGuide(src, item)));
    throw new Error("Could not reach the editor. Check your network and try again.");
  }
  const body = await response.json().catch(() => ({ error: "Edit failed" }));
  if (!response.ok) {
    if (guide?.length) return Promise.all(guide.map((item) => segmentFromGuide(src, item)));
    throw new Error(body.error || "Edit failed");
  }
  const named = Array.isArray(body.items) ? (body.items as NamedColor[]) : [];
  const paintedSrc = String(body.image ?? "");
  const byColor = await segmentsFromColors(
    dataUrl,
    paintedSrc,
    named,
    kind === "crowd" ? 220 : kind === "objects" ? 400 : 80,
    // Face / crowd: keep every blob. Room furniture: largest blob only.
    kind === "face" || kind === "crowd",
    kind === "room" ? 110 : 58,
    kind === "crowd" ? 16 : 8
  );
  if (byColor.length > 0) {
    if (kind === "crowd") {
      return byColor.map((object) => ({
        ...object,
        // Subjects stay; background people are toggled by the crowd slider.
        crowdRole: /^(man|woman|couple|subject)/i.test(object.name) ? ("subject" as const) : ("person" as const),
      }));
    }
    return byColor;
  }
  if (guide?.length) return Promise.all(guide.map((item) => segmentFromGuide(src, item)));
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

function softenMask(mask: HTMLCanvasElement | HTMLImageElement, width: number, height: number, radius: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;
  ctx.filter = `blur(${Math.max(0.5, radius)}px)`;
  ctx.drawImage(mask, 0, 0, width, height);
  ctx.filter = "none";
  const image = ctx.getImageData(0, 0, width, height);
  for (let i = 3; i < image.data.length; i += 4) {
    const alpha = image.data[i];
    // Keep a light feather only — do not inflate alpha (that caused ghost halos).
    image.data[i] = alpha < 12 ? 0 : alpha;
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

function rimRadius(width: number, height: number) {
  return Math.max(2, Math.round(Math.min(width, height) * 0.003));
}

/** Binary mask: alpha >= 128 → 255, else 0. Optional tiny dilate via blur+threshold. */
async function hardMask(maskSrc: string, width: number, height: number, dilate = 0) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;
  const source = await loadImage(maskSrc);
  if (dilate > 0) {
    ctx.filter = `blur(${dilate}px)`;
    ctx.drawImage(source, 0, 0, width, height);
    ctx.filter = "none";
  } else {
    ctx.drawImage(source, 0, 0, width, height);
  }
  const image = ctx.getImageData(0, 0, width, height);
  for (let i = 3; i < image.data.length; i += 4) {
    image.data[i] = image.data[i] >= 128 ? 255 : 0;
    if (image.data[i] === 0) {
      image.data[i - 3] = 0;
      image.data[i - 2] = 0;
      image.data[i - 1] = 0;
    } else {
      image.data[i - 3] = 255;
      image.data[i - 2] = 255;
      image.data[i - 1] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
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
  const soft = softenMask(mask, width, height, Math.max(8, Math.round(radius * 0.12)));
  layerCtx.filter = `blur(${radius}px)`;
  layerCtx.drawImage(ctx.canvas, 0, 0);
  layerCtx.filter = "none";
  layerCtx.globalCompositeOperation = "destination-in";
  layerCtx.drawImage(soft, 0, 0);
  ctx.drawImage(layer, 0, 0);
}

/**
 * Own mask owns contested pixels. Neighbors are punched out hard so two layers
 * never share the same pixel (soft dual-ownership caused the white ghosts).
 * A small outer feather is applied only after exclusivity.
 */
async function exclusiveMask(maskSrc: string, others: string[], width: number, height: number, radius: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;
  // Slight dilate so each object keeps a thin rim before neighbors reclaim overlap.
  ctx.drawImage(await hardMask(maskSrc, width, height, Math.max(1, Math.round(radius))), 0, 0);
  for (const other of others) {
    ctx.globalCompositeOperation = "destination-out";
    // Punch neighbors hard (no soft alpha) so contested pixels go to one owner only.
    ctx.drawImage(await hardMask(other, width, height, 0), 0, 0);
  }
  ctx.globalCompositeOperation = "source-over";
  if (radius <= 0) return canvas;
  // Light anti-alias on the final silhouette only.
  return softenMask(canvas, width, height, Math.min(2, radius));
}

/**
 * Cut each foreground object from the original photo with its original mask.
 * No neighbor punch-outs (those ate chair arms/legs). Removal uses the saved
 * empty background underneath when a layer is hidden.
 */
export async function liftLayers(originalSrc: string, objects: SegmentObject[]) {
  const original = await loadImage(originalSrc);
  const width = original.naturalWidth;
  const height = original.naturalHeight;
  const rim = rimRadius(width, height);
  const next: SegmentObject[] = [];
  for (const object of objects) {
    if (isBackdrop(object.name)) {
      next.push({ ...object, layer: undefined });
      continue;
    }
    const cut = await cutout(original, object.mask, width, height, [], rim);
    if (!cut) {
      next.push({ ...object, layer: undefined });
      continue;
    }
    next.push({ ...object, layer: cut.canvas.toDataURL("image/png") });
  }
  return next;
}

/** White = remove/edit, black = keep. Union of every foreground segment, slightly grown. */
async function unionMask(originalSrc: string, objects: SegmentObject[]) {
  const original = await loadImage(originalSrc);
  const width = original.naturalWidth;
  const height = original.naturalHeight;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);
  // Grow enough that chair legs, lamp stems, and soft edges are inside the edit region.
  const grow = Math.max(8, Math.round(Math.min(width, height) * 0.02));
  for (const object of objects) {
    const stamp = await hardMask(object.mask, width, height, grow);
    ctx.drawImage(stamp, 0, 0);
  }
  return canvas.toDataURL("image/png");
}

async function localClearBackground(originalSrc: string, objects: SegmentObject[]) {
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
    if (cut) eraseObject(ctx, cut.canvas, width, height);
  }
  return canvas.toDataURL("image/jpeg", 0.9);
}

/** Grow a mask so the editor can fill missing edges of an object. */
async function growMask(maskSrc: string, width: number, height: number, radius: number) {
  const soft = softenMask(await loadImage(maskSrc), width, height, radius);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return maskSrc;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(soft, 0, 0);
  return canvas.toDataURL("image/png");
}

/**
 * Ask the editor for a complete reconstruction of one object, then cut an
 * exclusive layer (neighbors punched hard so sprites never overlap).
 */
async function completeObjectLayer(
  editorId: string,
  originalSrc: string,
  object: SegmentObject,
  others: string[] = []
) {
  const original = await loadImage(originalSrc);
  const width = original.naturalWidth;
  const height = original.naturalHeight;
  const grow = Math.max(8, Math.round(Math.min(width, height) * 0.018));
  const region = await growMask(object.mask, width, height, grow);
  const prompt = [
    `Focus only on the ${object.name}.`,
    `Reconstruct the complete ${object.name} with every part intact — legs, frame, cushions, arms, and edges. Fill any holes or missing pieces.`,
    `Keep its exact place, size, pose, lighting, and material in the photo. Do not change any other object.`,
    `Output one finished photograph.`,
  ].join(" ");
  const image = await callEditor(editorId, prompt, originalSrc, region);
  const fitted = await fitEdited(originalSrc, image);
  const edited = await loadImage(fitted);
  const cut = await cutout(edited, object.mask, width, height, others, rimRadius(width, height));
  if (!cut) return object.layer;
  return cut.canvas.toDataURL("image/png");
}

export type PrepareProgress = (message: string) => void;

/**
 * Ask Nano Banana to erase every identified segment (white in the union mask)
 * and inpaint a clean empty room. One edit call — no per-object passes.
 */
/**
 * Bake 3 colors × N shades for sofa/chair switching.
 * Uses 3 soft single-color edits (contact sheets often hit IMAGE_RECITATION),
 * then derives shade steps locally from each color layer.
 */
export async function generateColorVariants(
  editorId: string,
  photoSrc: string,
  object: SegmentObject,
  colors: string[],
  shades: number[],
  toShade: (hex: string, shade: number) => string
): Promise<ColorVariant[]> {
  const picks = colors.slice(0, 3);
  const shadeSteps = shades.slice(0, 4);
  if (!picks.length || !shadeSteps.length) return [];

  const photo = await loadImage(photoSrc);
  const width = photo.naturalWidth;
  const height = photo.naturalHeight;
  const variants: ColorVariant[] = [];

  for (const hex of picks) {
    const target = toShade(hex, 0).toUpperCase();
    const prompt = [
      `Edit this living-room photo: recolor only the ${object.name} to ${target}.`,
      "Keep the room layout, camera, lighting, and other furniture looking natural.",
      "Return the edited photograph.",
    ].join(" ");
    const edited = await callEditor(editorId, prompt, photoSrc);
    const baseLayer = await layerFromEdited(edited, object.mask, width, height);
    if (!baseLayer) continue;
    for (const shade of shadeSteps) {
      const layer = shade === 0 ? baseLayer : await shadeLayer(baseLayer, shade);
      variants.push({ hex, shade, layer });
    }
  }
  return variants;
}

/** Combine two alpha masks so fill clears both the old object and the new edit. */
export async function unionMasks(a: string, b: string) {
  const [first, second] = await Promise.all([loadImage(a), loadImage(b)]);
  const width = Math.max(first.naturalWidth, second.naturalWidth);
  const height = Math.max(first.naturalHeight, second.naturalHeight);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return a;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(first, 0, 0, width, height);
  ctx.globalCompositeOperation = "source-over";
  ctx.drawImage(second, 0, 0, width, height);
  return canvas.toDataURL("image/png");
}

async function layerFromEdited(editedSrc: string, maskSrc: string, width: number, height: number) {
  const edited = await loadImage(editedSrc);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  ctx.drawImage(edited, 0, 0, width, height);
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(await loadImage(maskSrc), 0, 0, width, height);
  return canvas.toDataURL("image/png");
}

/** Darken / lighten opaque pixels of a masked layer for shade swatches. */
async function shadeLayer(layerSrc: string, shade: number) {
  const image = await loadImage(layerSrc);
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return layerSrc;
  ctx.drawImage(image, 0, 0);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = pixels.data;
  const amount = shade > 0 ? shade * 0.14 : shade * 0.16;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 8) continue;
    for (let c = 0; c < 3; c += 1) {
      const channel = data[i + c];
      data[i + c] =
        amount > 0
          ? Math.min(255, Math.round(channel + (255 - channel) * amount))
          : Math.max(0, Math.round(channel * (1 + amount)));
    }
  }
  ctx.putImageData(pixels, 0, 0);
  return canvas.toDataURL("image/png");
}

export async function generateEmptyBackground(editorId: string, originalSrc: string, objects: SegmentObject[]) {
  const foreground = objects.filter((object) => !isBackdrop(object.name));
  if (foreground.length === 0) return originalSrc;
  const mask = await unionMask(originalSrc, foreground);
  if (!mask) return originalSrc;
  const named = foreground.map((object) => object.name).filter((name) => name && !/^object$/i.test(name));
  const prompt = [
    `Remove ALL ${foreground.length} segmented objects marked in white in the mask — every one of them.`,
    "That includes both leather armchairs, the cream sofa, the round coffee table, the arc floor lamp with its shade, plants, and any other furniture covered by white.",
    named.length ? `Segment names: ${named.join(", ")}.` : "",
    "Erase them completely so none remain — no chairs, no lamp, no sofa, no table, no plant ghosts or shadows of them.",
    "Fill every white region with the empty room behind (floor, rug, walls, windows, built-in shelves) as if those objects were never there.",
    "Do not add new furniture. Keep the unmasked room areas looking consistent. Return the edited photograph.",
  ]
    .filter(Boolean)
    .join(" ");
  const image = await callEditor(editorId, prompt, originalSrc, mask);
  return fitEdited(originalSrc, image);
}

/** AI-complete each object into a full layer sprite. */
export async function completeObjectLayers(
  editorId: string,
  originalSrc: string,
  objects: SegmentObject[],
  onProgress?: PrepareProgress
) {
  const next: SegmentObject[] = [];
  const work = objects.filter((object) => !isBackdrop(object.name));
  for (let index = 0; index < objects.length; index += 1) {
    const object = objects[index];
    if (isBackdrop(object.name)) {
      next.push(object);
      continue;
    }
    const step = work.findIndex((item) => item.id === object.id) + 1;
    onProgress?.(`Completing ${object.name} (${step}/${work.length})…`);
    const others = objects
      .filter((item) => item.id !== object.id && !isBackdrop(item.name))
      .map((item) => item.mask);
    try {
      const layer = await completeObjectLayer(editorId, originalSrc, object, others);
      next.push({ ...object, layer: layer || object.layer });
    } catch {
      next.push(object);
    }
  }
  return next;
}

/**
 * First run: generate a clean empty background, then AI-complete each object
 * into a full layer sprite. Later runs should reuse the saved background/layers.
 */
export async function prepareScene(
  editorId: string,
  originalSrc: string,
  objects: SegmentObject[],
  allowAi = true,
  onProgress?: PrepareProgress
) {
  onProgress?.("Cutting object layers…");
  const lifted = await liftLayers(originalSrc, objects);
  const foreground = lifted.filter((object) => !isBackdrop(object.name));
  if (foreground.length === 0) return { objects: lifted, background: originalSrc };

  if (!allowAi) {
    const withLayers = lifted.filter((object) => object.layer && !isBackdrop(object.name));
    return { objects: lifted, background: await localClearBackground(originalSrc, withLayers) };
  }

  onProgress?.("Generating empty background…");
  const background = await generateEmptyBackground(editorId, originalSrc, lifted);
  return { objects: lifted, background };
}

/** Grow a binary mask a few pixels so removal covers fringe / residual edges. */
function expandMask(mask: HTMLCanvasElement, width: number, height: number, px: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return mask;
  const grow = Math.max(1, Math.round(px));
  ctx.filter = `blur(${grow}px)`;
  ctx.drawImage(mask, 0, 0, width, height);
  ctx.filter = "none";
  const image = ctx.getImageData(0, 0, width, height);
  for (let i = 3; i < image.data.length; i += 4) {
    // Low threshold after blur → dilated hard edge.
    const on = image.data[i] >= 24;
    image.data[i] = on ? 255 : 0;
    image.data[i - 3] = on ? 255 : 0;
    image.data[i - 2] = on ? 255 : 0;
    image.data[i - 1] = on ? 255 : 0;
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

/** Hard-paste empty background pixels inside a slightly expanded mask (no soft edge). */
function fillVacated(
  ctx: CanvasRenderingContext2D,
  mask: HTMLCanvasElement,
  fill: HTMLImageElement | null,
  width: number,
  height: number
) {
  if (!fill) {
    eraseObject(ctx, mask, width, height);
    return;
  }
  // ~1.2% of the short side — large enough to clear residual hair/edges, not huge.
  const grow = Math.max(10, Math.round(Math.min(width, height) * 0.012));
  const punched = expandMask(mask, width, height, grow);
  const layer = document.createElement("canvas");
  layer.width = width;
  layer.height = height;
  const layerCtx = layer.getContext("2d");
  if (!layerCtx) return;
  layerCtx.drawImage(fill, 0, 0, width, height);
  layerCtx.globalCompositeOperation = "destination-in";
  layerCtx.drawImage(punched, 0, 0);
  ctx.drawImage(layer, 0, 0);
}

/**
 * Start from the working photo, apply segment edits on top.
 * Empty-bg punch (`fillSrc`) runs only when an object is removed (hidden) —
 * not for adds, recolors, or moves.
 */
export async function renderLocal(
  originalSrc: string,
  objects: SegmentObject[],
  warmth: number,
  fillSrc?: string | null
) {
  const original = await loadImage(originalSrc);
  const fill = fillSrc ? await loadImage(fillSrc) : null;
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
    // Suggested add-ons (curtain, etc.) are not in the original — never punch holes for them.
    const additive = Boolean(object.suggested);

    // Empty-bg punch ONLY on remove, and only while a fill image is supplied.
    // Callers pass no fill after an AI photo is the working base (avoids rug blotches).
    if (object.hidden) {
      if (!additive && fill) fillVacated(ctx, cut.canvas, fill, width, height);
      continue;
    }

    if (object.layer) {
      const moved = scale !== 1 || object.dx !== 0 || object.dy !== 0;
      const tinting = Boolean(object.tint && (object.cover ?? TINT) > 0.01);
      // Approved AI add: already in the working photo — leave those pixels alone.
      if (object.baked && !moved && !tinting) {
        continue;
      }
      const sprite = await loadImage(object.layer);
      ctx.save();
      ctx.translate(cut.cx + object.dx * width, cut.cy + object.dy * height);
      ctx.scale(scale, scale);
      ctx.translate(-cut.cx, -cut.cy);
      if (tinting) {
        // Recolor only the sprite's opaque pixels so added items (curtain, etc.) pick up swatches.
        const tinted = document.createElement("canvas");
        tinted.width = width;
        tinted.height = height;
        const tintCtx = tinted.getContext("2d");
        if (tintCtx) {
          tintCtx.drawImage(sprite, 0, 0, width, height);
          tintCtx.globalCompositeOperation = "source-atop";
          tintCtx.globalAlpha = object.cover ?? TINT;
          tintCtx.fillStyle = shadeHex(object.tint!, object.shade || 0);
          tintCtx.fillRect(0, 0, width, height);
          ctx.drawImage(tinted, 0, 0);
        } else {
          ctx.drawImage(sprite, 0, 0, width, height);
        }
      } else {
        ctx.drawImage(sprite, 0, 0, width, height);
      }
      ctx.restore();
      continue;
    }

    const moved = scale !== 1 || object.dx !== 0 || object.dy !== 0;
    ctx.save();
    ctx.translate(cut.cx + object.dx * width, cut.cy + object.dy * height);
    ctx.scale(scale, scale);
    ctx.translate(-cut.cx, -cut.cy);
    if (moved) ctx.drawImage(cut.canvas, 0, 0);
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

async function segmentsFromColors(
  originalSrc: string,
  paintedSrc: string,
  named: NamedColor[],
  minArea = 400,
  keepAll = false,
  match = 58,
  limit = 8
) {
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
    .slice(0, Math.max(1, limit))
    .map(({ area: _area, ...object }) => object);
}

/** Empty Louvre courtyard behind the couple — only background people are removed. */
export async function generateCrowdBackground(editorId: string, originalSrc: string, people: SegmentObject[]) {
  const crowd = people.filter(isCrowdPerson);
  if (crowd.length === 0) return originalSrc;
  const mask = await unionMask(originalSrc, crowd);
  if (!mask) return originalSrc;
  const prompt = [
    "Remove every background tourist marked in white in the mask, including anyone at the edges of the frame.",
    "Also remove their shadows on the courtyard paving.",
    "Keep the man and woman in the foreground selfie completely unchanged — faces, clothes, pose, bags, and their own shadows.",
    "Fill vacated regions with continuous Louvre courtyard paving and architecture as if those people were never there.",
    "Keep the Louvre pyramid, palace, sky, framing, lighting, and camera identical.",
    "Do not add new people, silhouettes, or wire outlines. Return only the couple on an empty courtyard.",
  ].join(" ");
  const image = await callEditor(editorId, prompt, originalSrc, mask);
  return fitEdited(originalSrc, image);
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

/**
 * Keep only the largest connected blob in a mask and recompute the box.
 * Fixes room segments that kept color speckles across the whole photo
 * (e.g. right-chair box spanning most of the frame).
 */
export async function tightenSegment(object: SegmentObject): Promise<SegmentObject> {
  // Pre-baked add-ons keep their colored layer + mask as saved.
  if (object.suggested || object.layer) return object;
  const mask = await loadImage(object.mask);
  const width = mask.naturalWidth;
  const height = mask.naturalHeight;
  if (!width || !height) return object;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return object;
  ctx.drawImage(mask, 0, 0);
  const pixels = ctx.getImageData(0, 0, width, height).data;
  const owner = new Int16Array(width * height);
  for (let i = 0; i < owner.length; i++) owner[i] = pixels[i * 4 + 3] >= 128 ? 0 : -1;
  const blob = largestBlob(owner, 0, width, height);
  if (blob.area < 80) return object;
  const fill = blob.area / Math.max(1, object.box.w * object.box.h * width * height);
  // Already tight enough.
  if (fill > 0.2) return { ...object, box: boxOf(blob.pixels, width, height) };
  return {
    ...object,
    mask: maskUrl(blob.pixels, width, height),
    box: boxOf(blob.pixels, width, height),
  };
}

export async function tightenSegments(objects: SegmentObject[]) {
  return Promise.all(objects.map((object) => (isBackdrop(object.name) ? object : tightenSegment(object))));
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

async function maskCenter(maskSrc: string, width: number, height: number) {
  const mask = await loadImage(maskSrc);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
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
  return { cx: sx / count, cy: sy / count };
}

async function cutout(
  original: HTMLImageElement,
  maskSrc: string,
  width: number,
  height: number,
  others: string[] = [],
  radius = rimRadius(width, height)
) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(original, 0, 0, width, height);
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(await exclusiveMask(maskSrc, others, width, height, radius), 0, 0);
  const center = await maskCenter(maskSrc, width, height);
  if (!center) return null;
  return { canvas, cx: center.cx, cy: center.cy };
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
