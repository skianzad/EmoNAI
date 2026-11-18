export const PROFESSIONAL = "/scenes/professional/original.jpg?v=3";
export const PERSON_MASK = "/scenes/professional/person-original-clean.png?v=7";
export const COAT_MASK = "/scenes/professional/coat-mask.png?v=3";

export type LookOption = {
  id: string;
  src?: string;
  mask?: string;
  hex?: string;
  label: string;
  zh: string;
};

export const LOOK_BACKGROUNDS: LookOption[] = [
  { id: "original", src: PROFESSIONAL, label: "Original", zh: "原背景" },
  { id: "office", src: "/scenes/professional/office.jpg", label: "Office", zh: "办公室" },
  { id: "studio", src: "/scenes/professional/studio.jpg", label: "Studio", zh: "影棚" },
  { id: "lobby", src: "/scenes/professional/lobby.jpg", label: "Lobby", zh: "大堂" },
];

/** Masks are cleaned offline by scripts/bake-professional-masks.py. */
export const LOOK_COATS: LookOption[] = [
  { id: "original", src: PROFESSIONAL, mask: PERSON_MASK, hex: "#7a8aa0", label: "Original", zh: "原外套" },
  { id: "navy", src: "/scenes/professional/coat-navy.jpg", mask: "/scenes/professional/person-navy-clean.png?v=6", hex: "#12324f", label: "Navy", zh: "藏青" },
  { id: "charcoal", src: "/scenes/professional/coat-charcoal.jpg", mask: "/scenes/professional/person-charcoal-clean.png?v=6", hex: "#3a3d42", label: "Charcoal", zh: "炭灰" },
  { id: "camel", src: "/scenes/professional/coat-camel.jpg", mask: "/scenes/professional/person-camel-clean.png?v=6", hex: "#c4a574", label: "Camel", zh: "驼色" },
];

const imageCache = new Map<string, Promise<HTMLImageElement>>();
const maskCache = new Map<string, HTMLCanvasElement>();
const lookCache = new Map<string, string>();

function loadImage(src: string) {
  const hit = imageCache.get(src);
  if (hit) return hit;
  const pending = new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => {
      imageCache.delete(src);
      reject(new Error("Could not read the look"));
    };
    image.src = src;
  });
  imageCache.set(src, pending);
  return pending;
}

async function alphaMask(maskSrc: string, width: number, height: number) {
  const key = `${maskSrc}|${width}x${height}`;
  const cached = maskCache.get(key);
  if (cached) return cached;
  const mask = await loadImage(maskSrc);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;
  ctx.drawImage(mask, 0, 0, width, height);
  const image = ctx.getImageData(0, 0, width, height);
  const pixels = image.data;
  for (let i = 0; i < pixels.length; i += 4) {
    pixels[i + 3] = pixels[i];
    pixels[i] = 255;
    pixels[i + 1] = 255;
    pixels[i + 2] = 255;
  }
  ctx.putImageData(image, 0, 0);
  maskCache.set(key, canvas);
  return canvas;
}

/** Cover-draw with a slight zoom, biased toward the upper backdrop (windows, not floor). */
function coverDraw(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  width: number,
  height: number,
  focusY = 0.2,
  zoom = 1.38
) {
  const scale = Math.max(width / image.naturalWidth, height / image.naturalHeight) * zoom;
  const drawW = image.naturalWidth * scale;
  const drawH = image.naturalHeight * scale;
  const x = (width - drawW) / 2;
  const y = (height - drawH) * focusY;
  ctx.drawImage(image, x, y, drawW, drawH);
}

function averageColor(source: HTMLCanvasElement) {
  const sample = document.createElement("canvas");
  sample.width = 16;
  sample.height = 16;
  const ctx = sample.getContext("2d");
  if (!ctx) return "rgb(160,160,160)";
  ctx.drawImage(source, 0, 0, 16, 16);
  const pixels = ctx.getImageData(0, 0, 16, 16).data;
  let r = 0;
  let g = 0;
  let b = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    r += pixels[i];
    g += pixels[i + 1];
    b += pixels[i + 2];
  }
  const n = pixels.length / 4;
  return `rgb(${Math.round(r / n)},${Math.round(g / n)},${Math.round(b / n)})`;
}

/**
 * Local composite like sky replace: baked suit photo + background cover,
 * using a per-coat person mask. Coat switch is a pre-baked layer, not a tint.
 */
export async function applyProfessionalLook(photoSrc: string, backgroundSrc: string | null, maskSrc = PERSON_MASK) {
  if (!backgroundSrc) return photoSrc;
  const cacheKey = `${photoSrc}|${backgroundSrc}|${maskSrc}`;
  const cached = lookCache.get(cacheKey);
  if (cached) return cached;

  const [photo, background] = await Promise.all([loadImage(photoSrc), loadImage(backgroundSrc)]);
  const width = photo.naturalWidth;
  const height = photo.naturalHeight;
  const personMask = await alphaMask(maskSrc, width, height);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return photoSrc;

  coverDraw(ctx, background, width, height);
  const wash = averageColor(canvas);

  const sw = Math.max(64, Math.round(width / 4));
  const sh = Math.max(64, Math.round(height / 4));
  const shadow = document.createElement("canvas");
  shadow.width = sw;
  shadow.height = sh;
  const shadowCtx = shadow.getContext("2d");
  if (shadowCtx) {
    shadowCtx.filter = "blur(5px)";
    shadowCtx.drawImage(personMask, 0, 3, sw, sh);
    shadowCtx.filter = "none";
    shadowCtx.globalCompositeOperation = "source-in";
    shadowCtx.fillStyle = "#1a1814";
    shadowCtx.fillRect(0, 0, sw, sh);
    ctx.globalAlpha = 0.22;
    ctx.drawImage(shadow, 0, 0, width, height);
    ctx.globalAlpha = 1;
  }

  const person = document.createElement("canvas");
  person.width = width;
  person.height = height;
  const personCtx = person.getContext("2d");
  if (personCtx) {
    personCtx.drawImage(photo, 0, 0, width, height);
    personCtx.globalCompositeOperation = "destination-in";
    personCtx.drawImage(personMask, 0, 0);
    personCtx.globalCompositeOperation = "source-atop";
    personCtx.globalAlpha = 0.1;
    personCtx.fillStyle = wash;
    personCtx.fillRect(0, 0, width, height);
    ctx.drawImage(person, 0, 0);
  }

  const url = canvas.toDataURL("image/jpeg", 0.88);
  lookCache.set(cacheKey, url);
  return url;
}
