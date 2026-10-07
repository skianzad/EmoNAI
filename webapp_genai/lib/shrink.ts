export const MAX_IMAGE_BYTES = 500 * 1024;

const QUALITIES = [0.9, 0.8, 0.7, 0.6];

function encode(canvas: HTMLCanvasElement, quality: number) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Could not compress the photo"))),
      "image/jpeg",
      quality
    );
  });
}

/** Re-encodes as JPEG, lowering quality then size, until it fits under the limit. */
export async function fitImage(source: Blob, limit = MAX_IMAGE_BYTES): Promise<Blob> {
  if (source.size <= limit) return source;
  const bitmap = await createImageBitmap(source);
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not compress the photo");

  let scale = Math.min(1, 4096 / Math.max(bitmap.width, bitmap.height));
  try {
    while (scale > 0.05) {
      canvas.width = Math.max(1, Math.round(bitmap.width * scale));
      canvas.height = Math.max(1, Math.round(bitmap.height * scale));
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      for (const quality of QUALITIES) {
        const blob = await encode(canvas, quality);
        if (blob.size <= limit) return blob;
      }
      scale *= 0.8;
    }
  } finally {
    bitmap.close();
  }
  throw new Error("Could not fit the photo under 500 KB");
}

export function readDataUrl(blob: Blob) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Could not read the photo"));
    reader.readAsDataURL(blob);
  });
}
