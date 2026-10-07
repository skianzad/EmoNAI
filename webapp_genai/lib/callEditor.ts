import { fitImage, readDataUrl } from "./shrink";

export async function callEditor(editor: string, prompt: string, source: string, mask?: string) {
  const { mime, image } = await encodeImage(source, true);
  const region = mask ? await encodeImage(editor === "gpt_image_2" ? await openAIMask(mask) : mask) : null;
  const response = await fetch("/api/edit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      editor,
      prompt,
      image,
      mime,
      mask: region?.image,
      maskMime: region?.mime,
    }),
  });
  const body = await response.json().catch(() => ({ error: "Edit failed" }));
  if (!response.ok) throw new Error(body.error || "Edit failed");
  return body.image as string;
}

async function encodeImage(source: string, fit = false) {
  if (source.startsWith("data:") && !fit) {
    const [head, image] = source.split(",");
    return { mime: head.slice(5, head.indexOf(";")), image };
  }
  const original = await fetch(source).then((response) => response.blob());
  const blob = fit ? await fitImage(original) : original;
  const dataUrl = await readDataUrl(blob);
  const [head, image] = dataUrl.split(",");
  return { mime: head.slice(5, head.indexOf(";")) || blob.type || "image/jpeg", image };
}

/** GPT edits the transparent pixels. White in our region becomes transparent. */
async function openAIMask(maskUrl: string) {
  const image = await new Promise<HTMLImageElement>((resolve, reject) => {
    const picture = new Image();
    picture.onload = () => resolve(picture);
    picture.onerror = () => reject(new Error("Could not read the region"));
    picture.src = maskUrl;
  });
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return maskUrl;
  ctx.drawImage(image, 0, 0);
  const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = pixels.data;
  for (let i = 0; i < data.length; i += 4) {
    const editable = data[i] > 128 || data[i + 1] > 128 || data[i + 2] > 128;
    if (editable) {
      data[i] = 0;
      data[i + 1] = 0;
      data[i + 2] = 0;
      data[i + 3] = 0;
    } else {
      data[i] = 255;
      data[i + 1] = 255;
      data[i + 2] = 255;
      data[i + 3] = 255;
    }
  }
  ctx.putImageData(pixels, 0, 0);
  return canvas.toDataURL("image/png");
}
