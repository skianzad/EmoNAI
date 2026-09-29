import { editorChoices, type EditorId } from "@/data/editors";
import { key } from "./keys";

const KEY_NAME: Record<EditorId, string> = {
  nano_banana: "GEMINI_API_KEY",
  qwen_image_3: "DASHSCOPE_API_KEY",
  seedream_5_pro: "ARK_API_KEY",
  gpt_image_2: "OPENAI_API_KEY",
};

export function editorStatus() {
  return editorChoices.map((editor) => ({
    ...editor,
    ready: Boolean(key(KEY_NAME[editor.id])),
  }));
}

export type MaskImage = { data: string; mime: string };

function guidedPrompt(prompt: string, mask?: MaskImage) {
  if (!mask) return prompt;
  return [
    "The first image is the photo. The second image is a mask of the same size: white pixels are the only region you may change, and black pixels must stay identical to the photo.",
    `Further change, only inside the white region: ${prompt}`,
  ].join("\n");
}

export async function runEditor(
  editor: EditorId,
  prompt: string,
  image: string,
  mime: string,
  mask?: MaskImage
) {
  const apiKey = key(KEY_NAME[editor]);
  const name = editorChoices.find((item) => item.id === editor)?.name ?? editor;
  if (!apiKey) throw new Error(`No key for ${name}`);
  const text = guidedPrompt(prompt, mask);

  if (editor === "nano_banana") return nanoBanana(apiKey, text, image, mime, mask);
  if (editor === "qwen_image_3") return qwen(apiKey, text, image, mime, mask);
  if (editor === "seedream_5_pro") return seedream(apiKey, text, image, mime, mask);
  return gptImage(apiKey, text, image, mime, mask);
}

async function nanoBanana(apiKey: string, prompt: string, image: string, mime: string, mask?: MaskImage) {
  const model = key("GEMINI_IMAGE_MODEL") || "gemini-3.1-flash-image";
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              { text: prompt },
              { inline_data: { mime_type: mime, data: image } },
              ...(mask ? [{ inline_data: { mime_type: mask.mime, data: mask.data } }] : []),
            ],
          },
        ],
        generationConfig: { responseModalities: ["TEXT", "IMAGE"] },
      }),
      signal: AbortSignal.timeout(120_000),
    }
  );
  const body = await readJson(response, "Nano Banana");
  const parts = body.candidates?.[0]?.content?.parts ?? [];
  for (const part of parts) {
    const inline = part.inlineData ?? part.inline_data;
    if (inline?.data) return dataUrl(inline.mimeType || inline.mime_type || "image/png", inline.data);
  }
  throw new Error("Nano Banana returned no image");
}

async function qwen(apiKey: string, prompt: string, image: string, mime: string, mask?: MaskImage) {
  const base = (key("DASHSCOPE_BASE_URL") || "https://dashscope-intl.aliyuncs.com/api/v1").replace(/\/$/, "");
  const model = key("QWEN_IMAGE_MODEL") || "qwen-image-3.0-pro";
  const payload = {
    model,
    input: {
      messages: [
        {
          role: "user",
          content: [
            { image: `data:${mime};base64,${image}` },
            ...(mask ? [{ image: `data:${mask.mime};base64,${mask.data}` }] : []),
            { text: prompt },
          ],
        },
      ],
    },
    parameters: { n: 1, watermark: false, prompt_extend: true },
  };
  let response = await fetch(`${base}/services/aigc/multimodal-generation/generation`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "X-DashScope-Async": "enable",
    },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(120_000),
  });
  if (!response.ok) {
    response = await fetch(`${base}/services/aigc/multimodal-generation/generation`, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(120_000),
    });
  }
  let body = await readJson(response, "Qwen");
  const direct = qwenImage(body);
  if (direct) return fetchDataUrl(direct);
  const taskId = body.output?.task_id || body.task_id;
  if (!taskId) throw new Error("Qwen returned no image");
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    await wait(2000);
    const status = await fetch(`${base}/tasks/${taskId}`, {
      headers: { Authorization: `Bearer ${apiKey}` },
    });
    body = await readJson(status, "Qwen");
    const state = String(body.output?.task_status || body.task_status || "").toUpperCase();
    if (state === "SUCCEEDED" || state === "SUCCESS") {
      const url = qwenImage(body);
      if (!url) throw new Error("Qwen returned no image");
      return fetchDataUrl(url);
    }
    if (["FAILED", "CANCELED", "CANCELLED", "UNKNOWN"].includes(state)) {
      throw new Error(`Qwen failed (${state})`);
    }
  }
  throw new Error("Qwen timed out");
}

async function seedream(apiKey: string, prompt: string, image: string, mime: string, mask?: MaskImage) {
  const base = (key("ARK_BASE_URL") || "https://ark.ap-southeast.bytepluses.com/api/v3").replace(/\/$/, "");
  const model = key("ARK_MODEL") || "seedream-5-0-pro-260628";
  const response = await fetch(`${base}/images/generations`, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      prompt,
      image: mask
        ? [`data:${mime};base64,${image}`, `data:${mask.mime};base64,${mask.data}`]
        : `data:${mime};base64,${image}`,
      size: "2K",
      output_format: "png",
      response_format: "url",
      watermark: false,
    }),
    signal: AbortSignal.timeout(120_000),
  });
  const body = await readJson(response, "Seedream");
  const item = body.data?.[0];
  if (item?.b64_json) return dataUrl("image/png", item.b64_json);
  if (item?.url) return fetchDataUrl(item.url);
  throw new Error("Seedream returned no image");
}

async function gptImage(apiKey: string, prompt: string, image: string, mime: string, mask?: MaskImage) {
  const model = key("OPENAI_IMAGE_MODEL") || "gpt-image-2.5-sunburst";
  const form = new FormData();
  form.set("model", model);
  form.set("prompt", prompt);
  form.set("output_format", "png");
  form.set("image", new Blob([Buffer.from(image, "base64")], { type: mime }), "source.png");
  if (mask) {
    form.set("mask", new Blob([Buffer.from(mask.data, "base64")], { type: mask.mime }), "mask.png");
  }
  const response = await fetch("https://api.openai.com/v1/images/edits", {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}` },
    body: form,
    signal: AbortSignal.timeout(120_000),
  });
  const body = await readJson(response, "GPT Image");
  const encoded = body.data?.[0]?.b64_json;
  if (!encoded) throw new Error("GPT Image returned no image");
  return dataUrl("image/png", encoded);
}

function qwenImage(body: { output?: { choices?: { message?: { content?: { image?: string }[] } }[] } }) {
  return body.output?.choices?.[0]?.message?.content?.[0]?.image;
}

async function fetchDataUrl(url: string) {
  const response = await fetch(url, { signal: AbortSignal.timeout(60_000) });
  if (!response.ok) throw new Error("Could not download the edited image");
  const mime = response.headers.get("content-type") || "image/png";
  const bytes = Buffer.from(await response.arrayBuffer());
  return dataUrl(mime.split(";")[0], bytes.toString("base64"));
}

function dataUrl(mime: string, encoded: string) {
  return `data:${mime};base64,${encoded}`;
}

async function readJson(response: Response, name: string) {
  const text = await response.text();
  if (!response.ok) throw new Error(`${name} ${response.status}: ${text.slice(0, 240)}`);
  return JSON.parse(text);
}

function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
