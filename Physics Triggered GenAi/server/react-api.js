import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

function readEnv(file) {
  const out = {};
  if (!fs.existsSync(file)) return out;
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue;
    const index = trimmed.indexOf("=");
    const name = trimmed.slice(0, index).trim();
    const value = trimmed.slice(index + 1).trim().replace(/^["']|["']$/g, "");
    if (name && value) out[name] = value;
  }
  return out;
}

function env() {
  const file = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../webapp_genai/genai_image_editing/.env");
  return readEnv(file);
}

function promptFor(x, y) {
  const left = Math.round(x * 100);
  const top = Math.round(y * 100);
  return [
    "Edit this photograph.",
    `A soccer ball is about to strike at ${left}% from the left and ${top}% from the top. That point is the impact and the activation.`,
    "The person closest to that exact point reacts in a funny, exaggerated, photoreal way: head tipped into the hit, eyes wide or squeezed shut, mouth open, body staying planted.",
    "Keep the same people, clothes, background, and framing.",
    "Do not draw a ball, marker, circle, or text.",
  ].join(" ");
}

async function gemini(key, model, image, mime, text) {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              { text },
              { inline_data: { mime_type: mime, data: image } },
            ],
          },
        ],
        generationConfig: { responseModalities: ["TEXT", "IMAGE"], temperature: 0.8 },
      }),
      signal: AbortSignal.timeout(120_000),
    }
  );
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = body.error?.message || `Gemini ${response.status}`;
    throw new Error(message.slice(0, 240));
  }
  const parts = body.candidates?.[0]?.content?.parts ?? [];
  for (const part of parts) {
    const inline = part.inlineData ?? part.inline_data;
    if (inline?.data) {
      const type = inline.mimeType || inline.mime_type || "image/png";
      return `data:${type};base64,${inline.data}`;
    }
  }
  const reason = body.candidates?.[0]?.finishReason || body.promptFeedback?.blockReason || "no image";
  throw new Error(`Gemini returned ${reason}`);
}

async function openai(key, model, image, mime, text) {
  const form = new FormData();
  form.set("model", model);
  form.set("prompt", text);
  form.set("output_format", "png");
  form.set("image", new Blob([Buffer.from(image, "base64")], { type: mime }), "photo.jpg");
  const response = await fetch("https://api.openai.com/v1/images/edits", {
    method: "POST",
    headers: { Authorization: `Bearer ${key}` },
    body: form,
    signal: AbortSignal.timeout(120_000),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error((body.error?.message || `OpenAI ${response.status}`).slice(0, 240));
  const encoded = body.data?.[0]?.b64_json;
  if (!encoded) throw new Error("OpenAI returned no image");
  return `data:image/png;base64,${encoded}`;
}

async function createReaction(body) {
  const keys = env();
  const text = promptFor(Number(body.x), Number(body.y));
  const image = String(body.image || "");
  const mime = String(body.mime || "image/jpeg");
  if (!image || !Number.isFinite(Number(body.x)) || !Number.isFinite(Number(body.y))) {
    throw new Error("Choose a photo and an impact point.");
  }
  if (keys.GEMINI_API_KEY) {
    try {
      return await gemini(
        keys.GEMINI_API_KEY,
        keys.GEMINI_IMAGE_MODEL || "gemini-3.1-flash-image",
        image,
        mime,
        text
      );
    } catch (error) {
      if (!keys.OPENAI_API_KEY) throw error;
    }
  }
  if (keys.OPENAI_API_KEY) {
    return openai(keys.OPENAI_API_KEY, keys.OPENAI_IMAGE_MODEL || "gpt-image-2.5-sunburst", image, mime, text);
  }
  throw new Error("No image API key is set.");
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > 8_000_000) {
        reject(new Error("Photo is too large."));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

export function reactApi() {
  return {
    name: "react-api",
    configureServer(server) {
      server.middlewares.use("/api/react", async (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end(JSON.stringify({ error: "POST only" }));
          return;
        }
        try {
          const raw = await readBody(req);
          const image = await createReaction(JSON.parse(raw || "{}"));
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ image }));
        } catch (error) {
          res.statusCode = 502;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ error: error instanceof Error ? error.message : "Reaction failed" }));
        }
      });
    },
  };
}
