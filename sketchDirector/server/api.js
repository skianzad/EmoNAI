import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const scenesDir = path.join(root, "public", "scenes");
const indexFile = path.join(scenesDir, "index.json");

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
  const file = path.resolve(root, "../webapp_genai/genai_image_editing/.env");
  return { ...readEnv(file), ...process.env };
}

function keys() {
  const cfg = env();
  return {
    gemini: cfg.GEMINI_API_KEY || "",
    imageModel: cfg.GEMINI_IMAGE_MODEL || "gemini-3.1-flash-image",
    textModel: cfg.GEMINI_TEXT_MODEL || "gemini-3.8-flash",
  };
}

function slug(concept) {
  return (
    String(concept || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 48) || "scene"
  );
}

function readIndex() {
  try {
    return JSON.parse(fs.readFileSync(indexFile, "utf8"));
  } catch {
    return [];
  }
}

function writeIndex(items) {
  fs.mkdirSync(scenesDir, { recursive: true });
  fs.writeFileSync(indexFile, JSON.stringify(items, null, 2));
}

function deleteScene(id) {
  const safe = String(id || "").replace(/[^a-z0-9-]/g, "");
  if (!safe) throw Object.assign(new Error("Missing scene."), { status: 400 });
  const dir = path.join(scenesDir, safe);
  if (!dir.startsWith(scenesDir + path.sep)) throw Object.assign(new Error("Invalid scene."), { status: 400 });
  fs.rmSync(dir, { recursive: true, force: true });
  writeIndex(readIndex().filter((item) => item.id !== safe));
  return { id: safe };
}

function urls(id) {
  return {
    sketch: `/scenes/${id}/sketch.svg`,
    photo: `/scenes/${id}/photo.png`,
    source: `/scenes/${id}/source.jpg`,
  };
}

function sceneRecord(id, concept, extra = {}) {
  const dir = path.join(scenesDir, id);
  const links = urls(id);
  return {
    id,
    concept,
    sketch: fs.existsSync(path.join(dir, "sketch.svg")) ? links.sketch : null,
    photo: fs.existsSync(path.join(dir, "photo.png")) ? links.photo : null,
    source: fs.existsSync(path.join(dir, "source.jpg")) ? links.source : null,
    ...extra,
  };
}

function loadScene(concept) {
  const id = slug(concept);
  const dir = path.join(scenesDir, id);
  const sketchPath = path.join(dir, "sketch.svg");
  const photoPath = path.join(dir, "photo.png");
  const sourcePath = path.join(dir, "source.jpg");
  return {
    id,
    concept,
    svg: fs.existsSync(sketchPath)
      ? fs.readFileSync(sketchPath, "utf8")
      : fs.existsSync(path.join(dir, "overlay.svg"))
        ? fs.readFileSync(path.join(dir, "overlay.svg"), "utf8")
        : "",
    photo: fs.existsSync(photoPath) ? urls(id).photo : null,
    source: fs.existsSync(sourcePath) ? urls(id).source : null,
  };
}

function readPublicImage(url) {
  const clean = String(url || "").split("?")[0];
  if (!clean.startsWith("/scenes/")) return null;
  const file = path.normalize(path.join(root, "public", clean.replace(/^\//, "")));
  const allowed = path.join(root, "public", "scenes");
  if (!file.startsWith(allowed) || !fs.existsSync(file)) return null;
  return {
    mime: file.endsWith(".png") ? "image/png" : "image/jpeg",
    data: fs.readFileSync(file).toString("base64"),
  };
}

function writeBase64(file, image) {
  const data = String(image).replace(/^data:[^;]+;base64,/, "");
  fs.writeFileSync(file, Buffer.from(data, "base64"));
}

function saveScene({ concept, svg, image, source, overlay }) {
  const id = slug(concept);
  const dir = path.join(scenesDir, id);
  fs.mkdirSync(dir, { recursive: true });
  if (svg) fs.writeFileSync(path.join(dir, overlay ? "overlay.svg" : "sketch.svg"), svg);
  if (image && !String(image).startsWith("/")) writeBase64(path.join(dir, "photo.png"), image);
  if (source && !String(source).startsWith("/")) writeBase64(path.join(dir, "source.jpg"), source);
  const record = sceneRecord(id, concept, { updated: Date.now() });
  const items = readIndex().filter((item) => item.id !== id);
  items.unshift({
    id,
    concept,
    sketch: record.sketch,
    photo: record.photo,
    source: record.source,
    updated: record.updated,
  });
  writeIndex(items);
  return record;
}

function json(res, status, body) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(body));
}

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}");
}

function sketchPrompt(concept) {
  return [
    "You are SketchDirector. Produce one comic-book / storyboard panel as SVG:",
    concept,
    "",
    "Rules:",
    `- Root: <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 390 520">`,
    "- White background rect covering the viewBox.",
    "- Black strokes only. stroke-width 1.8 to 2.6. fill=none except the background.",
    "- Clean sketch look: simple contours, no shading, no text, no labels, no watermark.",
    '- Each object in its own <g id="obj-NAME" data-name="NAME"> with a coherent spatial layout.',
    "- Include a ground/horizon if the scene needs one, as its own group.",
    "- Objects should not stack on the same point. Use the full frame.",
    "Return ONLY the SVG.",
  ].join("\n");
}

function bananaPrompt(concept) {
  return [
    "Turn this black-and-white panel sketch into a finished story or comic-book frame.",
    `Beat: ${concept}`,
    "Keep the same composition, camera, character count, and placement.",
    "Cinematic or illustrated, consistent characters, no extra objects, no text, no speech balloons.",
    "Return only the finished panel.",
  ].join(" ");
}

function overlayPrompt(concept) {
  return [
    "You are SketchDirector. The image is a still from a story.",
    concept ? `Next beat: ${concept}` : "Mark the main characters and props so they can be restaged.",
    "Draw an SVG overlay aligned to this photo, viewBox 0 0 390 520.",
    "No background rectangle. Transparent canvas.",
    "Black or #1a2610 strokes, stroke-width 2.2, fill=none.",
    "If the user named a change, sketch the objects in their NEW intended positions/sizes, and keep unchanged objects where they are.",
    'Each object in its own <g id="obj-NAME" data-name="NAME">.',
    "Use the full frame. No text, no labels, no watermark.",
    "Return ONLY the SVG.",
  ].join("\n");
}

function bananaEditPrompt(edit) {
  return [
    "The first image is a still from the story.",
    "The second image is that still with a sketch overlay of the next story beat.",
    `Apply this beat: ${edit || "Follow the sketch overlay."}`,
    "Keep the same characters unless the beat says otherwise. Cinematic or comic-book, no text, no extra objects.",
    "Return only the finished panel.",
  ].join(" ");
}

async function geminiGenerate(apiKey, model, parts, modalities) {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts }],
        generationConfig: {
          responseModalities: modalities,
          temperature: modalities.includes("IMAGE") ? 0.6 : 0.4,
          ...(modalities.includes("IMAGE")
            ? {}
            : { maxOutputTokens: 4096, thinkingConfig: { thinkingBudget: 0 } }),
        },
      }),
      signal: AbortSignal.timeout(120_000),
    }
  );
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = body.error?.message || `Gemini ${response.status}`;
    throw new Error(message.slice(0, 240));
  }
  return body.candidates?.[0]?.content?.parts ?? [];
}

function extractSvg(parts) {
  const text = parts.map((part) => part.text || "").join("\n");
  const match = text.match(/<svg[\s\S]*<\/svg>/i);
  if (!match) throw new Error("Sketch model returned no SVG.");
  return match[0];
}

function extractImage(parts) {
  for (const part of parts) {
    const inline = part.inlineData ?? part.inline_data;
    if (inline?.data) {
      const type = inline.mimeType || inline.mime_type || "image/png";
      return `data:${type};base64,${inline.data}`;
    }
  }
  throw new Error("Nano Banana returned no image.");
}

export function sketchApi() {
  return {
    name: "sketch-api",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = req.url?.split("?")[0];
        try {
          if (req.method === "GET" && url === "/api/scenes") {
            json(res, 200, { scenes: readIndex() });
            return;
          }
          if (req.method !== "POST" || !["/api/sketch", "/api/banana", "/api/save", "/api/overlay", "/api/delete"].includes(url)) {
            next();
            return;
          }
          const payload = await readBody(req);
          const concept = String(payload.concept || payload.prompt || "").trim();
          const force = Boolean(payload.force);
          const editMode = payload.mode === "edit" || Boolean(payload.overlay);

          if (url === "/api/delete") {
            json(res, 200, deleteScene(payload.id));
            return;
          }

          if (url === "/api/save") {
            if (!concept) throw Object.assign(new Error("Describe a scene."), { status: 400 });
            json(res, 200, saveScene({ concept, svg: payload.svg, image: payload.image, source: payload.source }));
            return;
          }

          const cfg = keys();
          if (url === "/api/sketch") {
            if (!concept) throw Object.assign(new Error("Describe a scene."), { status: 400 });
            const cached = loadScene(concept);
            if (cached.svg && !force) {
              json(res, 200, { svg: cached.svg, cached: true, source: "disk" });
              return;
            }
            if (!cfg.gemini) throw new Error("Set GEMINI_API_KEY in webapp_genai/genai_image_editing/.env");
            const parts = await geminiGenerate(
              cfg.gemini,
              cfg.textModel,
              [{ text: sketchPrompt(concept) }],
              ["TEXT"]
            );
            const svg = extractSvg(parts);
            saveScene({ concept, svg });
            json(res, 200, { svg, cached: false, source: "gemini" });
            return;
          }

          if (url === "/api/overlay") {
            let image = String(payload.image || "").replace(/^data:[^;]+;base64,/, "");
            let mime = String(payload.mime || "image/jpeg");
            const fromDisk = readPublicImage(payload.imageUrl || image);
            if (fromDisk) {
              image = fromDisk.data;
              mime = fromDisk.mime;
            } else if (payload.imageUrl) {
              throw Object.assign(new Error("Could not open that photo."), { status: 400 });
            }
            if (!image || image.length < 80) {
              throw Object.assign(new Error("Add a photo first."), { status: 400 });
            }
            const label = concept || "Photo edit";
            const cachedOverlay = loadScene(label);
            if (cachedOverlay.svg && !force) {
              json(res, 200, { svg: cachedOverlay.svg, cached: true, source: "disk" });
              return;
            }
            if (!cfg.gemini) throw new Error("Set GEMINI_API_KEY in webapp_genai/genai_image_editing/.env");
            const parts = await geminiGenerate(
              cfg.gemini,
              cfg.textModel,
              [
                { text: overlayPrompt(label) },
                { inline_data: { mime_type: mime, data: image } },
              ],
              ["TEXT"]
            );
            const svg = extractSvg(parts);
            saveScene({ concept: label, svg, overlay: true, source: `data:${mime};base64,${image}` });
            json(res, 200, { svg, cached: false, source: "gemini" });
            return;
          }

          if (!concept) throw Object.assign(new Error("Describe a scene."), { status: 400 });
          const cached = loadScene(concept);
          if (cached.photo && !force && !editMode) {
            json(res, 200, { image: cached.photo, cached: true });
            return;
          }
          const image = String(payload.image || "").replace(/^data:[^;]+;base64,/, "");
          const mime = String(payload.mime || "image/png");
          if (!image) throw Object.assign(new Error("Missing sketch image."), { status: 400 });
          if (!cfg.gemini) throw new Error("Set GEMINI_API_KEY in webapp_genai/genai_image_editing/.env");
          const overlay = String(payload.overlay || "").replace(/^data:[^;]+;base64,/, "");
          const parts = overlay
            ? [
                { text: bananaEditPrompt(payload.prompt || concept) },
                { inline_data: { mime_type: mime, data: image } },
                { inline_data: { mime_type: payload.overlayMime || "image/png", data: overlay } },
              ]
            : [
                { text: bananaPrompt(concept || "a sketched scene") },
                { inline_data: { mime_type: mime, data: image } },
              ];
          const result = await geminiGenerate(cfg.gemini, cfg.imageModel, parts, ["TEXT", "IMAGE"]);
          const photo = extractImage(result);
          saveScene({ concept, svg: payload.svg || cached.svg, image: photo, source: overlay ? `data:${mime};base64,${image}` : undefined });
          json(res, 200, { image: photo, cached: false });
        } catch (error) {
          if (error instanceof SyntaxError) {
            json(res, 400, { error: "Invalid JSON." });
            return;
          }
          json(res, error.status || 502, { error: (error.message || "Request failed").slice(0, 240) });
        }
      });
    },
  };
}
