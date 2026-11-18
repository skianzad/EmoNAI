import fs from "fs";
import path from "path";
import { key } from "@/lib/keys";

const scenesDir = path.join(process.cwd(), "public", "sd-scenes");
const indexFile = path.join(scenesDir, "index.json");
const libraryFile = path.join(process.cwd(), "public", "sd-library", "index.json");

function slug(concept: string) {
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
    return JSON.parse(fs.readFileSync(indexFile, "utf8")) as Record<string, unknown>[];
  } catch {
    return [];
  }
}

function writeIndex(items: unknown[]) {
  fs.mkdirSync(scenesDir, { recursive: true });
  fs.writeFileSync(indexFile, JSON.stringify(items, null, 2));
}

function urls(id: string) {
  return {
    sketch: `/sd-scenes/${id}/sketch.svg`,
    photo: `/sd-scenes/${id}/photo.png`,
    source: `/sd-scenes/${id}/source.jpg`,
  };
}

function sceneRecord(id: string, concept: string, extra: Record<string, unknown> = {}) {
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

function loadScene(concept: string) {
  const id = slug(concept);
  const dir = path.join(scenesDir, id);
  const sketchPath = path.join(dir, "sketch.svg");
  const overlayPath = path.join(dir, "overlay.svg");
  return {
    id,
    concept,
    svg: fs.existsSync(sketchPath)
      ? fs.readFileSync(sketchPath, "utf8")
      : fs.existsSync(overlayPath)
        ? fs.readFileSync(overlayPath, "utf8")
        : "",
    photo: fs.existsSync(path.join(dir, "photo.png")) ? urls(id).photo : null,
    source: fs.existsSync(path.join(dir, "source.jpg")) ? urls(id).source : null,
  };
}

function writeBase64(file: string, image: string) {
  fs.writeFileSync(file, Buffer.from(String(image).replace(/^data:[^;]+;base64,/, ""), "base64"));
}

function saveScene({
  concept,
  svg,
  image,
  source,
  overlay,
}: {
  concept: string;
  svg?: string;
  image?: string;
  source?: string;
  overlay?: boolean;
}) {
  const id = slug(concept);
  const dir = path.join(scenesDir, id);
  fs.mkdirSync(dir, { recursive: true });
  if (svg) fs.writeFileSync(path.join(dir, overlay ? "overlay.svg" : "sketch.svg"), svg);
  if (image && !String(image).startsWith("/")) writeBase64(path.join(dir, "photo.png"), image);
  if (source && !String(source).startsWith("/")) writeBase64(path.join(dir, "source.jpg"), source);
  const record = sceneRecord(id, concept, { updated: Date.now() });
  writeIndex([record, ...readIndex().filter((item) => item.id !== id)]);
  return record;
}

function readPublicImage(url: string) {
  const clean = String(url || "").split("?")[0];
  if (!clean.startsWith("/sd-scenes/")) return null;
  const file = path.normalize(path.join(process.cwd(), "public", clean.replace(/^\//, "")));
  const allowed = scenesDir;
  if (!file.startsWith(allowed) || !fs.existsSync(file)) return null;
  return {
    mime: file.endsWith(".png") ? "image/png" : "image/jpeg",
    data: fs.readFileSync(file).toString("base64"),
  };
}

function models() {
  return {
    gemini: key("GEMINI_API_KEY"),
    imageModel: key("GEMINI_IMAGE_MODEL") || "gemini-3.1-flash-image",
    textModel: key("GEMINI_TEXT_MODEL") || "gemini-3.8-flash",
  };
}

async function geminiGenerate(apiKey: string, model: string, parts: unknown[], modalities: string[]) {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts }],
        generationConfig: {
          responseModalities: modalities,
          temperature: modalities.includes("IMAGE") ? 0.7 : 0.4,
          ...(modalities.includes("IMAGE")
            ? {}
            : { maxOutputTokens: 4096, thinkingConfig: { thinkingBudget: 0 } }),
        },
      }),
      signal: AbortSignal.timeout(120_000),
    }
  );
  const body = (await response.json().catch(() => ({}))) as { error?: { message?: string }; candidates?: { content?: { parts?: { text?: string; inlineData?: { data?: string; mimeType?: string }; inline_data?: { data?: string; mime_type?: string } }[] } }[] };
  if (!response.ok) {
    throw new Error((body.error?.message || `Gemini ${response.status}`).slice(0, 240));
  }
  return body.candidates?.[0]?.content?.parts ?? [];
}

function extractSvg(parts: { text?: string }[]) {
  const text = parts.map((part) => part.text || "").join("\n");
  const match = text.match(/<svg[\s\S]*<\/svg>/i);
  if (!match) throw new Error("Sketch model returned no SVG.");
  return match[0];
}

function extractImage(parts: { inlineData?: { data?: string; mimeType?: string; mime_type?: string }; inline_data?: { data?: string; mimeType?: string; mime_type?: string } }[]) {
  for (const part of parts) {
    const inline = part.inlineData ?? part.inline_data;
    if (inline?.data) {
      const type = inline.mimeType || inline.mime_type || "image/png";
      return `data:${type};base64,${inline.data}`;
    }
  }
  throw new Error("Nano Banana returned no image.");
}

function readLibrary() {
  try {
    const items = JSON.parse(fs.readFileSync(libraryFile, "utf8"));
    return Array.isArray(items) ? items : [];
  } catch {
    return [];
  }
}

function writeLibrary(items: unknown[]) {
  fs.mkdirSync(path.dirname(libraryFile), { recursive: true });
  fs.writeFileSync(libraryFile, JSON.stringify(items, null, 2));
}

export async function handleSketch(action: string, method: string, payload: Record<string, unknown> = {}) {
  if (method === "GET" && action === "scenes") {
    return { scenes: readIndex() };
  }
  if (method === "GET" && action === "library") {
    return { symbols: readLibrary() };
  }
  if (action === "library") {
    const incoming = Array.isArray(payload.symbols) ? payload.symbols : [payload];
    const next = [...readLibrary()];
    for (const item of incoming) {
      const id = String((item as { id?: string }).id || "").replace(/[^a-z0-9\u4e00-\u9fff-]/gi, "").slice(0, 32);
      const body = String((item as { body?: string }).body || "");
      if (!id || !body) continue;
      const record = { ...(item as object), id, body };
      const index = next.findIndex((row) => (row as { id?: string }).id === id);
      if (index >= 0) next[index] = record;
      else next.unshift(record);
    }
    writeLibrary(next.slice(0, 80));
    return { symbols: next.slice(0, 80) };
  }
  const concept = String(payload.concept || payload.prompt || "").trim();
  const force = Boolean(payload.force);
  const editMode = payload.mode === "edit" || Boolean(payload.overlay);

  if (action === "delete") {
    const safe = String(payload.id || "").replace(/[^a-z0-9-]/g, "");
    if (!safe) throw Object.assign(new Error("Missing scene."), { status: 400 });
    const dir = path.join(scenesDir, safe);
    if (!dir.startsWith(scenesDir + path.sep)) throw Object.assign(new Error("Invalid scene."), { status: 400 });
    fs.rmSync(dir, { recursive: true, force: true });
    writeIndex(readIndex().filter((item) => item.id !== safe));
    return { id: safe };
  }

  if (action === "save") {
    if (!concept) throw Object.assign(new Error("Describe a scene."), { status: 400 });
    return saveScene({
      concept,
      svg: payload.svg as string | undefined,
      image: payload.image as string | undefined,
      source: payload.source as string | undefined,
    });
  }

  const cfg = models();
  if (action === "sketch") {
    if (!concept) throw Object.assign(new Error("Describe a scene."), { status: 400 });
    const cached = loadScene(concept);
    const keep = keepNames(payload);
    if (cached.svg && !force && keep.length === 0) return { svg: cached.svg, cached: true, source: "disk" };
    if (!cfg.gemini) throw new Error("Set GEMINI_API_KEY in genai_image_editing/.env");
    const parts = await geminiGenerate(cfg.gemini, cfg.textModel, [{ text: sketchPrompt(concept, keep) }], ["TEXT"]);
    const svg = extractSvg(parts);
    saveScene({ concept, svg });
    return { svg, cached: false, source: "gemini" };
  }

  if (action === "overlay") {
    let image = String(payload.image || "").replace(/^data:[^;]+;base64,/, "");
    let mime = String(payload.mime || "image/jpeg");
    const fromDisk = readPublicImage(String(payload.imageUrl || image));
    if (fromDisk) {
      image = fromDisk.data;
      mime = fromDisk.mime;
    } else if (payload.imageUrl) {
      throw Object.assign(new Error("Could not open that photo."), { status: 400 });
    }
    if (!image || image.length < 80) throw Object.assign(new Error("Add a photo first."), { status: 400 });
    const label = concept || "Photo edit";
    const cachedOverlay = loadScene(label);
    const keep = keepNames(payload);
    if (cachedOverlay.svg && !force && keep.length === 0) return { svg: cachedOverlay.svg, cached: true, source: "disk" };
    if (!cfg.gemini) throw new Error("Set GEMINI_API_KEY in genai_image_editing/.env");
    const parts = await geminiGenerate(
      cfg.gemini,
      cfg.textModel,
      [{ text: overlayPrompt(label, keep) }, { inline_data: { mime_type: mime, data: image } }],
      ["TEXT"]
    );
    const svg = extractSvg(parts);
    saveScene({ concept: label, svg, overlay: true, source: `data:${mime};base64,${image}` });
    return { svg, cached: false, source: "gemini" };
  }

  if (action === "banana") {
    if (!concept) throw Object.assign(new Error("Describe a scene."), { status: 400 });
    const cached = loadScene(concept);
    if (cached.photo && !force && !editMode && payload.svg && cached.svg === payload.svg) {
      return { image: cached.photo, cached: true };
    }
    const image = String(payload.image || "").replace(/^data:[^;]+;base64,/, "");
    const mime = String(payload.mime || "image/png");
    if (!image) throw Object.assign(new Error("Missing sketch image."), { status: 400 });
    if (!cfg.gemini) throw new Error("Set GEMINI_API_KEY in genai_image_editing/.env");
    const overlay = String(payload.overlay || "").replace(/^data:[^;]+;base64,/, "");
    const pose = String(payload.pose || "");
    const layout = String(payload.layout || "");
    const scene = String(payload.prompt || concept || "a sketched scene");
    const parts = overlay
      ? [
          { text: bananaEditPrompt(scene, pose, layout) },
          { inline_data: { mime_type: mime, data: image } },
          { inline_data: { mime_type: String(payload.overlayMime || "image/png"), data: overlay } },
        ]
      : [{ text: bananaPrompt(scene, pose, layout) }, { inline_data: { mime_type: mime, data: image } }];
    const result = await geminiGenerate(cfg.gemini, cfg.imageModel, parts, ["TEXT", "IMAGE"]);
    const photo = extractImage(result);
    saveScene({
      concept,
      svg: (payload.svg as string) || cached.svg,
      image: photo,
      source: overlay ? `data:${mime};base64,${image}` : undefined,
    });
    return { image: photo, cached: false };
  }

  throw Object.assign(new Error("Unknown sketch action."), { status: 404 });
}

function keepNames(payload: Record<string, unknown>) {
  return Array.isArray(payload.keep) ? payload.keep.map((item) => String(item)).filter(Boolean) : [];
}

function keepLine(keep: string[]) {
  if (!keep.length) return "";
  return `Do not draw these objects; they are already stamped on the canvas and must be kept: ${keep.join(", ")}.`;
}

function sketchPrompt(concept: string, keep: string[] = []) {
  return [
    "You are SketchDirector. Produce one multi-object line sketch as SVG for this scene:",
    concept,
    keepLine(keep),
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

function bananaPrompt(concept: string, pose = "", layout = "") {
  return [
    `Take a natural handheld photograph of this scene: ${concept}.`,
    "The attached drawing is only a blocking sketch. Thin green cubes show each subject's place and 3D orientation. Cubes are not objects — never draw cubes, boxes, arrows, numbers, captions, or watermarks.",
    layout,
    pose,
    "Keep those places and facings, then shoot it like a real photo: believable lighting, depth, ground, and atmosphere. Real animals and objects, not clipart or a diagram.",
    "Same subjects, no extras. Return only the photograph.",
  ]
    .filter(Boolean)
    .join("\n");
}

function overlayPrompt(concept: string, keep: string[] = []) {
  return [
    "You are SketchDirector. The image is a real photograph.",
    concept ? `Intended edit: ${concept}` : "Mark the main objects so they can be rearranged.",
    keepLine(keep),
    "Draw an SVG overlay aligned to this photo, viewBox 0 0 390 520.",
    "No background rectangle. Transparent canvas.",
    "Black or #1a2610 strokes, stroke-width 2.2, fill=none.",
    "If the user named a change, sketch the objects in their NEW intended positions/sizes, and keep unchanged objects where they are.",
    'Each object in its own <g id="obj-NAME" data-name="NAME">.',
    "Use the full frame. No text, no labels, no watermark.",
    "Return ONLY the SVG.",
  ].join("\n");
}

function bananaEditPrompt(edit: string, pose = "", layout = "") {
  return [
    "Image 1 is the original photograph. Image 2 is a blocking overlay with thin green cubes for place and 3D pose. Cubes are not objects — never draw cubes, numbers, or labels.",
    `Apply this edit as a natural photograph: ${edit || "Follow the overlay."}`,
    layout,
    pose,
    "Match overlay location and facing, keep real lighting and the original people/setting unless the edit says otherwise. No text.",
    "Return only the edited photograph.",
  ]
    .filter(Boolean)
    .join("\n");
}
