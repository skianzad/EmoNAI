import { NextResponse } from "next/server";
import { editorChoices, type EditorId } from "@/data/editors";
import { COLOR_MAP_PROMPT, FACE_COLOR_MAP_PROMPT, ROOM_COLOR_MAP_PROMPT } from "@/lib/colorMapPrompt";
import { key } from "@/lib/keys";
import { runEditor } from "@/lib/runEdit";
import { MAX_IMAGE_BYTES } from "@/lib/shrink";

export const maxDuration = 120;

const ids = new Set<string>(editorChoices.map((editor) => editor.id));

type NamedColor = { name: string; color: string };

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const editor = String(body?.editor ?? "");
  const image = String(body?.image ?? "");
  const mime = String(body?.mime ?? "image/jpeg");
  const custom = typeof body?.prompt === "string" ? body.prompt.trim() : "";
  const prompt = custom
    ? custom.slice(0, 6000)
    : body?.kind === "face"
      ? FACE_COLOR_MAP_PROMPT
      : body?.kind === "room"
        ? ROOM_COLOR_MAP_PROMPT
        : COLOR_MAP_PROMPT;

  if (!ids.has(editor) || !image) {
    return NextResponse.json({ error: "Choose an editor and a photo." }, { status: 400 });
  }
  if (Buffer.byteLength(image, "base64") > MAX_IMAGE_BYTES) {
    return NextResponse.json({ error: "Photo must be 500 KB or smaller." }, { status: 413 });
  }

  try {
    if (editor === "nano_banana" || editor === "nano_banana_lite") {
      const result = await nanoBananaMap(image, mime, prompt);
      return NextResponse.json(result);
    }
    const painted = await runEditor(editor as EditorId, prompt, image, mime);
    return NextResponse.json({ image: painted, items: [] satisfies NamedColor[] });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Edit failed";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}

async function nanoBananaMap(image: string, mime: string, prompt: string) {
  const apiKey = key("GEMINI_API_KEY");
  if (!apiKey) throw new Error("No key for Nano Banana");
  const model = key("GEMINI_IMAGE_MODEL") || "gemini-3.1-flash-image";
  let response: Response;
  try {
    response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [
            {
              parts: [{ text: prompt }, { inline_data: { mime_type: mime, data: image } }],
            },
          ],
          generationConfig: { responseModalities: ["TEXT", "IMAGE"] },
        }),
        signal: AbortSignal.timeout(120_000),
      }
    );
  } catch (error) {
    const detail = error instanceof Error ? error.message : "network error";
    throw new Error(`Nano Banana unreachable (${detail}). Try again in a moment.`);
  }
  const textBody = await response.text();
  if (!response.ok) throw new Error(`Nano Banana ${response.status}: ${textBody.slice(0, 240)}`);
  const body = JSON.parse(textBody);
  const parts = body.candidates?.[0]?.content?.parts ?? [];
  let painted = "";
  let note = "";
  for (const part of parts) {
    if (typeof part.text === "string") note += part.text;
    const inline = part.inlineData ?? part.inline_data;
    if (inline?.data) painted = `data:${inline.mimeType || inline.mime_type || "image/png"};base64,${inline.data}`;
  }
  if (!painted) throw new Error("Nano Banana returned no image");
  return { image: painted, items: parseItems(note) };
}

function parseItems(text: string): NamedColor[] {
  const match = text.match(/\[[\s\S]*\]/);
  if (match) {
    try {
      const data = JSON.parse(match[0]) as unknown;
      if (Array.isArray(data)) {
        const items = data.flatMap((item) => {
          if (!item || typeof item !== "object") return [];
          const name = String((item as { name?: unknown }).name ?? "").trim();
          const color = normalizeHex(String((item as { color?: unknown }).color ?? ""));
          if (!name || !color) return [];
          return [{ name, color }];
        });
        if (items.length > 0) return items;
      }
    } catch {
      // Fall through to "bag (#FF0023)" lines.
    }
  }
  const found: NamedColor[] = [];
  for (const line of text.matchAll(/([A-Za-z][A-Za-z0-9 ]{0,24}?)\s*\(?#([0-9a-fA-F]{6})\)?/g)) {
    found.push({ name: line[1].trim(), color: `#${line[2].toUpperCase()}` });
  }
  return found;
}

function normalizeHex(value: string) {
  const match = value.trim().match(/^#?([0-9a-fA-F]{6})$/);
  return match ? `#${match[1].toUpperCase()}` : "";
}
