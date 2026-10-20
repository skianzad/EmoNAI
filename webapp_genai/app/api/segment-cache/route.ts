import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

function rootFor(kind: string) {
  const folder = kind === "room" ? "room-scene" : "face-scene";
  return path.join(process.cwd(), "data", folder);
}

function metaFile(kind: string) {
  return path.join(rootFor(kind), "meta.json");
}

function backgroundFile(kind: string) {
  return path.join(rootFor(kind), "background.jpg");
}

function layerFile(kind: string, id: string) {
  const safe = id.replace(/[^a-zA-Z0-9_-]/g, "_");
  return path.join(rootFor(kind), "layers", `${safe}.png`);
}

function parseDataUrl(value: string) {
  const match = /^data:([^;]+);base64,([\s\S]+)$/.exec(value);
  if (!match) return null;
  return { mime: match[1], buffer: Buffer.from(match[2], "base64") };
}

async function readDataUrl(file: string, mime: string) {
  try {
    const buffer = await fs.readFile(file);
    return `data:${mime};base64,${buffer.toString("base64")}`;
  } catch {
    return "";
  }
}

export async function GET(request: Request) {
  const kind = new URL(request.url).searchParams.get("kind") === "room" ? "room" : "face";
  try {
    const background = await readDataUrl(backgroundFile(kind), "image/jpeg");
    // Room: only the empty background is durable. Edits stay in memory.
    if (kind === "room") {
      return NextResponse.json({ objects: [], background });
    }
    const text = await fs.readFile(metaFile(kind), "utf8");
    const saved = JSON.parse(text) as { objects?: Array<Record<string, unknown>> };
    const listed = Array.isArray(saved.objects) ? saved.objects : [];
    const objects = await Promise.all(
      listed.map(async (object) => {
        const id = typeof object.id === "string" ? object.id : "";
        const layer = id ? await readDataUrl(layerFile(kind, id), "image/png") : "";
        const next = { ...object };
        if (layer) next.layer = layer;
        else delete next.layer;
        return next;
      })
    );
    return NextResponse.json({ objects, background });
  } catch {
    return NextResponse.json({ objects: [], background: "" });
  }
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const kind = body?.kind === "room" ? "room" : "face";
  const objects = Array.isArray(body?.objects) ? body.objects : [];
  const background = typeof body?.background === "string" ? body.background : "";
  const usable = objects.filter(
    (object: { mask?: unknown; name?: unknown; id?: unknown }) =>
      object && typeof object.mask === "string" && typeof object.name === "string" && typeof object.id === "string"
  );

  const root = rootFor(kind);
  await fs.mkdir(root, { recursive: true });

  // Room persists only the empty background — never segment edits or layers.
  if (kind === "room") {
    if (background.startsWith("data:")) {
      const bgPath = backgroundFile(kind);
      const exists = await fs
        .access(bgPath)
        .then(() => true)
        .catch(() => false);
      if (!exists) {
        const parsed = parseDataUrl(background);
        if (parsed) await fs.writeFile(bgPath, parsed.buffer);
      }
    }
    const hasBg = await fs
      .access(backgroundFile(kind))
      .then(() => true)
      .catch(() => false);
    // Drop any old meta/layers so edits never reload after restart.
    await fs.rm(path.join(root, "layers"), { recursive: true, force: true }).catch(() => undefined);
    await fs.rm(metaFile(kind), { force: true }).catch(() => undefined);
    if (!hasBg && !background.startsWith("data:")) {
      return NextResponse.json({ error: "Nothing to save." }, { status: 400 });
    }
    return NextResponse.json({ ok: true, count: 0, background: hasBg || background.startsWith("data:") });
  }

  if (usable.length === 0) {
    return NextResponse.json({ error: "Nothing to save." }, { status: 400 });
  }

  const layerDir = path.join(root, "layers");
  await fs.mkdir(layerDir, { recursive: true });

  if (background.startsWith("data:")) {
    const parsed = parseDataUrl(background);
    if (parsed) await fs.writeFile(backgroundFile(kind), parsed.buffer);
  }

  const metaObjects = [];
  for (const object of usable) {
    const { layer, ...rest } = object as { layer?: string; id: string; name: string; mask: string };
    if (typeof layer === "string" && layer.startsWith("data:")) {
      const parsed = parseDataUrl(layer);
      if (parsed) await fs.writeFile(layerFile(kind, object.id), parsed.buffer);
    }
    metaObjects.push(rest);
  }

  await fs.writeFile(metaFile(kind), JSON.stringify({ objects: metaObjects }, null, 0));
  const hasBg = await fs
    .access(backgroundFile(kind))
    .then(() => true)
    .catch(() => false);

  return NextResponse.json({ ok: true, count: metaObjects.length, background: hasBg });
}
