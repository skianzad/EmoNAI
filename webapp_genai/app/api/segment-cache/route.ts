import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

/** Base room masks from the last good push. Edits never overwrite this file. */
function roomSegmentsFile() {
  return path.join(process.cwd(), "data", "room-segments.json");
}

function faceRoot() {
  return path.join(process.cwd(), "data", "face-scene");
}

function faceMetaFile() {
  return path.join(faceRoot(), "meta.json");
}

function faceLayerFile(id: string) {
  const safe = id.replace(/[^a-zA-Z0-9_-]/g, "_");
  return path.join(faceRoot(), "layers", `${safe}.png`);
}

function backgroundFile(kind: string) {
  const folder = kind === "room" ? "room-scene" : "face-scene";
  return path.join(process.cwd(), "data", folder, "background.jpg");
}

function roomLayerFile(id: string) {
  const safe = id.replace(/[^a-zA-Z0-9_-]/g, "_");
  return path.join(process.cwd(), "data", "room-scene", "layers", `${safe}.png`);
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

async function exists(file: string) {
  return fs
    .access(file)
    .then(() => true)
    .catch(() => false);
}

function asBaseObjects(listed: Array<Record<string, unknown>>) {
  return listed.map((object) => {
    const next = { ...object };
    delete next.layer;
    next.tint = null;
    next.shade = 0;
    next.dx = 0;
    next.dy = 0;
    next.scale = 100;
    // Suggested extras (e.g. cushion) stay hidden until Add reveals them.
    next.hidden = Boolean(next.suggested);
    delete next.cover;
    delete next.blur;
    return next;
  });
}

export async function GET(request: Request) {
  const kind = new URL(request.url).searchParams.get("kind") === "room" ? "room" : "face";
  try {
    const background = await readDataUrl(backgroundFile(kind), "image/jpeg");

    if (kind === "room") {
      try {
        const text = await fs.readFile(roomSegmentsFile(), "utf8");
        const saved = JSON.parse(text) as { objects?: Array<Record<string, unknown>> };
        const listed = Array.isArray(saved.objects) ? saved.objects : [];
        const objects = await Promise.all(
          asBaseObjects(listed).map(async (object) => {
            const id = typeof object.id === "string" ? object.id : "";
            const layer = id ? await readDataUrl(roomLayerFile(id), "image/png") : "";
            if (layer) object.layer = layer;
            else delete object.layer;
            return object;
          })
        );
        return NextResponse.json({ objects, background });
      } catch {
        return NextResponse.json({ objects: [], background });
      }
    }

    const text = await fs.readFile(faceMetaFile(), "utf8");
    const saved = JSON.parse(text) as { objects?: Array<Record<string, unknown>> };
    const listed = Array.isArray(saved.objects) ? saved.objects : [];
    const objects = await Promise.all(
      listed.map(async (object) => {
        const id = typeof object.id === "string" ? object.id : "";
        const layer = id ? await readDataUrl(faceLayerFile(id), "image/png") : "";
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

  // Empty background: write once, never overwrite.
  if (background.startsWith("data:")) {
    const bgPath = backgroundFile(kind);
    await fs.mkdir(path.dirname(bgPath), { recursive: true });
    if (!(await exists(bgPath))) {
      const parsed = parseDataUrl(background);
      if (parsed) await fs.writeFile(bgPath, parsed.buffer);
    }
  }

  if (kind === "room") {
    // Base masks only (no session edits). Create or merge new named segments (e.g. cushion).
    if (usable.length > 0) {
      await fs.mkdir(path.dirname(roomSegmentsFile()), { recursive: true });
      const incoming = asBaseObjects(usable as Array<Record<string, unknown>>);
      let merged = incoming;
      if (await exists(roomSegmentsFile())) {
        try {
          const prior = JSON.parse(await fs.readFile(roomSegmentsFile(), "utf8")) as {
            objects?: Array<Record<string, unknown>>;
          };
          const existing = Array.isArray(prior.objects) ? asBaseObjects(prior.objects) : [];
          const byName = new Map(
            existing.map((object) => [String(object.name || "").toLowerCase(), object] as const)
          );
          for (const object of incoming) {
            byName.set(String(object.name || "").toLowerCase(), object);
          }
          merged = [...byName.values()];
        } catch {
          merged = incoming;
        }
      }
      await fs.writeFile(roomSegmentsFile(), JSON.stringify({ objects: merged }));
    }
    const hasBg = await exists(backgroundFile("room"));
    const hasSeg = await exists(roomSegmentsFile());
    if (!hasBg && !hasSeg && !background.startsWith("data:") && usable.length === 0) {
      return NextResponse.json({ error: "Nothing to save." }, { status: 400 });
    }
    return NextResponse.json({ ok: true, count: hasSeg ? usable.length : 0, background: hasBg });
  }

  if (usable.length === 0) {
    return NextResponse.json({ error: "Nothing to save." }, { status: 400 });
  }

  const root = faceRoot();
  const layerDir = path.join(root, "layers");
  await fs.mkdir(layerDir, { recursive: true });

  const metaObjects = [];
  for (const object of usable) {
    const { layer, ...rest } = object as { layer?: string; id: string; name: string; mask: string };
    if (typeof layer === "string" && layer.startsWith("data:")) {
      const parsed = parseDataUrl(layer);
      if (parsed) await fs.writeFile(faceLayerFile(object.id), parsed.buffer);
    }
    metaObjects.push(rest);
  }

  await fs.writeFile(faceMetaFile(), JSON.stringify({ objects: metaObjects }, null, 0));
  const hasBg = await exists(backgroundFile("face"));
  return NextResponse.json({ ok: true, count: metaObjects.length, background: hasBg });
}
