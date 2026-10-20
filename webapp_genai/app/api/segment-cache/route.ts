import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 60;

type CacheKind = "room" | "face" | "crowd";

function asKind(value: unknown): CacheKind {
  if (value === "room" || value === "crowd") return value;
  return "face";
}

function segmentsFile(kind: CacheKind) {
  if (kind === "room") return path.join(process.cwd(), "data", "room-segments.json");
  if (kind === "crowd") return path.join(process.cwd(), "data", "crowd-segments.json");
  return path.join(process.cwd(), "data", "face-scene", "meta.json");
}

function sceneFolder(kind: CacheKind) {
  if (kind === "room") return "room-scene";
  if (kind === "crowd") return "crowd-scene";
  return "face-scene";
}

function backgroundFile(kind: CacheKind) {
  return path.join(process.cwd(), "data", sceneFolder(kind), "background.jpg");
}

/** Public copy so the empty courtyard loads as a normal image URL (not a huge data URL). */
function publicBackgroundFile(kind: CacheKind) {
  if (kind === "crowd") return path.join(process.cwd(), "public", "scenes", "louvre", "empty.jpg");
  if (kind === "room") return path.join(process.cwd(), "public", "scenes", "living", "empty.jpg");
  return "";
}

function publicBackgroundUrl(kind: CacheKind) {
  if (kind === "crowd") return "/scenes/louvre/empty.jpg";
  if (kind === "room") return "/scenes/living/empty.jpg";
  return "";
}

function layerFile(kind: CacheKind, id: string) {
  const safe = id.replace(/[^a-zA-Z0-9_-]/g, "_");
  return path.join(process.cwd(), "data", sceneFolder(kind), "layers", `${safe}.png`);
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
    next.hidden = Boolean(next.suggested);
    delete next.cover;
    delete next.blur;
    return next;
  });
}

export async function GET(request: Request) {
  const kind = asKind(new URL(request.url).searchParams.get("kind"));
  try {
    const bgPath = backgroundFile(kind);
    const hasBg = await exists(bgPath);
    const publicUrl = publicBackgroundUrl(kind);
    const publicPath = publicBackgroundFile(kind);
    // Prefer a stable public URL when the empty background is on disk.
    let background = "";
    if (hasBg && publicUrl && publicPath) {
      // Keep public copy in sync if data/ has the file but public/ does not.
      if (!(await exists(publicPath))) {
        await fs.mkdir(path.dirname(publicPath), { recursive: true });
        await fs.copyFile(bgPath, publicPath);
      }
      const stamp = (await fs.stat(bgPath)).mtimeMs;
      background = `${publicUrl}?t=${Math.round(stamp)}`;
    } else if (hasBg) {
      background = await readDataUrl(bgPath, "image/jpeg");
    }

    if (kind === "room" || kind === "crowd") {
      try {
        const text = await fs.readFile(segmentsFile(kind), "utf8");
        const saved = JSON.parse(text) as { objects?: Array<Record<string, unknown>> };
        const listed = Array.isArray(saved.objects) ? saved.objects : [];
        const objects = await Promise.all(
          asBaseObjects(listed).map(async (object) => {
            const id = typeof object.id === "string" ? object.id : "";
            const layer = id ? await readDataUrl(layerFile(kind, id), "image/png") : "";
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

    const text = await fs.readFile(segmentsFile("face"), "utf8");
    const saved = JSON.parse(text) as { objects?: Array<Record<string, unknown>> };
    const listed = Array.isArray(saved.objects) ? saved.objects : [];
    const objects = await Promise.all(
      listed.map(async (object) => {
        const id = typeof object.id === "string" ? object.id : "";
        const layer = id ? await readDataUrl(layerFile("face", id), "image/png") : "";
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
  const kind = asKind(body?.kind);
  const objects = Array.isArray(body?.objects) ? body.objects : [];
  const background = typeof body?.background === "string" ? body.background : "";
  const usable = objects.filter(
    (object: { mask?: unknown; name?: unknown; id?: unknown }) =>
      object && typeof object.mask === "string" && typeof object.name === "string" && typeof object.id === "string"
  );

  // Empty background: write once, never overwrite (unless crowd forceReplace).
  const forceBg = Boolean(body?.replaceBackground) && kind === "crowd";
  if (background.startsWith("data:")) {
    const bgPath = backgroundFile(kind);
    await fs.mkdir(path.dirname(bgPath), { recursive: true });
    if (forceBg || !(await exists(bgPath))) {
      const parsed = parseDataUrl(background);
      if (parsed) {
        await fs.writeFile(bgPath, parsed.buffer);
        const pub = publicBackgroundFile(kind);
        if (pub) {
          await fs.mkdir(path.dirname(pub), { recursive: true });
          await fs.writeFile(pub, parsed.buffer);
        }
      }
    }
  }

  if (kind === "room" || kind === "crowd") {
    if (usable.length > 0) {
      const file = segmentsFile(kind);
      await fs.mkdir(path.dirname(file), { recursive: true });
      const incoming = asBaseObjects(usable as Array<Record<string, unknown>>);
      // Crowd bake replaces the whole set; room merges by name.
      let merged = incoming;
      if (kind === "room" && (await exists(file))) {
        try {
          const prior = JSON.parse(await fs.readFile(file, "utf8")) as {
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
      await fs.writeFile(file, JSON.stringify({ objects: merged }));
    }
    const hasBg = await exists(backgroundFile(kind));
    const hasSeg = await exists(segmentsFile(kind));
    if (!hasBg && !hasSeg && !background.startsWith("data:") && usable.length === 0) {
      return NextResponse.json({ error: "Nothing to save." }, { status: 400 });
    }
    return NextResponse.json({ ok: true, count: hasSeg ? usable.length : 0, background: hasBg });
  }

  if (usable.length === 0) {
    return NextResponse.json({ error: "Nothing to save." }, { status: 400 });
  }

  const layerDir = path.join(process.cwd(), "data", "face-scene", "layers");
  await fs.mkdir(layerDir, { recursive: true });

  const metaObjects = [];
  for (const object of usable) {
    const { layer, ...rest } = object as { layer?: string; id: string; name: string; mask: string };
    if (typeof layer === "string" && layer.startsWith("data:")) {
      const parsed = parseDataUrl(layer);
      if (parsed) await fs.writeFile(layerFile("face", object.id), parsed.buffer);
    }
    metaObjects.push(rest);
  }

  await fs.writeFile(segmentsFile("face"), JSON.stringify({ objects: metaObjects }, null, 0));
  const hasBg = await exists(backgroundFile("face"));
  return NextResponse.json({ ok: true, count: metaObjects.length, background: hasBg });
}
