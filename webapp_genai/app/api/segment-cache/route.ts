import { promises as fs } from "fs";
import path from "path";
import { NextResponse } from "next/server";

function fileFor(kind: string) {
  const name = kind === "room" ? "room-segments.json" : "face-segments.json";
  return path.join(process.cwd(), "data", name);
}

export async function GET(request: Request) {
  const kind = new URL(request.url).searchParams.get("kind") === "room" ? "room" : "face";
  try {
    const text = await fs.readFile(fileFor(kind), "utf8");
    const saved = JSON.parse(text) as { objects?: unknown };
    const objects = Array.isArray(saved.objects) ? saved.objects : [];
    return NextResponse.json({ objects });
  } catch {
    return NextResponse.json({ objects: [] });
  }
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const kind = body?.kind === "room" ? "room" : "face";
  const objects = Array.isArray(body?.objects) ? body.objects : [];
  const usable = objects.filter((object: { mask?: unknown; name?: unknown }) => object && typeof object.mask === "string" && typeof object.name === "string");
  if (usable.length === 0) {
    return NextResponse.json({ error: "Nothing to save." }, { status: 400 });
  }
  const file = fileFor(kind);
  await fs.mkdir(path.dirname(file), { recursive: true });
  await fs.writeFile(file, JSON.stringify({ objects: usable }));
  return NextResponse.json({ ok: true, count: usable.length });
}
