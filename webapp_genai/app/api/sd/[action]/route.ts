import { NextResponse } from "next/server";
import { handleSketch } from "@/lib/sketchDirector";

export const maxDuration = 120;

const POSTS = new Set(["sketch", "overlay", "banana", "save", "delete", "library"]);

export async function GET(_request: Request, context: { params: Promise<{ action: string }> }) {
  const { action } = await context.params;
  if (action !== "scenes" && action !== "library") {
    return NextResponse.json({ error: "Unknown sketch action." }, { status: 404 });
  }
  try {
    return NextResponse.json(await handleSketch(action, "GET"));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Request failed";
    return NextResponse.json({ error: message.slice(0, 240) }, { status: 502 });
  }
}

export async function POST(request: Request, context: { params: Promise<{ action: string }> }) {
  const { action } = await context.params;
  if (!POSTS.has(action)) {
    return NextResponse.json({ error: "Unknown sketch action." }, { status: 404 });
  }
  const payload = (await request.json().catch(() => null)) as Record<string, unknown> | null;
  if (!payload) {
    return NextResponse.json({ error: "Invalid JSON." }, { status: 400 });
  }
  try {
    return NextResponse.json(await handleSketch(action, "POST", payload));
  } catch (error) {
    const status = typeof error === "object" && error && "status" in error ? Number((error as { status?: number }).status) : 502;
    const message = error instanceof Error ? error.message : "Request failed";
    return NextResponse.json({ error: message.slice(0, 240) }, { status: status || 502 });
  }
}
