import type { SegmentObject } from "@/lib/localEdit";

export type SegmentKind = "room" | "face";

export type SavedScene = {
  objects: SegmentObject[];
  background: string | null;
};

export async function loadSegments(kind: SegmentKind): Promise<SavedScene | null> {
  const response = await fetch(`/api/segment-cache?kind=${kind}`);
  if (!response.ok) return null;
  const body = await response.json().catch(() => ({ objects: [], background: "" }));
  const objects = Array.isArray(body.objects) ? (body.objects as SegmentObject[]) : [];
  const background = typeof body.background === "string" && body.background.length > 0 ? body.background : null;
  if (objects.length === 0 && !background) return null;
  return { objects, background };
}

/**
 * Face: persist segments.
 * Room: may create room-segments.json once if missing, and/or save empty background once.
 * Existing room-segments.json is never overwritten (session edits stay in memory).
 */
export async function saveSegments(kind: SegmentKind, objects: SegmentObject[], background?: string | null) {
  const response = await fetch("/api/segment-cache", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      kind,
      objects,
      background: background || undefined,
    }),
  });
  const body = await response.json().catch(() => ({ error: "Save failed" }));
  if (!response.ok) throw new Error(body.error || "Could not save the scene");
  return body as { ok: boolean; count: number; background: boolean };
}

/** Save only the empty room background (never segment edits). */
export async function saveRoomBackground(background: string) {
  return saveSegments("room", [], background);
}

/** Fresh edit state for loaded base room masks. */
export function resetRoomEdits(objects: SegmentObject[]): SegmentObject[] {
  return objects.map((object) => ({
    ...object,
    tint: null,
    shade: 0,
    scale: 100,
    hidden: false,
    dx: 0,
    dy: 0,
    cover: undefined,
    blur: undefined,
    layer: undefined,
  }));
}
