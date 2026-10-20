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

/** Persist face segments, or room empty-background only (room ignores objects). */
export async function saveSegments(kind: SegmentKind, objects: SegmentObject[], background?: string | null) {
  const response = await fetch("/api/segment-cache", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      kind,
      // Room disk cache is background-only; do not send segment edits.
      objects: kind === "room" ? [] : objects,
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
