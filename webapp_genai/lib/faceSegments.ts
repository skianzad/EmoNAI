import type { SegmentObject } from "@/lib/localEdit";

export type SegmentKind = "room" | "face" | "crowd" | "sky";

export type SavedScene = {
  objects: SegmentObject[];
  background: string | null;
};

export async function loadSegments(kind: SegmentKind): Promise<SavedScene | null> {
  const response = await fetch(`/api/segment-cache?kind=${kind}`);
  if (!response.ok) return null;
  const body = await response.json().catch(() => ({ objects: [], background: "" }));
  const objects = Array.isArray(body.objects) ? (body.objects as SegmentObject[]) : [];
  const background =
    typeof body.background === "string" && body.background.length > 0
      ? // Public URLs (/scenes/...) or data URLs both count as a stored empty bg.
        body.background
      : null;
  if (objects.length === 0 && !background) return null;
  return { objects, background };
}

/**
 * Face: persist segments.
 * Room: may create room-segments.json once if missing, and/or save empty background once.
 * Crowd: write crowd-segments.json + empty courtyard background (bake once).
 * Existing room-segments.json is never overwritten by session edits.
 */
export async function saveSegments(
  kind: SegmentKind,
  objects: SegmentObject[],
  background?: string | null,
  options?: { replaceBackground?: boolean }
) {
  const response = await fetch("/api/segment-cache", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      kind,
      objects,
      background: background || undefined,
      replaceBackground: Boolean(options?.replaceBackground),
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

/** Fresh edit state for loaded base room masks. Suggested extras stay hidden until Add. */
export function resetRoomEdits(objects: SegmentObject[]): SegmentObject[] {
  return objects.map((object) => {
    const suggested = Boolean(object.suggested);
    return {
      ...object,
      tint: null,
      shade: 0,
      scale: 100,
      // Suggested segments (e.g. curtain) stay hidden until the Add chip reveals them.
      hidden: suggested,
      suggested,
      dx: 0,
      dy: 0,
      cover: undefined,
      blur: undefined,
      // Keep pre-baked layers (curtain PNG) so Add is local, not Nano Banana.
      layer: object.layer,
    };
  });
}
