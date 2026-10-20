import type { SegmentObject } from "@/lib/localEdit";

export type SegmentKind = "room" | "face";

export async function loadSegments(kind: SegmentKind) {
  const response = await fetch(`/api/segment-cache?kind=${kind}`);
  const body = await response.json().catch(() => ({ objects: [] }));
  const objects = Array.isArray(body.objects) ? (body.objects as SegmentObject[]) : [];
  return objects.length > 0 ? objects : null;
}

export function saveSegments(kind: SegmentKind, objects: SegmentObject[]) {
  void fetch("/api/segment-cache", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind, objects }),
  });
}
