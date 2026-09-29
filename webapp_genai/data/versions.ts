import type { Island } from "./edits";

export type Layer = {
  id: string;
  parentId: string | null;
  prompt: string;
  visible: boolean;
  edited: string;
  islands: Island[];
};

export type VersionNode = {
  version: Layer;
  label: string;
  depth: number;
};

export function versionTree(layers: Layer[]): VersionNode[] {
  const nodes: VersionNode[] = [];

  function walk(version: Layer, label: string, depth: number) {
    nodes.push({ version, label, depth });
    layers
      .filter((item) => item.parentId === version.id)
      .forEach((child, index) => walk(child, `${label}.${index + 1}`, depth + 1));
  }

  layers
    .filter((item) => item.parentId === null)
    .forEach((root, index) => walk(root, `v${index + 1}`, 0));

  return nodes;
}

/**
 * Photo for the open prompt: every top-level prompt, plus this prompt if it is nested.
 * A subprompt stays off the main prompt's photo until that subprompt is opened.
 */
export function shownLayers(layers: Layer[], activeId: string) {
  const open = new Set<string>();
  let current: string | null = activeId;
  let steps = 0;
  while (current && steps <= layers.length) {
    open.add(current);
    current = layers.find((layer) => layer.id === current)?.parentId ?? null;
    steps += 1;
  }
  return layers.filter((layer) => layer.parentId === null || open.has(layer.id));
}

/** This version plus its parents, oldest first, in the order they paint. */
export function branchEndingAt(layers: Layer[], id: string) {
  const chain = new Set<string>();
  let current: string | null = id;
  let steps = 0;
  while (current && steps <= layers.length) {
    chain.add(current);
    current = layers.find((layer) => layer.id === current)?.parentId ?? null;
    steps += 1;
  }
  return layers.filter((layer) => chain.has(layer.id));
}

export function insertAfterBranch(layers: Layer[], parentId: string, child: Layer) {
  const descendants = new Set<string>();
  const stack = layers.filter((layer) => layer.parentId === parentId).map((layer) => layer.id);
  while (stack.length) {
    const next = stack.pop() as string;
    if (descendants.has(next)) continue;
    descendants.add(next);
    for (const layer of layers) {
      if (layer.parentId === next) stack.push(layer.id);
    }
  }

  let last = layers.findIndex((layer) => layer.id === parentId);
  layers.forEach((layer, index) => {
    if (descendants.has(layer.id)) last = Math.max(last, index);
  });
  const next = layers.slice();
  next.splice(Math.max(last, 0) + 1, 0, child);
  return next;
}

/** This version and every version that hangs under it. */
export function removedIds(layers: Layer[], id: string) {
  const remove = new Set<string>([id]);
  let grew = true;
  while (grew) {
    grew = false;
    for (const layer of layers) {
      if (layer.parentId && remove.has(layer.parentId) && !remove.has(layer.id)) {
        remove.add(layer.id);
        grew = true;
      }
    }
  }
  return remove;
}

/** Where a dragged prompt lands relative to another prompt. */
export type DropPlace = "before" | "after" | "inside";

/** Move a prompt, including under a different prompt or out to the top. */
export function moveVersion(layers: Layer[], dragId: string, overId: string, place: DropPlace) {
  const movingIds = removedIds(layers, dragId);
  if (movingIds.has(overId)) return layers;
  const over = layers.find((layer) => layer.id === overId);
  if (!over) return layers;

  const parentId = place === "inside" ? overId : over.parentId;
  const block = layers
    .filter((layer) => movingIds.has(layer.id))
    .map((layer) => (layer.id === dragId ? { ...layer, parentId } : layer));
  const rest = layers.filter((layer) => !movingIds.has(layer.id));
  const overIndex = rest.findIndex((layer) => layer.id === overId);
  if (overIndex < 0) return layers;

  let insertAt = overIndex;
  if (place === "before" || place === "inside") {
    const overIds = removedIds(rest, overId);
    let end = overIndex;
    rest.forEach((layer, index) => {
      if (overIds.has(layer.id)) end = Math.max(end, index);
    });
    insertAt = end + 1;
  }

  const next = rest.slice();
  next.splice(insertAt, 0, ...block);
  return next;
}
