import type { Layer } from "@/data/versions";

export type Snapshot = {
  original: string | null;
  name: string;
  layers: Layer[];
  activeId: string;
  selectedIds: string[];
};

export type ProjectFile = {
  kind: "edit-islands";
  name: string;
  original: string;
  layers: Layer[];
  activeId: string;
  selectedIds: string[];
  minArea: number;
  drafts: Record<string, string>;
  newPrompt: string;
  undo: Snapshot[];
  redo: Snapshot[];
};

async function embed(src: string) {
  if (src.startsWith("data:")) return src;
  const response = await fetch(src);
  if (!response.ok) throw new Error("Could not read a photo for this process.");
  const blob = await response.blob();
  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Could not read a photo for this process."));
    reader.readAsDataURL(blob);
  });
}

async function packLayer(layer: Layer): Promise<Layer> {
  return {
    ...layer,
    edited: await embed(layer.edited),
    islands: await Promise.all(
      layer.islands.map(async (island) => ({ ...island, mask: await embed(island.mask) }))
    ),
  };
}

async function packSnapshot(shot: Snapshot): Promise<Snapshot> {
  return {
    ...shot,
    original: shot.original ? await embed(shot.original) : null,
    layers: await Promise.all(shot.layers.map(packLayer)),
  };
}

export async function packProject(
  input: Omit<ProjectFile, "kind" | "undo" | "redo"> & { undo?: Snapshot[]; redo?: Snapshot[] }
): Promise<ProjectFile> {
  const [layers, undo, redo] = await Promise.all([
    Promise.all(input.layers.map(packLayer)),
    Promise.all((input.undo ?? []).map(packSnapshot)),
    Promise.all((input.redo ?? []).map(packSnapshot)),
  ]);
  return {
    kind: "edit-islands",
    name: input.name,
    original: await embed(input.original),
    layers,
    activeId: input.activeId,
    selectedIds: input.selectedIds,
    minArea: input.minArea,
    drafts: input.drafts,
    newPrompt: input.newPrompt,
    undo,
    redo,
  };
}

export function unpackProject(text: string): ProjectFile {
  let data: ProjectFile;
  try {
    data = JSON.parse(text) as ProjectFile;
  } catch {
    throw new Error("That file is not an Edit Islands process.");
  }
  if (data?.kind !== "edit-islands" || typeof data.original !== "string" || !Array.isArray(data.layers)) {
    throw new Error("That file is not an Edit Islands process.");
  }
  return { ...data, undo: data.undo ?? [], redo: data.redo ?? [] };
}

export function downloadProject(file: ProjectFile) {
  const blob = new Blob([JSON.stringify(file)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${file.name || "Edit Islands"}.json`;
  link.click();
  URL.revokeObjectURL(url);
}
