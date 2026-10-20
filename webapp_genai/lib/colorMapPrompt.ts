/** Colors we ask the model to use. They are rare in natural photos, so each object stays easy to tell apart. */
export const COLOR_MAP_PALETTE = [
  "#FF00FF",
  "#00FF00",
  "#00FFFF",
  "#FF8000",
  "#4040FF",
  "#FFFF00",
  "#FF1493",
  "#7CFC00",
] as const;

export const COLOR_MAP_PROMPT = [
  "Look at this photo and choose each important separate object (people, bags, phones, furniture, cups, signs).",
  "Leave the background, walls, floor, and sky exactly as they are.",
  "Repaint each chosen object as one flat solid color. One color per object.",
  `Use only colors from this list that do not already appear in the photo: ${COLOR_MAP_PALETTE.join(", ")}.`,
  "Do not change shapes, positions, lighting of untouched areas, or the frame. Do not add or remove objects.",
  'After the image, reply with JSON only, no markdown: [{"name":"bag","color":"#FF0023"}]',
  "Use a short name and the exact hex you painted, like bag (#FF0023).",
].join("\n");

export const ROOM_COLOR_MAP_PROMPT = [
  "Look at this living room and repaint only these objects as flat solid colors. Fill each one completely. One color per object.",
  "Objects: the cream sofa, the whole left leather armchair including its wooden arms and legs, the whole right leather armchair including its wooden arms and legs, the hanging plant on the left, the leafy plant on the right, the round wooden coffee table in the center, and the arc floor lamp with the pleated shade behind the left chair.",
  "Do not paint any walls, the floor, the rug, the windows, the bookshelves, the books, the paintings, or the ladder.",
  `Use only colors from this list that do not already appear in the photo: ${COLOR_MAP_PALETTE.join(", ")}.`,
  "Do not move anything, change the frame, or add objects.",
  'After the image, reply with JSON only, no markdown: [{"name":"sofa","color":"#00FFFF"},{"name":"table","color":"#4040FF"}]',
  "Use these short names: sofa, left chair, right chair, left plant, right plant, table, lamp. Include the exact hex you painted.",
].join("\n");

export function guidePrompt(regions: { name: string; left: number; top: number; width: number; height: number }[]) {
  const lines = regions.map((region) => {
    const across = region.left + region.width / 2;
    const down = region.top + region.height / 2;
    const side = across < 0.34 ? "on the left" : across > 0.66 ? "on the right" : "in the center";
    const band = down < 0.34 ? "near the top" : down > 0.66 ? "near the bottom" : "in the middle";
    return `${region.name}, ${side} ${band}`;
  });
  return [
    "Repaint only the regions listed below. The user has moved, resized, or removed boxes, so follow this list and these positions. Fill each region completely with one flat solid color.",
    ...lines.map((line) => `- ${line}`),
    "Do not paint anything that is not in this list.",
    `Use only colors from this list that do not already appear in the photo: ${COLOR_MAP_PALETTE.join(", ")}.`,
    "Do not move objects or change the frame.",
    `After the image, reply with JSON only, no markdown: ${JSON.stringify(regions.map((region) => ({ name: region.name, color: "#00FF00" })))}`,
    "Use these exact names and the exact hex you painted.",
  ].join("\n");
}

/** Extra rare colors so we can mark many background people separately. */
export const CROWD_COLOR_MAP_PALETTE = [
  ...COLOR_MAP_PALETTE,
  "#FF0040",
  "#40FFC0",
  "#C040FF",
  "#FFC040",
  "#4080FF",
  "#80FF40",
  "#FF40A0",
  "#40A0FF",
] as const;

/**
 * Louvre courtyard selfie: keep the couple as subjects, paint each background
 * person (or tight group) as its own flat color so the crowd slider can hide them.
 */
export const CROWD_COLOR_MAP_PROMPT = [
  "Look at this Louvre courtyard selfie.",
  "Repaint only these as flat solid colors, one color each, filling each person completely:",
  "- the man in the foreground (left, sunglasses, hooded jacket)",
  "- the woman in the foreground (right, peace signs, tan jacket)",
  "- every separate background tourist / person (or tight pair standing together) in the courtyard behind them — name them person 1, person 2, person 3, … from nearest to farthest",
  "Do not paint the Louvre pyramid, palace walls, paving, sky, bags on the couple, or anything that is not a person.",
  `Use only colors from this list that do not already appear in the photo: ${CROWD_COLOR_MAP_PALETTE.join(", ")}.`,
  "Do not move anyone, change the frame, or invent people.",
  'After the image, reply with JSON only, no markdown: [{"name":"man","color":"#FF00FF"},{"name":"woman","color":"#00FF00"},{"name":"person 1","color":"#00FFFF"}]',
  "Use short names: man, woman, person 1, person 2, … Include the exact hex you painted.",
].join("\n");

export const FACE_COLOR_MAP_PROMPT = [
  "Look at this portrait and repaint only these face regions as flat solid colors. One color per region.",
  "Regions: both under-eye shadows together, the lips, the left cheek, the right cheek, the forehead, and the background behind the person.",
  "Leave the eyes, eyebrows, nose, hair, teeth, and clothes exactly as they are.",
  `Use only colors from this list that do not already appear in the photo: ${COLOR_MAP_PALETTE.join(", ")}.`,
  "Do not change face shape, expression, or the frame. Do not add makeup that is not a flat color fill.",
  'After the image, reply with JSON only, no markdown: [{"name":"under eyes","color":"#FF00FF"},{"name":"lips","color":"#00FF00"}]',
  "Use these short names: under eyes, lips, left cheek, right cheek, forehead, background. Include the exact hex you painted.",
].join("\n");
