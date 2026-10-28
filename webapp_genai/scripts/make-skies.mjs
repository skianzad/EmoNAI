import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { dirname } from "path";

const env = Object.fromEntries(
  readFileSync("webapp_genai/genai_image_editing/.env", "utf8")
    .split("\n")
    .filter((line) => line && !line.startsWith("#") && line.includes("="))
    .map((line) => {
      const i = line.indexOf("=");
      return [line.slice(0, i).trim(), line.slice(i + 1).trim().replace(/^["']|["']$/g, "")];
    })
);
const key = env.GEMINI_API_KEY;
const model = env.GEMINI_IMAGE_MODEL || "gemini-3.1-flash-image";

const jobs = [
  {
    file: "webapp_genai/public/scenes/desert/original.jpg",
    prompt:
      "Photorealistic wide landscape photograph of a desert like Wadi Rum: a large sandstone butte in the center, orange sand in the foreground, a tiny camel walking left of center, pale creamy sky with almost no clouds, golden hour light, no people, no text, no watermark, no UI. Landscape orientation 16:9.",
  },
  {
    file: "webapp_genai/public/scenes/desert/skies/golden-a.jpg",
    prompt: "Photograph of only sky, no ground, no horizon, no mountains. Warm golden hour clouds, glowing orange and peach, cinematic, 16:9.",
  },
  {
    file: "webapp_genai/public/scenes/desert/skies/golden-b.jpg",
    prompt: "Photograph of only sky, no ground. Dramatic golden sunset clouds with sun rays, warm orange, 16:9.",
  },
  {
    file: "webapp_genai/public/scenes/desert/skies/dawn-a.jpg",
    prompt: "Photograph of only sky, no ground. Soft dawn pink and lavender gradient with thin clouds, 16:9.",
  },
  {
    file: "webapp_genai/public/scenes/desert/skies/dawn-b.jpg",
    prompt: "Photograph of only sky, no ground. Sunrise orange band over pale blue, wispy clouds, 16:9.",
  },
  {
    file: "webapp_genai/public/scenes/desert/skies/blue.jpg",
    prompt: "Photograph of only sky, no ground. Clear deep blue sky with scattered white cumulus clouds, 16:9.",
  },
  {
    file: "webapp_genai/public/scenes/desert/skies/storm.jpg",
    prompt: "Photograph of only sky, no ground. Dark storm clouds, moody grey and purple, 16:9.",
  },
  {
    file: "webapp_genai/public/scenes/desert/skies/overcast.jpg",
    prompt: "Photograph of only sky, no ground. Soft overcast grey-white clouds filling the frame, 16:9.",
  },
  {
    file: "webapp_genai/public/scenes/desert/skies/night.jpg",
    prompt: "Photograph of only sky, no ground. Deep night sky with stars and a faint Milky Way, 16:9.",
  },
];

async function one(job) {
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: job.prompt }] }],
      generationConfig: { responseModalities: ["IMAGE"], temperature: 0.7 },
    }),
  });
  const json = await res.json();
  if (!res.ok) throw new Error(`${job.file} ${res.status} ${(json.error?.message || "").slice(0, 180)}`);
  const parts = json.candidates?.[0]?.content?.parts || [];
  const image = parts.find((part) => part.inlineData?.data || part.inline_data?.data);
  const data = image?.inlineData?.data || image?.inline_data?.data;
  if (!data) throw new Error("no image " + job.file);
  mkdirSync(dirname(job.file), { recursive: true });
  writeFileSync(job.file, Buffer.from(data, "base64"));
  return job.file.split("/").pop();
}

const results = await Promise.allSettled(jobs.map(one));
for (const result of results) {
  console.log(result.status === "fulfilled" ? "saved " + result.value : "fail " + result.reason.message);
}
