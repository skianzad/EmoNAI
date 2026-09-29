import fs from "fs";
import path from "path";

let cached: Record<string, string> | null = null;

function fileKeys(): Record<string, string> {
  if (cached) return cached;
  const file = path.join(process.cwd(), "..", "genai_image_editing", ".env");
  const out: Record<string, string> = {};
  if (fs.existsSync(file)) {
    for (const line of fs.readFileSync(file, "utf8").split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue;
      const index = trimmed.indexOf("=");
      const name = trimmed.slice(0, index).trim();
      const value = trimmed.slice(index + 1).trim();
      if (value) out[name] = value;
    }
  }
  cached = out;
  return out;
}

export function key(name: string) {
  return process.env[name] || fileKeys()[name] || "";
}
