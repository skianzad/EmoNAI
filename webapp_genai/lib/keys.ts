import fs from "fs";
import path from "path";

let cached: Record<string, string> | null = null;

function parseEnvFile(file: string, out: Record<string, string>) {
  if (!fs.existsSync(file)) return;
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) continue;
    const index = trimmed.indexOf("=");
    const name = trimmed.slice(0, index).trim();
    const value = trimmed.slice(index + 1).trim().replace(/^["']|["']$/g, "");
    if (name && value && out[name] === undefined) out[name] = value;
  }
}

function fileKeys(): Record<string, string> {
  if (cached) return cached;
  const root = process.cwd();
  const out: Record<string, string> = {};
  // Prefer standalone / server copies first, then the repo editor env.
  parseEnvFile(path.join(root, ".env.local"), out);
  parseEnvFile(path.join(root, ".env"), out);
  parseEnvFile(path.join(root, "genai_image_editing", ".env"), out);
  cached = out;
  return out;
}

export function key(name: string) {
  return process.env[name] || fileKeys()[name] || "";
}
