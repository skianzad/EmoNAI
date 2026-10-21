/**
 * After `npm run build`, assemble a standalone folder ready for the server.
 * Output: ./deploy/  (and deploy.zip if zip/Compress-Archive is available)
 */
import { cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { spawnSync } from "child_process";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const standalone = path.join(root, ".next", "standalone");
const outDir = path.join(root, "deploy");
const outZip = path.join(root, "deploy.zip");

function must(p, label) {
  if (!existsSync(p)) throw new Error(`Missing ${label}: ${p}\nRun: npm run build`);
}

must(path.join(standalone, "server.js"), "standalone server.js");
must(path.join(root, ".next", "static"), ".next/static");
must(path.join(root, "public"), "public");
must(path.join(root, "data"), "data");

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });

cpSync(standalone, outDir, { recursive: true });
mkdirSync(path.join(outDir, ".next"), { recursive: true });
cpSync(path.join(root, ".next", "static"), path.join(outDir, ".next", "static"), { recursive: true });
cpSync(path.join(root, "public"), path.join(outDir, "public"), { recursive: true });
cpSync(path.join(root, "data"), path.join(outDir, "data"), { recursive: true });
cpSync(path.join(root, "ecosystem.config.cjs"), path.join(outDir, "ecosystem.config.cjs"));

mkdirSync(path.join(outDir, "genai_image_editing"), { recursive: true });
const editorEnv = path.join(root, "genai_image_editing", ".env");
const localEnv = path.join(root, ".env.local");
if (existsSync(editorEnv)) {
  cpSync(editorEnv, path.join(outDir, "genai_image_editing", ".env"));
  cpSync(editorEnv, path.join(outDir, ".env.local"));
} else if (existsSync(localEnv)) {
  cpSync(localEnv, path.join(outDir, ".env.local"));
  cpSync(localEnv, path.join(outDir, "genai_image_editing", ".env"));
} else {
  console.warn("WARNING: no API key file — add genai_image_editing/.env or .env.local before packing.");
}

writeFileSync(
  path.join(outDir, "README.txt"),
  [
    "On the server:",
    "  cd deploy   (or whatever you named this folder)",
    "  pm2 start ecosystem.config.cjs",
    "  pm2 save",
    "",
    "Needs on server: Node.js, pm2, and",
    "  ~/HuaweiWebSecureInternetGatewayCAV2_soheil.pem",
    "Proxy settings are already in ecosystem.config.cjs.",
    "",
  ].join("\n")
);

rmSync(outZip, { force: true });
const zipped =
  process.platform === "win32"
    ? spawnSync(
        "powershell",
        [
          "-NoProfile",
          "-Command",
          `Compress-Archive -Path '${outDir.replace(/'/g, "''")}\\*' -DestinationPath '${outZip.replace(/'/g, "''")}' -Force`,
        ],
        { stdio: "inherit" }
      )
    : spawnSync("zip", ["-r", "-q", outZip, "."], { cwd: outDir, stdio: "inherit" });

if (zipped.status === 0) {
  console.log("Ready:", outDir);
  console.log("Zip:  ", outZip);
} else {
  console.log("Ready folder (zip skipped):", outDir);
}
