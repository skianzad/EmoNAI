import { NextResponse } from "next/server";
import { editorChoices, type EditorId } from "@/data/editors";
import { runEditor } from "@/lib/runEdit";

export const maxDuration = 120;

const ids = new Set<string>(editorChoices.map((editor) => editor.id));

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const editor = body?.editor as string;
  const prompt = String(body?.prompt ?? "").trim();
  const image = String(body?.image ?? "");
  const mime = String(body?.mime ?? "image/jpeg");
  const mask = body?.mask ? { data: String(body.mask), mime: String(body.maskMime ?? "image/png") } : undefined;

  if (!ids.has(editor) || !prompt || !image) {
    return NextResponse.json({ error: "Choose an editor, a prompt, and a photo." }, { status: 400 });
  }

  try {
    const result = await runEditor(editor as EditorId, prompt, image, mime, mask);
    return NextResponse.json({ image: result });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Edit failed";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
