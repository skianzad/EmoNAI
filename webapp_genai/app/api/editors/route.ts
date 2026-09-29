import { NextResponse } from "next/server";
import { editorStatus } from "@/lib/runEdit";

export function GET() {
  return NextResponse.json(editorStatus());
}
