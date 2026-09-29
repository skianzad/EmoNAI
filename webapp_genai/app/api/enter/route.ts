import { NextResponse } from "next/server";
import { GATE_COOKIE, GATE_TOKEN, PAGE_PASSWORD } from "@/lib/gate";

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as { password?: string } | null;
  if (body?.password !== PAGE_PASSWORD) {
    return NextResponse.json({ error: "Wrong password." }, { status: 401 });
  }
  const response = NextResponse.json({ ok: true });
  response.cookies.set(GATE_COOKIE, GATE_TOKEN, {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
  });
  return response;
}
