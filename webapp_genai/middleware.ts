import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { GATE_COOKIE, GATE_TOKEN } from "@/lib/gate";

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  if (pathname === "/enter" || pathname === "/api/enter" || pathname.startsWith("/_next")) {
    return NextResponse.next();
  }
  if (request.cookies.get(GATE_COOKIE)?.value === GATE_TOKEN) {
    return NextResponse.next();
  }
  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "Enter the password first." }, { status: 401 });
  }
  const url = request.nextUrl.clone();
  url.pathname = "/enter";
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
