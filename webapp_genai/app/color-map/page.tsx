"use client";

import { ColorObjects } from "@/components/ColorObjects";
import { Phone } from "@/components/Phone";
import "./color-map.css";

export default function ColorMapPage() {
  return (
    <main className="stage">
      <Phone>
        <ColorObjects />
      </Phone>
    </main>
  );
}
