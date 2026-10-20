"use client";

import { AiEditApp } from "@/components/ai-edit/AiEditApp";
import { Phone } from "@/components/Phone";

export default function Page() {
  return (
    <main className="stage">
      <Phone>
        <AiEditApp />
      </Phone>
      <p className="caption">
        AI Edit
        <span>HMI Lab</span>
      </p>
    </main>
  );
}
