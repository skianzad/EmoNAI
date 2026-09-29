"use client";

import { useEffect, useState, type ReactNode } from "react";

function useClock() {
  const [time, setTime] = useState("");

  useEffect(() => {
    const tick = () => {
      setTime(
        new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })
      );
    };
    tick();
    const id = window.setInterval(tick, 10_000);
    return () => window.clearInterval(id);
  }, []);

  return time;
}

export function Phone({ children }: { children: ReactNode }) {
  const time = useClock();

  return (
    <div className="phone">
      <span className="key key-volume-up" />
      <span className="key key-volume-down" />
      <span className="key key-power" />
      <div className="screen">
        <div className="camera" />
        <div className="status">
          <span>{time}</span>
          <span className="status-icons" aria-hidden="true">
            <Signal />
            <Battery />
          </span>
        </div>
        {children}
        <div className="gesture" />
      </div>
      <div className="brand">HUAWEI</div>
    </div>
  );
}

function Signal() {
  return (
    <svg width="17" height="12" viewBox="0 0 17 12" fill="currentColor">
      <rect x="0" y="8" width="3" height="4" rx="0.5" />
      <rect x="4.5" y="5" width="3" height="7" rx="0.5" />
      <rect x="9" y="2.5" width="3" height="9.5" rx="0.5" />
      <rect x="13.5" y="0" width="3" height="12" rx="0.5" />
    </svg>
  );
}

function Battery() {
  return (
    <svg width="25" height="12" viewBox="0 0 25 12" fill="currentColor">
      <rect x="0.6" y="0.6" width="21" height="10.8" rx="2.2" fill="none" stroke="currentColor" strokeWidth="1.2" />
      <rect x="2.2" y="2.2" width="16" height="7.6" rx="1" />
      <rect x="22.4" y="3.6" width="1.8" height="4.8" rx="0.6" />
    </svg>
  );
}
