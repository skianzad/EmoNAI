"use client";

import { useState, type FormEvent } from "react";
import { useLang } from "@/lib/i18n";

export function Gate() {
  const { t } = useLang();
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/enter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (!response.ok) {
        setError(t.wrongPassword);
        return;
      }
      window.location.href = "/";
    } catch {
      setError(t.passwordCheckFailed);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="gate" onSubmit={submit}>
      <h1>{t.title}</h1>
      <p className="subtitle">{t.lab}</p>
      <input
        type="password"
        autoFocus
        autoComplete="current-password"
        aria-label={t.password}
        placeholder={t.password}
        value={password}
        onChange={(event) => setPassword(event.target.value)}
      />
      <button type="submit" disabled={busy || password === ""}>
        {t.enter}
      </button>
      {error ? <p className="error">{error}</p> : null}
    </form>
  );
}
