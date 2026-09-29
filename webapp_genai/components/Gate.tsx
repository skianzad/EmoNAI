"use client";

import { useState, type FormEvent } from "react";

export function Gate() {
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
        setError("Wrong password.");
        return;
      }
      window.location.href = "/";
    } catch {
      setError("Could not check the password.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="gate" onSubmit={submit}>
      <h1>GenAI Image Editor</h1>
      <p className="subtitle">HMI Lab</p>
      <input
        type="password"
        autoFocus
        autoComplete="current-password"
        aria-label="Password"
        placeholder="Password"
        value={password}
        onChange={(event) => setPassword(event.target.value)}
      />
      <button type="submit" disabled={busy || password === ""}>
        Enter
      </button>
      {error ? <p className="error">{error}</p> : null}
    </form>
  );
}
