import { GIFEncoder, quantize, applyPalette } from "gifenc";

const sheet = document.getElementById("sheet");
const screen = document.querySelector(".screen");
const phone = document.getElementById("phone");
const photo = document.getElementById("photo");
const ballEl = document.getElementById("ball");
const reactEl = document.getElementById("react");

const PHOTO_W = 390;
const PHOTO_H = 520;
const BALL = 36;
const NEAR = 156;
let heads = [
  { id: "left", name: "Left", x: 146, y: 128, r: 36, line: "Heads it back" },
  { id: "right", name: "Right", x: 252, y: 124, r: 28, line: "Heads it back" },
];
const POSES = {
  base: "/friends.png",
  left: "/friends-left.png",
  right: "/friends-right.png",
  flinch: "/friends-flinch.png",
  catchLeft: "/friends-catch-left.png",
  catchRight: "/friends-catch-right.png",
  heartLeft: "/friends-heart-left.png",
  heartRight: "/friends-heart-right.png",
};
function bananaSvg(flex = 0) {
  const dip = 82 + flex * 16;
  return `<svg viewBox="0 0 160 100" aria-hidden="true"><path d="M26 58 Q 78 ${dip}, 134 32" fill="none" stroke="#f0b400" stroke-width="30" stroke-linecap="round"/><path d="M34 54 Q 76 ${dip - 14}, 120 40" fill="none" stroke="#ffe56a" stroke-width="8" stroke-linecap="round"/><path d="M132 30 Q 148 14, 138 28" fill="none" stroke="#6a3e1b" stroke-width="6" stroke-linecap="round"/></svg>`;
}
const HEART_SVG = `<svg viewBox="0 0 128 120" aria-hidden="true"><defs><radialGradient id="heartBody" cx="36%" cy="30%" r="78%"><stop offset="0%" stop-color="#ffe4ea"/><stop offset="22%" stop-color="#ff5d7a"/><stop offset="58%" stop-color="#e11d48"/><stop offset="100%" stop-color="#6b1230"/></radialGradient><linearGradient id="heartRim" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#fff" stop-opacity=".55"/><stop offset="42%" stop-color="#fff" stop-opacity="0"/><stop offset="100%" stop-color="#4c0519" stop-opacity=".45"/></linearGradient></defs><ellipse cx="64" cy="110" rx="34" ry="6" fill="rgba(80,10,24,.28)"/><path fill="url(#heartBody)" d="M64 100C64 100 16 68 16 42 16 24 32 12 48 22c8 5 13 13 16 20 3-7 8-15 16-20 16-10 32 2 32 20 0 26-48 58-48 58z"/><path fill="url(#heartRim)" d="M64 100C64 100 16 68 16 42 16 24 32 12 48 22c8 5 13 13 16 20 3-7 8-15 16-20 16-10 32 2 32 20 0 26-48 58-48 58z"/><ellipse cx="42" cy="40" rx="11" ry="6" fill="#fff" opacity=".72" transform="rotate(-32 42 40)"/><ellipse cx="78" cy="46" rx="6" ry="3.5" fill="#fff" opacity=".35" transform="rotate(-20 78 46)"/></svg>`;
const FLOWER_SVG = `<svg viewBox="0 0 140 190" aria-hidden="true"><defs><linearGradient id="stemShade" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#14532d"/><stop offset="48%" stop-color="#4ade80"/><stop offset="100%" stop-color="#166534"/></linearGradient><radialGradient id="petalBack" cx="38%" cy="28%" r="75%"><stop offset="0%" stop-color="#fda4af"/><stop offset="100%" stop-color="#881337"/></radialGradient><radialGradient id="petalFront" cx="34%" cy="28%" r="78%"><stop offset="0%" stop-color="#fff1f2"/><stop offset="38%" stop-color="#fb7185"/><stop offset="100%" stop-color="#9f1239"/></radialGradient><radialGradient id="leafShade" cx="32%" cy="30%" r="80%"><stop offset="0%" stop-color="#bbf7d0"/><stop offset="100%" stop-color="#166534"/></radialGradient><radialGradient id="roseCore" cx="40%" cy="34%" r="70%"><stop offset="0%" stop-color="#fff7ed"/><stop offset="46%" stop-color="#e11d48"/><stop offset="100%" stop-color="#4c0519"/></radialGradient></defs><ellipse cx="70" cy="180" rx="28" ry="7" fill="rgba(0,0,0,.22)"/><path d="M70 168C66 142 76 114 70 88" fill="none" stroke="url(#stemShade)" stroke-width="9" stroke-linecap="round"/><path d="M70 132C46 120 24 104 18 116c14-2 34 10 52 18z" fill="url(#leafShade)"/><path d="M70 118C98 104 122 92 128 106c-16 0-38 8-58 16z" fill="url(#leafShade)"/><g transform="translate(70 74)"><ellipse rx="20" ry="32" cy="-8" fill="url(#petalBack)" transform="rotate(0)"/><ellipse rx="20" ry="32" cy="-8" fill="url(#petalBack)" transform="rotate(72)"/><ellipse rx="20" ry="32" cy="-8" fill="url(#petalBack)" transform="rotate(144)"/><ellipse rx="16" ry="26" cy="-6" fill="url(#petalFront)" transform="rotate(36)"/><ellipse rx="16" ry="26" cy="-6" fill="url(#petalFront)" transform="rotate(108)"/><ellipse rx="16" ry="26" cy="-6" fill="url(#petalFront)" transform="rotate(180)"/><ellipse rx="11" ry="17" cy="-3" fill="url(#petalFront)" transform="rotate(18)"/><circle r="10" fill="url(#roseCore)"/><ellipse cx="-3" cy="-4" rx="4.5" ry="2.4" fill="#fff" opacity=".75" transform="rotate(-28)"/></g></svg>`;
const BOMB_SVG = `<svg viewBox="0 0 64 64" aria-hidden="true"><circle cx="32" cy="40" r="18" fill="#141414"/><circle cx="26" cy="34" r="4" fill="#2c2c2c"/><rect x="29" y="18" width="6" height="8" rx="1.5" fill="#3a3a3a"/><path d="M32 20c10-14 20-6 13 3" fill="none" stroke="#7a4a28" stroke-width="2.5" stroke-linecap="round"/><path d="M45 12l5-6M49 14l7-1M47 18l6 4" stroke="#ffb703" stroke-width="2" stroke-linecap="round"/></svg>`;
const GEARS = {
  ball: { id: "ball", label: "Ball", emoji: "⚽", w: 56, h: 56, near: 156, kind: "bounce" },
  // banana: { id: "banana", label: "Banana", w: 128, h: 80, near: 156, kind: "soft", svg: bananaSvg(0) },
  heart: { id: "heart", label: "Heart", emoji: "❤️", w: 56, h: 56, near: 148, kind: "soft" },
  flower: { id: "flower", label: "Flower", emoji: "🌸", w: 56, h: 56, near: 140, kind: "soft" },
  bomb: { id: "bomb", label: "Bomb", emoji: "💣", w: 56, h: 56, near: 128, kind: "bomb" },
};
let gearId = "ball";
let customGear = null;
const GROUND = 452;
const poseImages = {};
for (const [id, src] of Object.entries(POSES)) {
  const image = new Image();
  image.src = src;
  poseImages[id] = image;
}
const ballImg = new Image();
ballImg.src = "/soccer-ball.png";
const sprites = { ball: ballImg, custom: new Image() };
let pose = "base";
let poseUntil = 0;

const ball = { x: 40, y: 36, vx: 0, vy: 0, z: 1, spin: 0, fade: 1, tripped: false, rollT: 0, rollFrom: null, held: false, live: false, flying: false, returning: false, flight: null };
let custom = null;
const tape = [];
const hits = [];
let headers = 0;
let drag = null;
let mode = "open";
let clock = 0;
let last = performance.now();
let sampleAt = 0;
let replayAt = 0;

function fit() {
  const box = screen.getBoundingClientRect();
  const scale = Math.min(box.width / 390, box.height / 844);
  screen.style.setProperty("--s", String(scale || 1));
}
fit();
new ResizeObserver(fit).observe(screen);

const shots = [{ id: "friends", label: "Friends" }];
let shotId = "friends";

document.getElementById("play").onclick = animate;
document.getElementById("thumbs").onclick = (event) => {
  const thumb = event.target.closest(".thumb");
  if (thumb) selectShot(thumb.dataset.id);
};
document.getElementById("replay").onclick = startReplay;
document.getElementById("save-gif").onclick = saveGif;
document.getElementById("gear").onclick = (event) => {
  const button = event.target.closest("[data-gear]");
  if (!button || ball.live) return;
  gearId = button.dataset.gear;
  document.getElementById("emoji-list").hidden = true;
  paintGear();
  setHint();
  if (mode === "play") parkBall();
};
document.getElementById("emoji-list").onclick = (event) => {
  const button = event.target.closest("[data-emoji]");
  if (!button || ball.live) return;
  selectEmoji(button.dataset.emoji);
  document.getElementById("emoji-list").hidden = true;
};
document.getElementById("gear-add").onclick = (event) => {
  event.stopPropagation();
  if (ball.live) return;
  const list = document.getElementById("emoji-list");
  list.hidden = !list.hidden;
};
paintGear();
document.getElementById("load").onclick = () => document.getElementById("file").click();
document.getElementById("file").onchange = (event) => {
  const file = event.target.files?.[0];
  if (file) loadPhoto(file);
  event.target.value = "";
};
document.getElementById("make").onclick = createReactions;
document.getElementById("back").onclick = () => {
  if (mode === "replay") startPlay(false);
  else if (mode === "play") showOpen();
};

photo.addEventListener("pointerdown", onDown);
ballEl.addEventListener("pointerdown", onDown);
photo.addEventListener("pointermove", onMove);
ballEl.addEventListener("pointermove", onMove);
window.addEventListener("pointerup", onUp);

requestAnimationFrame(tick);

function showOpen() {
  mode = "open";
  sheet.dataset.mode = "open";
  ball.live = false;
  ball.held = false;
  ballEl.hidden = true;
  reactEl.hidden = true;
  document.getElementById("play-bar").hidden = true;
  document.getElementById("clip-actions").hidden = true;
  document.getElementById("bar-title").textContent = "October 22";
  document.getElementById("bar-sub").textContent = "";
  setPose("base");
  hideImpact();
}

function startPlay(reset = true) {
  mode = "play";
  sheet.dataset.mode = "play";
  document.getElementById("bar-title").textContent = "Animate";
  document.getElementById("bar-sub").textContent = "From you";
  document.getElementById("play-bar").hidden = false;
  ballEl.hidden = false;
  reactEl.hidden = true;
  if (!reset) {
    ball.live = tape.length > 0;
    placeBall();
    return;
  }
  document.getElementById("clip-actions").hidden = true;
  tape.length = 0;
  hits.length = 0;
  headers = 0;
  clock = 0;
  sampleAt = 0;
  poseUntil = 0;
  setHint();
  setPose("base");
  hideImpact();
  parkBall();
}

function onDown(event) {
  const local = photoPoint(event);
  if (mode === "open" && custom && event.target !== ballEl) {
    placeAim(local);
    return;
  }
  if (mode !== "play" || ballEl.hidden) return;
  const center = visualCenter();
  if (event.target !== ballEl && Math.hypot(local.x - center.x, local.y - center.y) > visualSize() / 2 + 36) return;
  ball.held = true;
  ball.live = false;
  ball.flying = false;
  ball.vx = 0;
  ball.vy = 0;
  ballEl.classList.add("selected");
  drag = { x: local.x, y: local.y, t: performance.now(), vx: 0, vy: 0 };
  try { ballEl.setPointerCapture(event.pointerId); } catch { /* synthetic pointers have no capture */ }
}

function onMove(event) {
  if (!drag) return;
  event.stopPropagation();
  const local = photoPoint(event);
  const now = performance.now();
  const dt = Math.max(16, now - drag.t) / 1000;
  drag.vx = (local.x - drag.x) / dt;
  drag.vy = (local.y - drag.y) / dt;
  drag.x = local.x;
  drag.y = local.y;
  drag.t = now;
  ball.x = clamp(local.x - gw() / 2, 8, PHOTO_W - gw() - 8);
  ball.y = clamp(local.y - gh() / 2, 8, PHOTO_H - gh() - 8);
  placeBall();
}

function onUp() {
  if (!drag) return;
  const speed = Math.hypot(drag.vx, drag.vy);
  const vx = drag.vx;
  const vy = drag.vy;
  ball.held = false;
  drag = null;
  if (speed < 140) {
    ball.live = false;
    ball.flying = false;
    placeBall();
    return;
  }
  if (gear().kind === "soft") {
    ball.live = true;
    ball.flying = false;
    ball.flight = null;
    ball.fade = 1;
    ball.flex = 0;
    ball.tripped = false;
    ball.rollT = 0;
    ball.spin = 0;
    ball.rollFrom = { x: ball.x, y: ball.y };
    ball.pitch = 0;
    ball.yaw = 0;
    ball.caught = false;
    if (gear().id === "heart" || gear().id === "flower") {
      ball.catchId = catcherFor(vx, vy).id;
      setPose("base");
    }
    ballEl.hidden = false;
    ballEl.style.opacity = "1";
    ballEl.classList.remove("selected");
    return;
  }
  const aim = landingPoint(vx, vy);
  headers = 0;
  for (const head of heads) head.touching = false;
  ball.live = true;
  ball.flying = true;
  ball.flight = {
    fromX: ball.x,
    fromY: ball.y,
    toX: aim.x - gw() / 2,
    toY: aim.y - gh() / 2,
    vx,
    vy,
  };
  ballEl.classList.remove("selected");
}

function tick(now) {
  const dt = Math.min(0.032, (now - last) / 1000);
  last = now;
  if (mode === "play" && ball.live && !ball.held) step(dt);
  else if (mode === "replay") replay(dt);
  else if (mode === "play" && ball.caught && gear().id === "heart" && !ball.held) {
    clock += dt;
    ball.flex = Math.pow(Math.sin(clock * 3.2), 2);
    placeBall();
  } else if (mode === "play" && poseUntil && clock < poseUntil + 2) {
    clock += dt;
    if (clock >= poseUntil) setPose("base");
  }
  requestAnimationFrame(tick);
}

function step(dt) {
  if (ball.flying) {
    fly(dt);
    clock += dt;
    if (poseUntil && clock >= poseUntil) setPose("base");
    sample();
    placeBall();
    return;
  }
  if (gear().kind === "soft") {
    glide(dt);
    return;
  }
  ball.z = 0;
  ball.x += ball.vx * dt;
  ball.y += ball.vy * dt;

  if (ball.x <= 0) {
    ball.x = 0;
    ball.vx = Math.abs(ball.vx);
  } else if (ball.x >= PHOTO_W - gw()) {
    ball.x = PHOTO_W - gw();
    ball.vx = -Math.abs(ball.vx);
  }
  if (ball.y <= 0) {
    ball.y = 0;
    ball.vy = Math.abs(ball.vy);
  } else if (ball.y >= PHOTO_H - gh()) {
    ball.y = PHOTO_H - gh();
    ball.vy = -Math.abs(ball.vy);
  }

  const speed = Math.hypot(ball.vx, ball.vy);
  if (gear().kind === "bounce" && speed < 260) {
    const scale = 260 / Math.max(speed, 1);
    ball.vx *= scale;
    ball.vy *= scale;
  }

  const cx = ball.x + gw() / 2;
  const cy = ball.y + gh() / 2;
  for (const head of heads) {
    const dx = cx - head.x;
    const dy = cy - head.y;
    const dist = Math.hypot(dx, dy) || 0.001;
    const min = Math.max(gw(), gh()) / 2 + head.r;
    if (dist >= min) {
      head.touching = false;
      continue;
    }
    const nx = dx / dist;
    const ny = dy / dist;
    const overlap = min - dist;
    ball.x += nx * overlap;
    ball.y += ny * overlap;
    const vn = ball.vx * nx + ball.vy * ny;
    if (vn < 0) {
      ball.vx -= 2 * vn * nx;
      ball.vy -= 2 * vn * ny;
    }
    if (!head.touching) {
      head.touching = true;
      header(head);
    }
  }
  clock += dt;
  const near = nearestHead();
  if (gear().kind === "bounce" && near) {
    setPose(near.id);
    poseUntil = Math.max(poseUntil, clock + 0.35);
  } else if (gear().kind === "bomb" && near) {
    setPose("flinch");
    poseUntil = Math.max(poseUntil, clock + 0.4);
  } else if (poseUntil && clock >= poseUntil) {
    setPose("base");
  }
  if (clock >= sampleAt) {
    tape.push({ t: clock, x: ball.x, y: ball.y, z: ball.z, spin: ball.spin, fade: ball.fade ?? 1, flex: ball.flex || 0, pitch: ball.pitch || 0, yaw: ball.yaw || 0, gear: gearId, pose });
    sampleAt = clock + 0.05;
  }
  placeBall();
}

function header(head) {
  if (gear().kind === "bomb") {
    explode();
    return;
  }
  if (gear().kind === "soft" || gear().kind === "emoji") return;
  hits.push({ t: clock, name: head.name, id: head.id, line: head.line });
  document.getElementById("clip-actions").hidden = false;
  setPose(head.id);
  poseUntil = clock + 1.2;
  bonk();
  shakePhone();
  headers += 1;
  if (headers >= 3) {
    const back = viewerSpot();
    ball.returning = true;
    ball.lobbing = false;
    ball.flying = true;
    ball.live = true;
    ball.vx = 0;
    ball.vy = 0;
    ball.flight = { fromX: ball.x, fromY: ball.y, toX: back.x, toY: back.y, t: 0 };
    return;
  }
  const next = heads.find((item) => item.id !== head.id) || head;
  ball.lobbing = true;
  ball.returning = false;
  ball.flying = true;
  ball.live = true;
  ball.z = 0;
  ball.vx = 0;
  ball.vy = 0;
  ball.flight = {
    fromX: ball.x,
    fromY: ball.y,
    toX: next.x - gw() / 2,
    toY: next.y - gh() / 2,
    lift: Math.min(120, Math.max(80, ball.y - 20)),
    t: 0,
  };
}

function explode() {
  ball.live = false;
  ball.flying = false;
  ball.vx = 0;
  ball.vy = 0;
  ballEl.hidden = true;
  setPose("flinch");
  poseUntil = clock + 1.6;
  const impact = document.getElementById("impact");
  impact.classList.add("blast");
  bonk();
  shakePhone();
  document.getElementById("clip-actions").hidden = false;
  clearTimeout(explode.timer);
  explode.timer = setTimeout(() => impact.classList.remove("blast"), 420);
  disneyExplosion(ball.x + gw() / 2, ball.y + gh() / 2);
}

function disneyExplosion(cx, cy) {
  const container = document.getElementById("photo");
  const colors = ["#ffd23a","#ff6b6b","#4ecdc4","#a8e6cf","#ff8b94","#ffd3b6","#c7f2a4","#ffaaa5","#98ddca","#f9ca24"];

  // Expanding rings
  for (let i = 0; i < 3; i++) {
    const ring = document.createElement("div");
    ring.className = "disney-ring";
    ring.style.left = `${cx}px`;
    ring.style.top = `${cy}px`;
    ring.style.animationDelay = `${i * 80}ms`;
    ring.style.borderColor = colors[i * 3];
    container.appendChild(ring);
    setTimeout(() => ring.remove(), 900 + i * 80);
  }

  // Colored confetti particles
  const count = 18;
  for (let i = 0; i < count; i++) {
    const angle = (360 / count) * i + Math.random() * 12 - 6;
    const dist = 55 + Math.random() * 75;
    const size = 5 + Math.random() * 9;
    const p = document.createElement("div");
    p.className = "disney-particle";
    p.style.cssText = `left:${cx}px;top:${cy}px;width:${size}px;height:${size}px;background:${colors[i % colors.length]};border-radius:${Math.random() > 0.4 ? "50%" : "3px"};--tx:${(Math.cos(angle * Math.PI / 180) * dist).toFixed(1)}px;--ty:${(Math.sin(angle * Math.PI / 180) * dist).toFixed(1)}px;animation-delay:${Math.random() * 60}ms`;
    container.appendChild(p);
    setTimeout(() => p.remove(), 1100);
  }

  // Emoji sparks flying out
  const sparks = ["⭐","✨","💥","🌟","⚡","💫"];
  for (let i = 0; i < 6; i++) {
    const angle = (360 / 6) * i + 30;
    const dist = 50 + Math.random() * 70;
    const el = document.createElement("div");
    el.className = "disney-spark";
    el.textContent = sparks[i];
    el.style.cssText = `left:${cx}px;top:${cy}px;--tx:${(Math.cos(angle * Math.PI / 180) * dist).toFixed(1)}px;--ty:${(Math.sin(angle * Math.PI / 180) * dist).toFixed(1)}px;animation-delay:${40 + i * 20}ms`;
    container.appendChild(el);
    setTimeout(() => el.remove(), 1100);
  }

  // BOOM! text
  const boom = document.createElement("div");
  boom.className = "disney-boom";
  boom.textContent = "BOOM!";
  boom.style.left = `${cx}px`;
  boom.style.top = `${cy - 28}px`;
  container.appendChild(boom);
  setTimeout(() => boom.remove(), 900);
}

function viewerSpot() {
  const item = gear();
  const visualW = item.near || item.w;
  const visualH = item.h * (visualW / item.w);
  const margin = item.id === "ball" ? 4 : 36;
  return {
    x: PHOTO_W / 2 - item.w / 2,
    y: PHOTO_H - visualH / 2 - item.h / 2 - margin,
  };
}

function playLob(dt) {
  const flight = ball.flight;
  flight.t = Math.min(1, (flight.t || 0) + dt / 0.8);
  const t = flight.t;
  const arc = Math.sin(Math.PI * t) * flight.lift;
  ball.x = flight.fromX + (flight.toX - flight.fromX) * t;
  ball.y = flight.fromY + (flight.toY - flight.fromY) * t - arc;
  ball.z = 0;
  ball.spin -= 520 * dt;
  if (t < 1) return;
  ball.lobbing = false;
  ball.flying = false;
  ball.x = flight.toX;
  ball.y = flight.toY;
  ball.vx = 0;
  ball.vy = 40;
  for (const head of heads) head.touching = false;
}

function returnToViewer(dt) {
  const flight = ball.flight;
  flight.t = Math.min(1, (flight.t || 0) + dt / 0.62);
  const ease = 1 - (1 - flight.t) ** 2;
  const lift = Math.min(flight.fromY, flight.toY) - 64;
  const midX = (flight.fromX + flight.toX) / 2;
  const u = 1 - ease;
  ball.x = u * u * flight.fromX + 2 * u * ease * midX + ease * ease * flight.toX;
  ball.y = u * u * flight.fromY + 2 * u * ease * lift + ease * ease * flight.toY;
  ball.z = ease;
  ball.spin += 720 * dt;
  if (flight.t < 1) return;
  ball.returning = false;
  ball.flying = false;
  ball.z = 1;
  ball.x = flight.toX;
  ball.y = flight.toY;
  ball.vx = 0;
  ball.vy = 0;
  ball.live = false;
  ballEl.classList.add("selected");
  bonk();
}
function feetPoint() {
  const x = heads.length ? heads.reduce((sum, head) => sum + head.x, 0) / heads.length : PHOTO_W / 2;
  return { x, y: 462 };
}

function chestPoint(id) {
  if (id === "right") return { x: 264, y: 208 };
  return { x: 156, y: 186 };
}

function handPoint(id) {
  if (id === "right") return { x: 256, y: 218 };
  return { x: 176, y: 192 };
}

function catcherFor(vx, vy) {
  const aim = landingPoint(vx, vy);
  let catcher = heads[0];
  let best = Infinity;
  for (const head of heads) {
    const dist = Math.hypot(aim.x - head.x, aim.y - head.y);
    if (dist < best) {
      best = dist;
      catcher = head;
    }
  }
  return catcher || heads[0];
}

function glide(dt) {
  const from = ball.rollFrom || { x: ball.x, y: ball.y };
  ball.rollFrom = from;
  ball.rollT += dt;
  const id = gear().id;
  const duration = id === "heart" ? 1.55 : id === "banana" ? 1.45 : 0.95;
  const t = clamp(ball.rollT / duration, 0, 1);
  const ease = t * t * (3 - 2 * t);
  if (id === "heart") {
    const head = heads.find((item) => item.id === ball.catchId) || heads[0];
    const chest = chestPoint(head.id);
    const toX = chest.x - gw() / 2;
    const toY = chest.y - gh() / 2;
    const sink = 0.38;
    if (t < sink) {
      const u = t / sink;
      const e = u * u * (3 - 2 * u);
      ball.x = from.x + (toX - from.x) * e;
      ball.y = from.y + (toY - from.y) * e - Math.sin(Math.PI * u) * 30;
      ball.z = 1 - e * 0.94;
      ball.flex = 0.1;
      ball.spin = 0;
      ball.yaw = Math.sin(Math.PI * u) * 16;
      ball.pitch = 0;
      ball.fade = 1;
    } else {
      const u = (t - sink) / (1 - sink);
      const pop = 1 - (1 - Math.min(1, u / 0.42)) ** 3;
      const thump = u > 0.4 ? Math.pow(Math.sin((u - 0.4) * Math.PI * 3), 2) : 0;
      ball.x = toX;
      ball.y = toY - pop * 8;
      ball.z = 0.06 + pop * 0.92;
      ball.flex = thump;
      ball.spin = 0;
      ball.yaw = 0;
      ball.pitch = 8;
      ball.fade = 1;
      if (!ball.tripped && u > 0.16) {
        ball.tripped = true;
        setPose(head.id === "right" ? "heartRight" : "heartLeft");
        document.getElementById("clip-actions").hidden = false;
      }
    }
    if (t >= 1) {
      ball.live = false;
      ball.caught = true;
      ball.x = toX;
      ball.y = toY - 8;
      ball.z = 0.98;
      ball.spin = 0;
      ball.yaw = 0;
      ball.pitch = 8;
      ball.flex = 1;
      ballEl.classList.add("selected");
    }
  } else if (id === "flower") {
    const head = heads.find((item) => item.id === ball.catchId) || heads[0];
    const hands = handPoint(head.id);
    const endZ = 0.08;
    const toX = hands.x - gw() / 2;
    const toY = hands.y - gh() / 2 - 18;
    const arc = Math.sin(Math.PI * t) * 56;
    ball.x = from.x + (toX - from.x) * ease;
    ball.y = from.y + (toY - from.y) * ease - arc;
    ball.z = 1 - ease * (1 - endZ);
    ball.fade = 1;
    ball.spin = Math.sin(t * Math.PI) * 14;
    ball.yaw = Math.sin(t * Math.PI) * 36;
    ball.pitch = Math.sin(t * Math.PI) * 24;
    ball.flex = 0.5 + 0.5 * Math.sin(t * Math.PI * 3);
    if (!ball.tripped && t > 0.58) {
      ball.tripped = true;
      setPose(head.id === "right" ? "catchRight" : "catchLeft");
      document.getElementById("clip-actions").hidden = false;
    }
    if (t >= 1) {
      ball.live = false;
      ball.caught = true;
      ball.x = toX;
      ball.y = toY;
      ball.z = endZ;
      ball.spin = 0;
      ball.yaw = 0;
      ball.pitch = 4;
      ball.flex = 1;
      ballEl.classList.add("selected");
      bonk();
    }
  } else {
    const pastX = 42;
    const pastY = 424;
    const arc = Math.sin(Math.PI * t) * 28;
    ball.x = from.x + (pastX - from.x) * ease;
    ball.y = from.y + (pastY - from.y) * ease - arc;
    ball.z = 1 - ease * 0.45;
    ball.flex = ease;
    ball.spin = -6 - ease * 14;
    ball.yaw = 0;
    ball.pitch = 0;
    ball.fade = t < 0.86 ? 1 : clamp((1 - t) / 0.14, 0, 1);
    if (!ball.tripped && t > 0.4) {
      ball.tripped = true;
      document.getElementById("clip-actions").hidden = false;
    }
    if (t >= 1) {
      ball.live = false;
      ball.fade = 0;
      ballEl.hidden = true;
    }
  }
  clock += dt;
  sample();
  placeBall();
}

function nearestHead() {
  const cx = ball.x + gw() / 2;
  const cy = ball.y + gh() / 2;
  let best = null;
  let bestDist = 78;
  for (const head of heads) {
    const dist = Math.hypot(cx - head.x, cy - head.y);
    if (dist < bestDist) {
      best = head;
      bestDist = dist;
    }
  }
  return best;
}

function setPose(next) {
  if (!POSES[next] || pose === next) return;
  pose = next;
  document.getElementById("scene").src = POSES[next];
}

function bonk() {
  const impact = document.getElementById("impact");
  impact.hidden = false;
  impact.style.left = `${ball.x + gw() / 2}px`;
  impact.style.top = `${ball.y + gh() / 2}px`;
  impact.classList.remove("go");
  ballEl.classList.remove("bonk");
  void impact.offsetWidth;
  impact.classList.add("go");
  ballEl.classList.add("bonk");
}

function hideImpact() {
  const impact = document.getElementById("impact");
  impact.hidden = true;
  impact.classList.remove("go");
  impact.classList.remove("blast");
  ballEl.classList.remove("bonk");
}

function startReplay() {
  if (!tape.length) return;
  mode = "replay";
  sheet.dataset.mode = "replay";
  replayAt = 0;
  replay.seen = new Set();
  ballEl.hidden = false;
  reactEl.hidden = true;
}

function replay(dt) {
  replayAt += dt;
  const end = tape[tape.length - 1].t;
  if (replayAt > end + 0.4) {
    replayAt = 0;
    replay.seen = new Set();
  }
  let frame = tape[0];
  for (const sample of tape) {
    if (sample.t > replayAt) break;
    frame = sample;
  }
  ball.x = frame.x;
  ball.y = frame.y;
  ball.z = frame.z || 0;
  ball.spin = frame.spin || 0;
  ball.fade = frame.fade == null ? 1 : frame.fade;
  ball.flex = frame.flex || 0;
  ball.pitch = frame.pitch || 0;
  ball.yaw = frame.yaw || 0;
  if (frame.gear && frame.gear !== gearId && GEARS[frame.gear]) {
    gearId = frame.gear;
    paintGear();
  }
  setPose(frame.pose || "base");
  placeBall();
  if (!replay.seen) replay.seen = new Set();
  for (const hit of hits) {
    if (replay.seen.has(hit) || replayAt < hit.t || replayAt - hit.t > 0.08) continue;
    replay.seen.add(hit);
    bonk();
    shakePhone();
  }
}

async function saveGif() {
  const button = document.getElementById("save-gif");
  button.disabled = true;
  button.textContent = "Saving…";
  const w = 195;
  const h = 260;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const frames = tape.filter((_, index) => index % 2 === 0).slice(0, 60);
  const gif = GIFEncoder();
  for (const frame of frames) {
    drawFrame(ctx, frame);
    const data = ctx.getImageData(0, 0, w, h).data;
    const palette = quantize(data, 32);
    const index = applyPalette(data, palette);
    gif.writeFrame(index, w, h, { palette, delay: 180 });
  }
  gif.finish();
  const blob = new Blob([gif.bytes()], { type: "image/gif" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "park-moments.gif";
  link.click();
  URL.revokeObjectURL(url);
  button.disabled = false;
  button.textContent = "Save GIF";
  document.getElementById("bar-sub").textContent = "GIF saved";
}

function drawFrame(ctx, frame) {
  const shot = poseImages[frame.pose] || poseImages.base;
  if (shot.complete) ctx.drawImage(shot, 0, 0, 195, 260);
  const active = hits.find((hit) => frame.t >= hit.t && frame.t - hit.t < 0.28);
  drawProjectile(ctx, frame);
  if (active && frame.gear === "bomb") drawBurst(ctx, (frame.x + 23) / 2, (frame.y + 23) / 2);
}

function drawProjectile(ctx, frame) {
  const fade = frame.fade == null ? 1 : frame.fade;
  if (fade <= 0.04) return;
  const item = (frame.gear === "emoji" || frame.gear === "custom") && customGear ? customGear : GEARS[frame.gear] || GEARS.ball;
  const z = frame.z || 0;
  const near = item.near || item.w;
  const w = (item.w + (near - item.w) * z) / 2;
  const h = w * (item.h / item.w);
  const cx = (frame.x + item.w / 2) / 2;
  const cy = (frame.y + item.h / 2) / 2;
  ctx.save();
  ctx.globalAlpha = fade;
  ctx.translate(cx, cy);
  if (frame.spin) ctx.rotate(((frame.spin || 0) * Math.PI) / 180);
  const sprite = item.emoji ? null : item.id === "custom" ? sprites.custom : item.id === "ball" ? sprites.ball : null;
  if (item.emoji) {
    ctx.font = `${Math.round(Math.min(w, h) * 0.86)}px "Apple Color Emoji", "Segoe UI Emoji", sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(item.emoji, 0, h * 0.06);
  } else if (sprite?.complete && sprite.naturalWidth) ctx.drawImage(sprite, -w / 2, -h / 2, w, h);
  else if (item.id === "banana") drawBanana(ctx, w, h, frame.flex || 0);
  else if (item.id === "heart") drawHeart(ctx, w, h);
  else if (item.id === "flower") drawFlower(ctx, w, h);
  else {
    ctx.fillStyle = "#161616";
    ctx.beginPath();
    ctx.arc(0, h * 0.08, Math.min(w, h) / 2.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawBanana(ctx, w, h, flex) {
  ctx.fillStyle = "#f5c400";
  ctx.beginPath();
  ctx.ellipse(0, flex * h * 0.08, w * 0.42, h * 0.28, -0.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#6b4423";
  ctx.lineWidth = Math.max(2, h * 0.08);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(w * 0.32, -h * 0.12);
  ctx.lineTo(w * 0.42, -h * 0.32);
  ctx.stroke();
}

function drawHeart(ctx, w, h) {
  const shade = ctx.createRadialGradient(-w * 0.12, -h * 0.18, w * 0.04, 0, h * 0.02, w * 0.55);
  shade.addColorStop(0, "#ffe4ea");
  shade.addColorStop(0.35, "#ff4d6d");
  shade.addColorStop(1, "#7f1239");
  ctx.fillStyle = shade;
  ctx.beginPath();
  ctx.moveTo(0, h * 0.36);
  ctx.bezierCurveTo(-w * 0.55, 0, -w * 0.34, -h * 0.5, 0, -h * 0.12);
  ctx.bezierCurveTo(w * 0.34, -h * 0.5, w * 0.55, 0, 0, h * 0.36);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.beginPath();
  ctx.ellipse(-w * 0.16, -h * 0.14, w * 0.11, h * 0.05, -0.55, 0, Math.PI * 2);
  ctx.fill();
}

function drawFlower(ctx, w, h) {
  const stem = ctx.createLinearGradient(-w * 0.08, 0, w * 0.08, 0);
  stem.addColorStop(0, "#14532d");
  stem.addColorStop(0.5, "#4ade80");
  stem.addColorStop(1, "#166534");
  ctx.strokeStyle = stem;
  ctx.lineWidth = Math.max(2, w * 0.08);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(0, h * 0.42);
  ctx.quadraticCurveTo(w * 0.04, h * 0.2, 0, -h * 0.02);
  ctx.stroke();
  ctx.fillStyle = "#22c55e";
  ctx.beginPath();
  ctx.ellipse(-w * 0.16, h * 0.16, w * 0.16, h * 0.06, -0.6, 0, Math.PI * 2);
  ctx.ellipse(w * 0.16, h * 0.08, w * 0.16, h * 0.06, 0.5, 0, Math.PI * 2);
  ctx.fill();
  const bloom = ctx.createRadialGradient(-w * 0.08, -h * 0.22, w * 0.02, 0, -h * 0.16, w * 0.34);
  bloom.addColorStop(0, "#fff1f2");
  bloom.addColorStop(0.45, "#fb7185");
  bloom.addColorStop(1, "#9f1239");
  ctx.fillStyle = bloom;
  for (let i = 0; i < 5; i++) {
    ctx.beginPath();
    ctx.ellipse(0, -h * 0.22, w * 0.13, h * 0.16, (Math.PI * 2 * i) / 5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#fff7ed";
  ctx.beginPath();
  ctx.arc(0, -h * 0.16, Math.min(w, h) * 0.07, 0, Math.PI * 2);
  ctx.fill();
}

function drawBurst(ctx, x, y) {
  ctx.save();
  ctx.translate(x, y);
  ctx.fillStyle = "#ffe566";
  ctx.strokeStyle = "#e07a00";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < 8; i++) {
    const angle = (Math.PI / 4) * i - Math.PI / 2;
    const radius = i % 2 === 0 ? 7 : 3;
    ctx.lineTo(Math.cos(angle) * radius, Math.sin(angle) * radius);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function shakePhone() {
  if (navigator.vibrate) navigator.vibrate([18, 28, 24]);
  phone.animate(
    [
      { transform: "translate(0px, 0px) rotate(0deg)" },
      { transform: "translate(-10px, 4px) rotate(-1.4deg)" },
      { transform: "translate(10px, -4px) rotate(1.4deg)" },
      { transform: "translate(-6px, 2px) rotate(-0.6deg)" },
      { transform: "translate(0px, 0px) rotate(0deg)" },
    ],
    { duration: 340, easing: "ease-out" }
  );
}

function placeBall() {
  const size = dims();
  const center = visualCenter();
  ballEl.style.width = `${size.w}px`;
  ballEl.style.height = `${size.h}px`;
  ballEl.style.left = `${center.x - size.w / 2}px`;
  ballEl.style.top = `${center.y - size.h / 2}px`;
  ballEl.style.zIndex = ball.z > 0.15 ? "8" : "5";
  ballEl.style.opacity = String(ball.fade == null ? 1 : ball.fade);
  if (gear().id === "banana") ballEl.innerHTML = bananaSvg(ball.flex || 0);
  const emoji = ballEl.querySelector(".emoji");
  if (emoji) emoji.style.fontSize = `${size.w * 0.78}px`;
  const pulse = gear().id === "heart" && (ball.live || ball.caught) ? 0.84 + (ball.flex || 0) * 0.36 : 1;
  const spin = ball.spin || 0;
  const pitch = ball.pitch || 0;
  const yaw = ball.yaw || 0;
  if (gear().kind === "soft" || spin || pitch || yaw) {
    ballEl.style.transform = `rotate(${spin}deg) rotateX(${pitch}deg) rotateY(${yaw}deg) scale(${pulse})`;
  } else if (!ballEl.classList.contains("bonk")) ballEl.style.transform = "";
}

function dims() {
  const item = gear();
  const near = item.near || item.w;
  const nearH = item.h * (near / item.w);
  return {
    w: item.w + (near - item.w) * ball.z,
    h: item.h + (nearH - item.h) * ball.z,
  };
}

function visualSize() {
  return dims().w;
}

function visualCenter() {
  return { x: ball.x + gw() / 2, y: ball.y + gh() / 2 };
}

function gear() {
  return (gearId === "custom" || gearId === "emoji") && customGear ? customGear : GEARS[gearId] || GEARS.ball;
}

function gw() {
  return gear().w;
}

function gh() {
  return gear().h;
}

function paintGear() {
  const item = gear();
  for (const button of document.querySelectorAll("#gear [data-gear]")) {
    button.classList.toggle("on", button.dataset.gear === gearId);
  }
  ballEl.classList.toggle("plain", Boolean(item.svg || item.emoji));
  ballEl.classList.toggle("round", item.id === "ball" && !item.emoji);
  for (const button of document.querySelectorAll("#emoji-list [data-emoji]")) {
    button.classList.toggle("on", gearId === "emoji" && item.emoji === button.dataset.emoji);
  }
  if (item.emoji) {
    ballEl.innerHTML = `<span class="emoji">${item.emoji}</span>`;
    ballEl.style.backgroundImage = "none";
  } else if (item.svg) {
    ballEl.innerHTML = item.svg;
    ballEl.style.backgroundImage = "none";
  } else {
    ballEl.replaceChildren();
    ballEl.style.backgroundImage = `url("${item.src}")`;
  }
}

function setHint() {
  const hints = {
    ball: "",
    banana: "Tap the banana, then flick. It bends past them.",
    heart: "Tap the heart, then flick it to one of them. His heart pops out.",
    flower: "Tap the flower, then flick it to one of them. He catches it.",
    bomb: "",
    custom: "Tap your image, then flick. It comes from you, then bounces.",
    emoji: "",
  };
  document.getElementById("play-hint").textContent = hints[gearId] || hints.ball;
}

function selectEmoji(emoji) {
  customGear = { id: "emoji", emoji, w: 56, h: 56, near: 148, kind: "emoji" };
  gearId = "emoji";
  paintGear();
  setHint();
  ball.live = false;
  ball.flying = false;
  ball.vx = 0;
  ball.vy = 0;
  if (mode === "play") parkBall();
}

function addGearImage(file) {
  const reader = new FileReader();
  reader.onload = () => {
    customGear = { id: "custom", w: 48, h: 48, near: 140, kind: "bounce", src: reader.result };
    sprites.custom.src = reader.result;
    gearId = "custom";
    paintGear();
    setHint();
    ball.live = false;
    ball.flying = false;
    ball.vx = 0;
    ball.vy = 0;
    if (mode === "play") parkBall();
  };
  reader.readAsDataURL(file);
}

function parkBall() {
  const item = gear();
  ball.z = 1;
  ball.flying = false;
  ball.flight = null;
  ball.spin = 0;
  ball.fade = 1;
  ball.tripped = false;
  ball.rollT = 0;
  ball.rollFrom = null;
  ball.flex = 0;
  ball.pitch = 0;
  ball.yaw = 0;
  ball.caught = false;
  ball.catchId = null;
  ball.returning = false;
  ball.lobbing = false;
  setPose("base");
  const visualW = item.near || item.w;
  const visualH = item.h * (visualW / item.w);
  const margin = item.id === "ball" ? 4 : 36;
  ball.x = PHOTO_W / 2 - item.w / 2;
  ball.y = PHOTO_H - visualH / 2 - item.h / 2 - margin;
  ball.vx = 0;
  ball.vy = 0;
  ball.live = false;
  ball.held = false;
  ballEl.hidden = false;
  ballEl.classList.remove("bonk");
  ballEl.classList.add("selected");
  placeBall();
}

function fly(dt) {
  if (ball.lobbing) {
    playLob(dt);
    return;
  }
  if (ball.returning) {
    returnToViewer(dt);
    return;
  }
  const kind = gear().kind;
  if (kind === "roll") ball.spin += 280 * dt;
  ball.z = Math.max(0, ball.z - dt / 0.7);
  const progress = 1 - ball.z ** 1.35;
  const flight = ball.flight;
  ball.x = flight.fromX + (flight.toX - flight.fromX) * progress;
  ball.y = flight.fromY + (flight.toY - flight.fromY) * progress;
  const near = nearestHead();
  if (kind === "bounce" && near && ball.z < 0.45) {
    setPose(near.id);
    poseUntil = Math.max(poseUntil, clock + 0.45);
  }
  if (kind === "bomb" && near && ball.z < 0.4) {
    setPose("flinch");
    poseUntil = Math.max(poseUntil, clock + 0.5);
  }
  if (ball.z > 0) return;
  ball.flying = false;
  ball.x = flight.toX;
  ball.y = flight.toY;
  if (kind === "roll") {
    ball.z = 0;
    ball.y = GROUND - gh();
    ball.vx = (flight.vx || 1) >= 0 ? 180 : -180;
    ball.vy = 0;
    ball.tripped = false;
    return;
  }
  const center = visualCenter();
  const hit = heads.find((head) => Math.hypot(center.x - head.x, center.y - head.y) < gw() / 2 + head.r + 8);
  if (kind === "bomb" && hit) {
    explode();
    return;
  }
  const speed = clamp(Math.hypot(flight.vx, flight.vy) || 360, 320, 680);
  const angle = Math.atan2(flight.vy || -1, flight.vx || 0.4);
  ball.vx = Math.cos(angle) * speed;
  ball.vy = -Math.abs(Math.sin(angle) * speed);
  if (Math.abs(ball.vy) < 200) ball.vy = -320;
  if (hit && !hit.touching) {
    hit.touching = true;
    header(hit);
  }
}

function landingPoint(vx, vy) {
  const originX = PHOTO_W / 2;
  const originY = PHOTO_H - 40;
  const len = Math.hypot(vx, vy);
  if (len < 80) {
    const head = heads[0];
    if (head) return { x: head.x, y: head.y };
    return { x: PHOTO_W / 2, y: PHOTO_H * 0.4 };
  }
  const tx = clamp(originX + (vx / len) * 360, 28, PHOTO_W - 28);
  const ty = clamp(originY + (vy / len) * 360, 28, PHOTO_H - 48);
  let best = null;
  let bestDist = 120;
  for (const head of heads) {
    const dist = Math.hypot(tx - head.x, ty - head.y);
    if (dist < bestDist) {
      best = head;
      bestDist = dist;
    }
  }
  if (best) return { x: best.x, y: best.y };
  return { x: tx, y: ty };
}

function sample() {
  if (clock < sampleAt) return;
  tape.push({ t: clock, x: ball.x, y: ball.y, z: ball.z, spin: ball.spin, fade: ball.fade ?? 1, flex: ball.flex || 0, pitch: ball.pitch || 0, yaw: ball.yaw || 0, gear: gearId, pose });
  sampleAt = clock + 0.05;
}

function photoPoint(event) {
  const rect = photo.getBoundingClientRect();
  const scaleX = PHOTO_W / rect.width;
  const scaleY = PHOTO_H / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function animate() {
  if (custom && custom.x == null) {
    setNote("Tap where the ball should hit, then create the reaction.");
    return;
  }
  if (custom && !POSES.spot) {
    setNote("Create the reaction, then animate.");
    return;
  }
  startPlay();
}

function setNote(text) {
  document.getElementById("moment-copy").textContent = text;
}

function selectShot(id) {
  shotId = id;
  const index = shots.findIndex((shot) => shot.id === id);
  document.getElementById("shot-count").textContent = `${index + 1} / ${shots.length}`;
  document.getElementById("shot-label").textContent = shots[index]?.label || "";
  for (const thumb of document.querySelectorAll(".thumb")) {
    thumb.classList.toggle("active", thumb.dataset.id === id);
  }
  document.getElementById("aim").hidden = true;
  if (id === "friends") {
    custom = null;
    heads = [
      { id: "left", name: "Left", x: 146, y: 128, r: 36, line: "Heads it back" },
      { id: "right", name: "Right", x: 252, y: 124, r: 28, line: "Heads it back" },
    ];
    POSES.base = "/friends.png";
    poseImages.base = imageFrom("/friends.png");
    delete POSES.spot;
    pose = "";
    setPose("base");
    document.getElementById("make").hidden = true;
    setNote("");
    return;
  }
  const shot = shots[index];
  custom = shot.custom;
  POSES.base = shot.src;
  poseImages.base = imageFrom(shot.src);
  if (shot.reaction) {
    POSES.spot = shot.reaction;
    poseImages.spot = imageFrom(shot.reaction);
  } else delete POSES.spot;
  pose = "";
  setPose("base");
  heads = custom.x == null ? [] : [{ id: "spot", name: "Them", x: custom.x * PHOTO_W, y: custom.y * PHOTO_H, r: 26, line: "Funny reaction" }];
  if (custom.x != null) {
    const aim = document.getElementById("aim");
    aim.hidden = false;
    aim.style.left = `${custom.x * PHOTO_W}px`;
    aim.style.top = `${custom.y * PHOTO_H}px`;
  }
  document.getElementById("make").hidden = custom.x == null;
  setNote(shot.reaction ? "" : "Tap where the ball should hit, then create the reaction.");
}
function loadPhoto(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const image = new Image();
    image.onload = () => {
      const framed = framePhoto(image);
      const id = `photo-${shots.length}`;
      const shot = {
        id,
        label: shots.length === 1 ? "Photo" : `Photo ${shots.length}`,
        src: framed.url,
        custom: { image: framed.image, mime: "image/jpeg", x: null, y: null },
        reaction: null,
      };
      shots.push(shot);
      const thumb = document.createElement("button");
      thumb.type = "button";
      thumb.className = "thumb";
      thumb.dataset.id = id;
      thumb.setAttribute("aria-label", shot.label);
      const picture = document.createElement("img");
      picture.src = framed.url;
      picture.alt = "";
      thumb.append(picture);
      document.getElementById("load").before(thumb);
      selectShot(id);
    };
    image.src = reader.result;
  };
  reader.readAsDataURL(file);
}

function framePhoto(image) {
  const canvas = document.createElement("canvas");
  canvas.width = 780;
  canvas.height = 1040;
  const ctx = canvas.getContext("2d");
  const scale = Math.max(canvas.width / image.naturalWidth, canvas.height / image.naturalHeight);
  const dw = image.naturalWidth * scale;
  const dh = image.naturalHeight * scale;
  ctx.drawImage(image, (canvas.width - dw) / 2, (canvas.height - dh) / 2, dw, dh);
  const url = canvas.toDataURL("image/jpeg", 0.88);
  return { url, image: url.split(",")[1] };
}

function imageFrom(src) {
  const picture = new Image();
  picture.src = src;
  return picture;
}

function placeAim(local) {
  const x = clamp(local.x, 24, PHOTO_W - 24);
  const y = clamp(local.y, 24, PHOTO_H - 24);
  custom.x = x / PHOTO_W;
  custom.y = y / PHOTO_H;
  const shot = shots.find((item) => item.id === shotId);
  if (shot) shot.custom = custom;
  heads = [{ id: "spot", name: "Them", x, y, r: 26, line: "Funny reaction" }];
  const aim = document.getElementById("aim");
  aim.hidden = false;
  aim.style.left = `${x}px`;
  aim.style.top = `${y}px`;
  document.getElementById("make").hidden = false;
  setNote("Impact set. Create the reaction, then Animate with AI.");
}

async function createReactions() {
  if (!custom?.image || custom.x == null) return;
  const button = document.getElementById("make");
  button.disabled = true;
  button.textContent = "Creating…";
  document.getElementById("moment-copy").textContent = "Making a funny reaction at the impact…";
  try {
    const response = await fetch("/api/react", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: custom.image, mime: custom.mime, x: custom.x, y: custom.y }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Reaction failed");
    POSES.spot = data.image;
    poseImages.spot = imageFrom(data.image);
    const shot = shots.find((item) => item.id === shotId);
    if (shot) shot.reaction = data.image;
    setNote("Reaction ready. Tap Animate with AI.");
  } catch (error) {
    document.getElementById("moment-copy").textContent = error.message;
  }
  button.disabled = false;
  button.textContent = "Create reaction";
}
