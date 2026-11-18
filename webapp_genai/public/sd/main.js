import { SYMBOLS, CATEGORIES, matchPrompt, leftoverPrompt, highlightPrompt, addCustomSymbol, loadCustomSymbols, symbolSlug, unknownWords, rejectSymbol, clearRejected, isRejected, syncRejected } from "./symbols.js?v=animals2";

const screen = document.querySelector(".screen");
const sheet = document.getElementById("sheet");
const stage = document.getElementById("stage");
const photoUnder = document.getElementById("photo-under");
const photoOut = document.getElementById("photo-out");
const busy = document.getElementById("busy");
const busyCopy = document.getElementById("busy-copy");
const promptEl = document.getElementById("prompt");
const promptMark = document.getElementById("prompt-mark");
const barTitle = document.getElementById("bar-title");
const barSub = document.getElementById("bar-sub");
const sketchBtn = document.getElementById("sketch");
const bananaBtn = document.getElementById("banana");
const thumbs = document.getElementById("thumbs");
const fileEl = document.getElementById("file");
const gizmo = document.getElementById("gizmo");
const gizmoBox = gizmo.querySelector(".gizmo-box");
const gizmoCube = gizmo.querySelector(".gizmo-cube");
const editBar = document.getElementById("edit-bar");
const undoBtn = document.getElementById("undo");
const redoBtn = document.getElementById("redo");
const deleteBtn = document.getElementById("delete-obj");
const libEl = document.getElementById("lib");
const libCats = document.getElementById("lib-cats");
const libGrid = document.getElementById("lib-grid");
const libToggle = document.getElementById("lib-toggle");
const libClose = document.getElementById("lib-close");

const PHOTO_W = 390;
const PHOTO_H = 520;
const HIT_PAD = 8;
const HIT_MIN = 22;

let svgMarkup = "";
let concept = "";
let scenes = [];
let sourcePhoto = "";
let mode = "sketch";
let selected = null;
let gesture = null;
let history = [];
let historyIndex = -1;
let libCat = "animals";
let typeTimer = 0;

function fit() {
  if (document.body.classList.contains("embed")) {
    screen.style.setProperty("--s", "1");
    return;
  }
  const box = screen.getBoundingClientRect();
  const scale = Math.min(box.width / 390, box.height / 844);
  screen.style.setProperty("--s", String(scale || 1));
}
fit();
new ResizeObserver(fit).observe(screen);

function emptyNode() {
  return stage.querySelector("#empty");
}

function setBusy(on, copy) {
  busy.hidden = !on;
  if (copy) busyCopy.textContent = copy;
  sketchBtn.disabled = on;
  bananaBtn.disabled = on || !svgMarkup;
}

const COPY = {
  en: {
    pick: "Pick a photo",
    work: "Fast preview",
    recents: "Recents",
    panelOrEmpty: "Photo or empty",
    emptyOpen: "Choose a photo, or start empty.",
    emptyBeat: "",
    emptyNext: "",
    placeholder: "Type dog, cat, bird… or open Symbols.",
    placeholderNext: "Type what to add, or open Symbols.",
    sketch: "Sketch",
    symbols: "Symbols",
    banana: "Nano Banana",
    describeFirst: "Describe a scene first.",
    drawing: "Drawing on the photo…",
    laying: "Laying out objects…",
    panelFrom: "Photograph from the sketch.",
  },
  zh: {
    pick: "选择照片",
    work: "快速预览",
    recents: "最近",
    panelOrEmpty: "照片或空白",
    emptyOpen: "选一张照片，或从空白开始。",
    emptyBeat: "",
    emptyNext: "",
    placeholder: "输入狗、猫、鸟…或打开符号库。",
    placeholderNext: "输入要加的物体，或打开符号库。",
    sketch: "草图",
    symbols: "符号",
    banana: "Nano Banana",
    describeFirst: "请先描述场景。",
    drawing: "正在照片上画草图…",
    laying: "正在排布物体…",
    panelFrom: "已由草图生成照片。",
  },
};

let uiLang = localStorage.getItem("lang") === "en" ? "en" : "zh";
function copy() {
  return COPY[uiLang];
}

function applyLang(next) {
  if (next === "en" || next === "zh") uiLang = next;
  const text = copy();
  const recents = document.getElementById("shot-count");
  const label = document.getElementById("shot-label");
  if (recents) recents.textContent = text.recents;
  if (label) label.textContent = text.panelOrEmpty;
  sketchBtn.textContent = text.sketch;
  bananaBtn.textContent = text.banana;
  if (libToggle) libToggle.textContent = text.symbols;
  renderLibrary();
  paintPrompt();
  promptEl.placeholder = sourcePhoto ? text.placeholderNext : text.placeholder;
  const empty = document.getElementById("empty");
  if (empty) {
    if (sheet.dataset.mode === "open") empty.textContent = text.emptyOpen;
    else if (sourcePhoto && !svgMarkup) empty.textContent = text.emptyNext;
    else if (!svgMarkup) empty.textContent = text.emptyBeat;
  }
  if (sheet.dataset.mode === "open") barTitle.textContent = text.pick;
  else barTitle.textContent = text.work;
}

function setView(view) {
  sheet.dataset.mode = view;
  if (view === "open") {
    barTitle.textContent = copy().pick;
    barSub.textContent = "";
    gizmo.hidden = true;
    editBar.hidden = true;
    hideResult();
    selected = null;
  } else {
    barTitle.textContent = copy().work;
  }
  applyLang(uiLang);
  syncEditButtons();
}

function setStatus(text, error = false) {
  barSub.textContent = text;
  barSub.style.color = error ? "#f0a0a0" : "";
}

function wrapSvg(raw) {
  const match = String(raw).match(/<svg[\s\S]*<\/svg>/i);
  let svg = match ? match[0] : raw;
  if (!/<svg/i.test(svg)) {
    svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${PHOTO_W} ${PHOTO_H}">${svg}</svg>`;
  }
  svg = svg.replace(/<svg([^>]*)>/i, (open, attrs) => {
    let next = attrs;
    if (!/xmlns=/.test(next)) next += ` xmlns="http://www.w3.org/2000/svg"`;
    if (!/viewBox=/.test(next)) next += ` viewBox="0 0 ${PHOTO_W} ${PHOTO_H}"`;
    return `<svg${next}>`;
  });
  return svg;
}

function hideResult() {
  photoOut.hidden = true;
}

function setUnderlay(url) {
  sourcePhoto = url || "";
  if (url) {
    photoUnder.src = url;
    photoUnder.hidden = false;
    stage.classList.add("over-photo");
  } else {
    photoUnder.hidden = true;
    photoUnder.removeAttribute("src");
    stage.classList.remove("over-photo");
  }
}

function serializeSketch() {
  const svg = stage.querySelector("svg");
  if (!svg) return svgMarkup;
  const clone = svg.cloneNode(true);
  clone.querySelectorAll(".hit").forEach((node) => node.remove());
  clone.querySelectorAll(".selected").forEach((node) => node.classList.remove("selected"));
  return clone.outerHTML;
}

function contentBox(group) {
  const tile = [...group.children].find((el) => el.tagName && el.tagName.toLowerCase() === "svg");
  if (tile) {
    try {
      const ink = tile.getBBox();
      const vb = tile.viewBox?.baseVal;
      const tw = Number(tile.getAttribute("width") || vb?.width || 64);
      const th = Number(tile.getAttribute("height") || vb?.height || 64);
      if (ink.width && ink.height && vb?.width && vb?.height) {
        return {
          x: ((ink.x - vb.x) / vb.width) * tw,
          y: ((ink.y - vb.y) / vb.height) * th,
          width: (ink.width / vb.width) * tw,
          height: (ink.height / vb.height) * th,
        };
      }
      if (ink.width && ink.height) return ink;
    } catch {
      /* use attributes */
    }
    const width = Number(tile.getAttribute("width") || 64);
    const height = Number(tile.getAttribute("height") || 64);
    return { x: 0, y: 0, width, height };
  }
  let x = Infinity;
  let y = Infinity;
  let right = -Infinity;
  let bottom = -Infinity;
  let found = false;
  for (const el of group.children) {
    if (el.classList?.contains("hit") || /\bhit\b/.test(el.getAttribute("class") || "")) continue;
    try {
      const box = el.getBBox();
      if (!box.width && !box.height) continue;
      found = true;
      x = Math.min(x, box.x);
      y = Math.min(y, box.y);
      right = Math.max(right, box.x + box.width);
      bottom = Math.max(bottom, box.y + box.height);
    } catch {
      /* ignore */
    }
  }
  if (!found) return group.getBBox();
  return { x, y, width: right - x, height: bottom - y };
}

const SCALE_MIN = 0.08;
const SCALE_MAX = 24;

function scaleXY(group) {
  const sx = Number(group.dataset.sx || group.dataset.scale || 1);
  const sy = Number(group.dataset.sy || group.dataset.scale || 1);
  return {
    sx: Number.isFinite(sx) ? sx : 1,
    sy: Number.isFinite(sy) ? sy : 1,
  };
}

function clampScale(value) {
  return Math.min(SCALE_MAX, Math.max(SCALE_MIN, value));
}

function orbitXY(group) {
  return {
    rx: Number(group.dataset.rx || 0),
    ry: Number(group.dataset.ry || 0),
  };
}

function applyXform(group) {
  const tx = Number(group.dataset.tx || 0);
  const ty = Number(group.dataset.ty || 0);
  const rot = Number(group.dataset.rot || 0);
  const { sx, sy } = scaleXY(group);
  const { rx, ry } = orbitXY(group);
  const cx = Number(group.dataset.cx);
  const cy = Number(group.dataset.cy);
  const squashX = Math.max(0.42, Math.abs(Math.cos((ry * Math.PI) / 180))) * Math.sign(Math.cos((ry * Math.PI) / 180) || 1);
  const squashY = Math.max(0.48, Math.abs(Math.cos((rx * Math.PI) / 180)));
  const skew = ry * 0.28;
  group.dataset.sx = String(sx);
  group.dataset.sy = String(sy);
  group.dataset.rx = String(rx);
  group.dataset.ry = String(ry);
  group.setAttribute(
    "transform",
    `translate(${tx + cx} ${ty + cy}) rotate(${rot}) skewX(${skew}) scale(${sx * squashX} ${sy * squashY}) translate(${-cx} ${-cy})`
  );
}

function readXform(group) {
  const box = contentBox(group);
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  group.dataset.cx = String(cx);
  group.dataset.cy = String(cy);
  const t = group.getAttribute("transform") || "";
  const translates = [...t.matchAll(/translate\(\s*([-\d.]+)[\s,]+([-\d.]+)\s*\)/g)];
  const rot = /rotate\(\s*([-\d.]+)/.exec(t);
  const sc = /scale\(\s*([-\d.]+)(?:[\s,]+([-\d.]+))?\s*\)/.exec(t);
  const sx = sc ? sc[1] : "1";
  const sy = sc && sc[2] ? sc[2] : sx;
  if (translates.length && rot) {
    group.dataset.tx = String(Number(translates[0][1]) - cx);
    group.dataset.ty = String(Number(translates[0][2]) - cy);
    group.dataset.rot = rot[1];
    group.dataset.sx = sx;
    group.dataset.sy = sy;
  } else if (translates.length) {
    group.dataset.tx = translates[0][1];
    group.dataset.ty = translates[0][2];
    group.dataset.rot = "0";
    group.dataset.sx = sx;
    group.dataset.sy = sy;
  } else {
    group.dataset.tx = "0";
    group.dataset.ty = "0";
    group.dataset.rot = "0";
    group.dataset.sx = "1";
    group.dataset.sy = "1";
  }
}

function svgPoint(event) {
  const svg = stage.querySelector("svg");
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  return point.matrixTransform(svg.getScreenCTM().inverse());
}

function photoPointFromGroup(group, x, y) {
  const svg = group.ownerSVGElement;
  const point = svg.createSVGPoint();
  point.x = x;
  point.y = y;
  const screenPt = point.matrixTransform(group.getScreenCTM());
  const frame = document.getElementById("photo").getBoundingClientRect();
  const scale = Number(getComputedStyle(screen).getPropertyValue("--s")) || 1;
  return { x: (screenPt.x - frame.left) / scale, y: (screenPt.y - frame.top) / scale };
}

function layoutGizmo(group) {
  if (!group || photoOut.hidden === false) {
    gizmo.hidden = true;
    return;
  }
  const box = contentBox(group);
  const { sx, sy } = scaleXY(group);
  const { rx, ry } = orbitXY(group);
  const rot = Number(group.dataset.rot || 0);
  const inkCx = box.x + box.width / 2;
  const inkCy = box.y + box.height / 2;
  const center = photoPointFromGroup(group, inkCx, inkCy);
  const squashX = Math.max(0.42, Math.abs(Math.cos((ry * Math.PI) / 180)));
  const squashY = Math.max(0.48, Math.abs(Math.cos((rx * Math.PI) / 180)));
  const width = Math.max(26, box.width * Math.abs(sx) * squashX + 6);
  const height = Math.max(26, box.height * Math.abs(sy) * squashY + 6);
  gizmo.hidden = false;
  gizmoBox.style.width = `${width}px`;
  gizmoBox.style.height = `${height}px`;
  gizmoBox.style.left = `${center.x - width / 2}px`;
  gizmoBox.style.top = `${center.y - height / 2}px`;
  gizmoBox.style.transform = `rotate(${rot}deg)`;
  gizmo.querySelector(".gizmo-rot").style.top = center.y - height / 2 < 36 ? `${height + 12}px` : "-34px";
  if (gizmoCube) {
    const depth = Math.max(8, Math.min(width, height) * 0.28);
    gizmoCube.style.setProperty("--d", `${depth}px`);
    gizmoCube.style.transform = `rotateX(${-rx}deg) rotateY(${ry}deg)`;
  }
}

function syncEditButtons() {
  const hasSketch = Boolean(stage.querySelector("svg"));
  editBar.hidden = sheet.dataset.mode !== "work" || !hasSketch || photoOut.hidden === false;
  undoBtn.disabled = historyIndex <= 0;
  redoBtn.disabled = historyIndex < 0 || historyIndex >= history.length - 1;
  deleteBtn.disabled = !selected;
}

function resetHistory() {
  history = [serializeSketch()];
  historyIndex = 0;
  syncEditButtons();
}

function pushHistory() {
  const next = serializeSketch();
  if (history[historyIndex] === next) {
    syncEditButtons();
    return;
  }
  history = history.slice(0, historyIndex + 1);
  history.push(next);
  if (history.length > 40) history.shift();
  historyIndex = history.length - 1;
  syncEditButtons();
}

function applySketch(svg, { reset = false } = {}) {
  svgMarkup = wrapSvg(svg);
  hideResult();
  const empty = emptyNode();
  if (empty) empty.hidden = true;
  stage.innerHTML = svgMarkup;
  const node = stage.querySelector("svg");
  refreshBuiltinStamps(node);
  if (mode === "edit") {
    node?.querySelectorAll("rect").forEach((rect) => {
      const fill = (rect.getAttribute("fill") || "").toLowerCase();
      const w = Number(rect.getAttribute("width") || 0);
      const h = Number(rect.getAttribute("height") || 0);
      if ((fill === "#ffffff" || fill === "#fff" || fill === "white") && w >= PHOTO_W - 4 && h >= PHOTO_H - 4) {
        rect.remove();
      }
    });
    svgMarkup = node?.outerHTML || svgMarkup;
  }
  selected = null;
  gizmo.hidden = true;
  if (node) bindObjects(node);
  bananaBtn.disabled = !node;
  if (reset) resetHistory();
  else syncEditButtons();
}

function selectGroup(group) {
  const svg = stage.querySelector("svg");
  if (!svg) return;
  for (const other of svg.querySelectorAll("g[data-name]")) {
    other.classList.toggle("selected", other === group);
  }
  selected = group;
  if (group) layoutGizmo(group);
  else gizmo.hidden = true;
  syncEditButtons();
}

function commitSketch() {
  svgMarkup = serializeSketch();
  pushHistory();
  saveCurrent();
}

function undo() {
  if (historyIndex <= 0) return;
  historyIndex -= 1;
  applySketch(history[historyIndex]);
  saveCurrent();
  setStatus("Undo.");
}

function redo() {
  if (historyIndex >= history.length - 1) return;
  historyIndex += 1;
  applySketch(history[historyIndex]);
  saveCurrent();
  setStatus("Redo.");
}

function closeSelected() {
  if (!selected) return;
  const group = selected;
  const name = group.getAttribute("data-name") || "object";
  const id = group.getAttribute("data-symbol") || symbolSlug(name);
  if (id) {
    const count = matchPrompt(promptEl.value).find((item) => item.id === id)?.count || 0;
    rejectSymbol(id, count);
  }
  group.remove();
  selected = null;
  gizmo.hidden = true;
  commitSketch();
  bananaBtn.disabled = !stage.querySelector("svg");
  if (!objectGroups(stage.querySelector("svg") || stage).length) clearEmptyStage();
  paintPrompt();
  setStatus(uiLang === "zh" ? `${name} 已关闭，词变红后会重新生成。` : `Closed ${name}. The word is red — Sketch will redraw it.`);
}

function groupArea(group) {
  const box = contentBox(group);
  const { sx, sy } = scaleXY(group);
  return Math.max(1, box.width * box.height * Math.abs(sx * sy));
}

function isBackdrop(group) {
  const box = contentBox(group);
  return box.width > PHOTO_W - 8 && box.height > PHOTO_H - 8;
}

function pointInGroup(group, event) {
  const svg = group.ownerSVGElement;
  if (!svg) return false;
  const point = svg.createSVGPoint();
  point.x = event.clientX;
  point.y = event.clientY;
  let local;
  try {
    local = point.matrixTransform(group.getScreenCTM().inverse());
  } catch {
    return false;
  }
  const box = contentBox(group);
  const pad = HIT_PAD;
  return (
    local.x >= box.x - pad &&
    local.x <= box.x + box.width + pad &&
    local.y >= box.y - pad &&
    local.y <= box.y + box.height + pad
  );
}

function smallestAt(event, groups) {
  const hits = groups.filter((group) => !isBackdrop(group) && pointInGroup(group, event));
  if (!hits.length) return null;
  hits.sort((a, b) => groupArea(a) - groupArea(b) || (groups.indexOf(b) - groups.indexOf(a)));
  return hits[0];
}

function addHitTarget(group) {
  group.querySelector(".hit")?.remove();
  let box;
  try {
    box = contentBox(group);
  } catch {
    return;
  }
  if (!box.width && !box.height) return;
  if (isBackdrop(group)) return;
  const pad = group.hasAttribute("data-symbol") ? 3 : HIT_PAD;
  const min = group.hasAttribute("data-symbol") ? 14 : HIT_MIN;
  const width = Math.max(min, box.width + pad * 2);
  const height = Math.max(min, box.height + pad * 2);
  const hit = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  hit.setAttribute("class", "hit");
  hit.setAttribute("x", String(box.x + box.width / 2 - width / 2));
  hit.setAttribute("y", String(box.y + box.height / 2 - height / 2));
  hit.setAttribute("width", String(width));
  hit.setAttribute("height", String(height));
  hit.setAttribute("rx", "6");
  group.insertBefore(hit, group.firstChild);
}

function objectGroups(root) {
  return [...root.querySelectorAll("g[data-name], g[id]")].filter((group) => group.closest("svg") === root);
}

function topLevelGroups(root) {
  return [...root.children].filter((node) => node.tagName && node.tagName.toLowerCase() === "g");
}

function isSceneBackdrop(group) {
  const name = `${group.getAttribute("data-name") || ""} ${group.id || ""}`.toLowerCase();
  return /background|backdrop|horizon|ground|sky|floor|wall|ocean|sea(?!l)|water/.test(name);
}

function bindObjects(root) {
  const groups = objectGroups(root);
  for (const group of groups) {
    if (!group.getAttribute("data-name")) {
      group.setAttribute("data-name", group.id || "object");
    }
    addHitTarget(group);
    readXform(group);
  }

  if (root.dataset.bound) return;
  root.dataset.bound = "1";
  root.addEventListener("pointerdown", (event) => {
    if (event.button && event.button !== 0) return;
    const group = smallestAt(event, objectGroups(root));
    if (!group) {
      selectGroup(null);
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
    selectGroup(group);
    const point = svgPoint(event);
    gesture = {
      op: "move",
      pointer: event.pointerId,
      tx: Number(group.dataset.tx || 0),
      ty: Number(group.dataset.ty || 0),
      x: point.x,
      y: point.y,
    };
    try {
      root.setPointerCapture(event.pointerId);
    } catch {
      /* synthetic events have no real pointer */
    }
  }, true);

  root.addEventListener("pointermove", (event) => {
    if (gesture?.op !== "move" || !selected || event.pointerId !== gesture.pointer) return;
    const point = svgPoint(event);
    selected.dataset.tx = String(gesture.tx + (point.x - gesture.x));
    selected.dataset.ty = String(gesture.ty + (point.y - gesture.y));
    applyXform(selected);
    layoutGizmo(selected);
  });

  root.addEventListener("pointerup", (event) => {
    if (gesture?.op !== "move" || event.pointerId !== gesture.pointer) return;
    gesture = null;
    commitSketch();
  });
}

function ensureSketchSvg() {
  let svg = stage.querySelector("svg");
  if (svg) return svg;
  const empty = emptyNode();
  if (empty) empty.remove();
  stage.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${PHOTO_W} ${PHOTO_H}"></svg>`;
  svg = stage.querySelector("svg");
  bindObjects(svg);
  return svg;
}

function captureStamps() {
  return [...stage.querySelectorAll("g[data-symbol]")].map((node) => node.cloneNode(true));
}

function nextSlot(svg) {
  const taken = objectGroups(svg).map((group) => {
    try {
      const box = contentBox(group);
      return { x: box.x + box.width / 2 + Number(group.dataset.tx || 0), y: box.y + box.height / 2 + Number(group.dataset.ty || 0) };
    } catch {
      return { x: 0, y: 0 };
    }
  });
  const cols = [80, 195, 310];
  const rows = [130, 270, 410];
  for (const y of rows) {
    for (const x of cols) {
      if (!taken.some((point) => Math.hypot(point.x - x, point.y - y) < 70)) return { x, y };
    }
  }
  return { x: 90 + (taken.length % 3) * 100, y: 140 + Math.floor(taken.length / 3) * 70 };
}

function refreshBuiltinStamps(root) {
  if (!root) return;
  for (const group of objectGroups(root)) {
    const id = group.getAttribute("data-symbol") || symbolSlug(group.getAttribute("data-name") || group.id || "");
    const def = SYMBOLS[id];
    if (!def || def.custom) continue;
    group.setAttribute("data-symbol", id);
    const xform = group.getAttribute("transform");
    [...group.children].forEach((child) => child.remove());
    group.appendChild(iconTile(def.body));
    if (xform) group.setAttribute("transform", xform);
    fitStampTile(group);
  }
}

function iconTile(body) {
  const ns = "http://www.w3.org/2000/svg";
  const tile = document.createElementNS(ns, "svg");
  tile.setAttribute("width", "64");
  tile.setAttribute("height", "64");
  tile.setAttribute("viewBox", "0 0 64 64");
  tile.setAttribute("overflow", "visible");
  const raw = String(body || "");
  tile.innerHTML = /<svg[\s>]/i.test(raw) ? raw.replace(/^[\s\S]*?<svg[^>]*>/i, "").replace(/<\/svg>\s*$/i, "") : raw;
  return tile;
}

function fitIconTile(tile) {
  let box;
  try {
    box = tile.getBBox();
  } catch {
    return;
  }
  if (!box.width || !box.height) return;
  const pad = 2;
  const x = box.x - pad;
  const y = box.y - pad;
  const w = Math.max(10, box.width + pad * 2);
  const h = Math.max(10, box.height + pad * 2);
  tile.setAttribute("viewBox", `${x} ${y} ${w} ${h}`);
  tile.setAttribute("width", String(w));
  tile.setAttribute("height", String(h));
}

function fitStampTile(group) {
  const tile = [...group.children].find((el) => el.tagName && el.tagName.toLowerCase() === "svg");
  if (!tile) return;
  const prevW = Number(tile.getAttribute("width") || 64);
  const prevH = Number(tile.getAttribute("height") || 64);
  fitIconTile(tile);
  const w = Number(tile.getAttribute("width") || prevW);
  const h = Number(tile.getAttribute("height") || prevH);
  const cx = w / 2;
  const cy = h / 2;
  const oldCx = Number(group.dataset.cx);
  const oldCy = Number(group.dataset.cy);
  if (Number.isFinite(oldCx) && Number.isFinite(oldCy)) {
    group.dataset.tx = String(Number(group.dataset.tx || 0) + oldCx - cx);
    group.dataset.ty = String(Number(group.dataset.ty || 0) + oldCy - cy);
  }
  if (!group.dataset.fitted) {
    const boost = Math.max(prevW, prevH) / Math.max(w, h, 1);
    group.dataset.sx = String(Number(group.dataset.sx || 1.65) * boost);
    group.dataset.sy = String(Number(group.dataset.sy || 1.65) * boost);
    group.dataset.fitted = "1";
  }
  group.dataset.cx = String(cx);
  group.dataset.cy = String(cy);
  applyXform(group);
}

function stampSymbol(id, { select = true, record = true } = {}) {
  const def = SYMBOLS[id];
  if (!def) return null;
  hideResult();
  if (sheet.dataset.mode !== "work") setView("work");
  const svg = ensureSketchSvg();
  const count = svg.querySelectorAll(`[data-symbol="${id}"]`).length;
  const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
  group.setAttribute("data-name", def.en);
  group.setAttribute("data-symbol", id);
  group.setAttribute("id", `sym-${id}-${count + 1}`);
  group.appendChild(iconTile(def.body));
  const slot = nextSlot(svg);
  group.dataset.rot = "0";
  group.dataset.sx = "1.65";
  group.dataset.sy = "1.65";
  svg.appendChild(group);
  fitStampTile(group);
  group.dataset.tx = String(slot.x - Number(group.dataset.cx || 32));
  group.dataset.ty = String(slot.y - Number(group.dataset.cy || 32));
  applyXform(group);
  addHitTarget(group);
  svgMarkup = serializeSketch();
  bananaBtn.disabled = false;
  if (select) selectGroup(group);
  if (record) {
    if (!history.length) resetHistory();
    else commitSketch();
  }
  syncEditButtons();
  return group;
}

function paintPrompt() {
  if (!promptMark) return;
  syncRejected(promptEl.value);
  promptMark.innerHTML = `${highlightPrompt(promptEl.value)}\n`;
  promptMark.scrollTop = promptEl.scrollTop;
}

function stampWord(id) {
  const symbol = SYMBOLS[id];
  if (!symbol) return "";
  return uiLang === "zh" ? symbol.zh : symbol.en.toLowerCase();
}

function insertSymbolWord(id) {
  const word = stampWord(id);
  if (!word) return;
  clearRejected(id);
  const start = promptEl.selectionStart ?? promptEl.value.length;
  const end = promptEl.selectionEnd ?? promptEl.value.length;
  const before = promptEl.value.slice(0, start);
  const after = promptEl.value.slice(end);
  const left = before && !/\s$/.test(before) ? " " : "";
  const right = after && !/^\s/.test(after) ? " " : "";
  promptEl.value = `${before}${left}${word}${right}${after}`;
  concept = promptEl.value.trim() || concept;
  paintPrompt();
  placeFromPrompt();
}

function stripSymbolFromPrompt(id) {
  const aliases = [...SYMBOLS[id].aliases].sort((a, b) => b.length - a.length);
  let next = promptEl.value;
  for (const alias of aliases) {
    const escaped = alias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    if (/[\u4e00-\u9fff]/.test(alias)) {
      const at = next.lastIndexOf(alias);
      if (at >= 0) {
        next = `${next.slice(0, at)}${next.slice(at + alias.length)}`.replace(/\s{2,}/g, " ").trim();
        promptEl.value = next;
        concept = next || concept;
        paintPrompt();
        return;
      }
      continue;
    }
    const pattern = new RegExp(`(^|\\s)${escaped}(?=\\s|$)`, "i");
    if (pattern.test(next)) {
      next = next.replace(pattern, "$1").replace(/\s{2,}/g, " ").trim();
      promptEl.value = next;
      concept = next || concept;
      paintPrompt();
      return;
    }
  }
}

function clearEmptyStage() {
  const svg = stage.querySelector("svg");
  if (!svg) return;
  if (objectGroups(svg).length) return;
  selected = null;
  gizmo.hidden = true;
  svgMarkup = "";
  bananaBtn.disabled = true;
  history = [];
  historyIndex = -1;
  stage.innerHTML = `<p id="empty">${sourcePhoto ? copy().emptyNext : copy().emptyBeat}</p>`;
  syncEditButtons();
}

function placeFromPrompt() {
  tagExistingByName();
  syncRejected(promptEl.value);
  const wanted = Object.fromEntries(matchPrompt(promptEl.value).map((item) => [item.id, item.count]));
  const present = [...stage.querySelectorAll("g[data-symbol]")].map((node) => node.getAttribute("data-symbol"));
  const ids = [...new Set([...Object.keys(wanted), ...present])];
  let changed = 0;
  let last = null;
  for (const id of ids) {
    const need = wanted[id] || 0;
    const nodes = [...stage.querySelectorAll(`g[data-symbol="${id}"]`)];
    while (nodes.length > need) {
      const gone = nodes.pop();
      if (selected === gone) {
        selected = null;
        gizmo.hidden = true;
      }
      gone.remove();
      changed += 1;
    }
    while (nodes.length < need) {
      if (isRejected(id)) break;
      last = stampSymbol(id, { select: false, record: false });
      nodes.push(last);
      changed += 1;
    }
  }
  if (!changed) return;
  if (last) selectGroup(last);
  if (!stage.querySelector("g[data-name], g[id]")) {
    clearEmptyStage();
    setStatus(uiLang === "zh" ? "已按提示词更新符号。" : "Symbols follow the prompt.");
    return;
  }
  svgMarkup = serializeSketch();
  bananaBtn.disabled = !svgMarkup;
  if (!history.length) resetHistory();
  else commitSketch();
  syncEditButtons();
  setStatus(uiLang === "zh" ? "已按提示词更新符号。" : "Symbols follow the prompt.");
}

function renderLibrary() {
  if (!libCats || !libGrid) return;
  const zh = uiLang === "zh";
  libCats.innerHTML = CATEGORIES.map(
    (cat) => `<button type="button" class="${cat.id === libCat ? "on" : ""}" data-cat="${cat.id}">${zh ? cat.zh : cat.en}</button>`
  ).join("");
  const ids = CATEGORIES.find((cat) => cat.id === libCat)?.ids || [];
  libGrid.innerHTML = ids
    .map((id) => {
      const symbol = SYMBOLS[id];
      return `<button type="button" class="lib-item" data-symbol="${id}" aria-label="${symbol.en}"><svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">${symbol.body}</svg><span>${zh ? symbol.zh : symbol.en}</span></button>`;
    })
    .join("");
}

function setLibOpen(on) {
  libEl.hidden = !on;
  libToggle.classList.toggle("on", on);
  if (on) renderLibrary();
}

function groupToIcon(group) {
  const clone = group.cloneNode(true);
  clone.querySelectorAll(".hit").forEach((node) => node.remove());
  clone.classList.remove("selected");
  let box = { x: 0, y: 0, width: 64, height: 64 };
  try {
    box = contentBox(group);
  } catch {
    /* overlay may not have a layout box yet */
  }
  if (!box.width && !box.height) {
    box = { x: 0, y: 0, width: 64, height: 64 };
  }
  const inner = clone.innerHTML;
  if (!inner) return "";
  const size = Math.max(box.width, box.height, 8);
  const scale = 52 / size;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  return `<g transform="translate(${32 - cx * scale} ${32 - cy * scale}) scale(${scale})">${inner}</g>`;
}

function groupLabel(group) {
  return `${group.getAttribute("data-name") || ""} ${group.id || ""}`.toLowerCase().replace(/^obj[-_]/, "");
}

function harvestGenerated() {
  const root = stage.querySelector("svg");
  if (!root) return 0;
  const groups = topLevelGroups(root).filter((group) => !isSceneBackdrop(group));
  const pending = unknownWords(promptEl.value);
  const saved = [];

  function saveGroup(group, name) {
    const label = String(name || groupLabel(group).replace(/[-_]+/g, " ")).trim() || "symbol";
    const body = groupToIcon(group);
    if (!body) return null;
    const aliases = [...new Set(label.toLowerCase().split(/[\s/_-]+/).filter(Boolean))];
    const def = addCustomSymbol({
      id: symbolSlug(label),
      en: label,
      aliases,
      body,
    });
    group.setAttribute("data-symbol", def.id);
    group.setAttribute("data-name", def.en);
    saved.push(def);
    clearRejected(def.id);
    return def;
  }

  for (const word of pending) {
    const group = groups.find((item) => !item.getAttribute("data-symbol") && groupLabel(item).includes(word));
    if (group) saveGroup(group, word);
  }
  for (const group of groups) {
    if (group.getAttribute("data-symbol")) continue;
    const raw = groupLabel(group).replace(/[-_]+/g, " ").trim();
    if (!raw || /^object\d*$/i.test(raw) || /^g\d+$/i.test(raw)) {
      const word = pending.find((item) => !SYMBOLS[symbolSlug(item)]?.custom && !groups.some((g) => g.getAttribute("data-symbol") === symbolSlug(item)));
      if (word) saveGroup(group, word);
      continue;
    }
    saveGroup(group, raw);
  }
  if (!saved.length) return 0;
  renderLibrary();
  paintPrompt();
  fetch("/api/sd/library", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbols: saved }),
  }).catch(() => {});
  return saved.length;
}

function tagExistingByName() {
  const root = stage.querySelector("svg");
  if (!root) return;
  const words = unknownWords(promptEl.value);
  const groups = topLevelGroups(root).filter((group) => !group.getAttribute("data-symbol") && !isSceneBackdrop(group));
  if (!groups.length) return;
  let tagged = 0;
  for (const group of groups) {
    const label = groupLabel(group);
    const word = words.find((item) => item.length >= 2 && (label === item || label.split(/[\s/_-]+/).includes(item) || label.includes(item)));
    if (!word && !SYMBOLS[symbolSlug(label)]) continue;
    const body = groupToIcon(group);
    if (!body) continue;
    const name = word || label.replace(/[-_]+/g, " ").trim();
    const def = addCustomSymbol({
      id: symbolSlug(name),
      en: name,
      aliases: name.toLowerCase().split(/[\s/_-]+/),
      body,
    });
    group.setAttribute("data-symbol", def.id);
    group.setAttribute("data-name", def.en);
    clearRejected(def.id);
    tagged += 1;
  }
  if (tagged) {
    renderLibrary();
    paintPrompt();
  }
}

function showSketch(svg) {
  const stamps = captureStamps();
  applySketch(svg, { reset: true });
  const root = stage.querySelector("svg");
  if (root && stamps.length) {
    for (const stamp of stamps) root.appendChild(stamp);
    bindObjects(root);
    svgMarkup = serializeSketch();
    resetHistory();
  }
  const added = harvestGenerated();
  tagExistingByName();
  paintPrompt();
  barSub.textContent = mode === "edit" ? "Move, resize, rotate" : "Sketch ready";
  setStatus(
    added
      ? uiLang === "zh"
        ? `已保存 ${added} 个符号到库。`
        : `Saved ${added} symbol${added === 1 ? "" : "s"} to the library.`
      : mode === "edit"
        ? "Move, resize, or rotate, then Banana."
        : "Move, resize, or rotate, then Nano Banana."
  );
}

function showPhoto(url) {
  photoOut.src = url;
  photoOut.hidden = false;
  gizmo.hidden = true;
  editBar.hidden = true;
  barSub.textContent = "Photograph";
  bananaBtn.disabled = !svgMarkup;
}

function renderThumbs() {
  const items = scenes
    .map((scene) => {
      const src = scene.photo || scene.source || scene.sketch;
      if (!src) return "";
      const on = scene.concept === concept ? " active" : "";
      return `<div class="thumb-wrap"><button type="button" class="thumb${on}" data-id="${scene.id}" aria-label="${scene.concept}"><img src="${src}?v=${scene.updated || 1}" alt="" /></button><button type="button" class="thumb-del" data-id="${scene.id}" aria-label="Remove ${scene.concept}">×</button></div>`;
    })
    .join("");
  thumbs.innerHTML = `<button type="button" class="thumb blank" data-id="blank" aria-label="Empty">Empty</button>${items}<button type="button" class="add" id="load" aria-label="Add a photo">+</button>`;
}

async function refreshScenes() {
  const response = await fetch("/api/sd/scenes");
  const body = await response.json().catch(() => ({}));
  scenes = body.scenes || [];
  renderThumbs();
}

async function saveCurrent(image) {
  if (!concept || (!svgMarkup && !sourcePhoto)) return;
  await fetch("/api/sd/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      concept,
      svg: serializeSketch(),
      image,
      source: sourcePhoto.startsWith("data:") ? sourcePhoto : undefined,
    }),
  });
  await refreshScenes();
}

function wrapDeg(value) {
  return ((Number(value) + 180) % 360 + 360) % 360 - 180;
}

function turn3d(v, rot, rx, ry) {
  const a = (ry * Math.PI) / 180;
  const b = (-rx * Math.PI) / 180;
  const c = (rot * Math.PI) / 180;
  const x1 = v[0] * Math.cos(a) + v[2] * Math.sin(a);
  const z1 = -v[0] * Math.sin(a) + v[2] * Math.cos(a);
  const y2 = v[1] * Math.cos(b) - z1 * Math.sin(b);
  const z2 = v[1] * Math.sin(b) + z1 * Math.cos(b);
  return [x1 * Math.cos(c) - y2 * Math.sin(c), x1 * Math.sin(c) + y2 * Math.cos(c), z2];
}

function pitchView(rx) {
  const x = wrapDeg(rx);
  const a = Math.abs(x);
  if (a < 12) return "level camera, not from above or below";
  if (x > 0) return `seen from above (pitch ${Math.round(a)}°) — top of the object is visible`;
  return `seen from below (pitch ${Math.round(a)}°) — underside is visible`;
}

function planeSpin(rot) {
  const r = wrapDeg(rot);
  const a = Math.abs(r);
  if (a < 8) return "upright in the picture plane";
  const way = r > 0 ? "clockwise" : "counterclockwise";
  if (a < 50) return `leaning ${way} ${Math.round(a)}°`;
  if (a < 130) {
    const top = r > 0 ? "right" : "left";
    return `on its side, spun ${way} ${Math.round(a)}° so the original top points toward the ${top} of the frame`;
  }
  return `upside down (${way} ${Math.round(a)}°) — keep it inverted, do not flip it back upright`;
}

function groupInkBox(group) {
  const hit = group.querySelector(".hit");
  if (hit) hit.setAttribute("display", "none");
  let box;
  try {
    box = group.getBBox();
  } catch {
    box = { x: 0, y: 0, width: 0, height: 0 };
  }
  if (hit) hit.removeAttribute("display");
  return box;
}

const DRAWN_FACING = {
  dog: [-1, 0, 0],
  fish: [-1, 0, 0],
  horse: [-1, 0, 0],
  rabbit: [-1, 0, 0],
  whale: [-1, 0, 0],
  bird: [1, 0, 0],
  plane: [1, 0, 0],
  car: [-1, 0, 0],
  bus: [1, 0, 0],
  bike: [1, 0, 0],
  boat: [1, 0, 0],
  cat: [0, 0, 1],
  person: [0, 0, 1],
  man: [0, 0, 1],
  woman: [0, 0, 1],
  child: [0, 0, 1],
};

function compassWord(deg) {
  const dirs = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"];
  return dirs[Math.round((((deg % 360) + 360) % 360) / 45) % 8];
}

function stampRecords() {
  const root = stage.querySelector("svg");
  if (!root) return [];
  const w = root.viewBox?.baseVal?.width || PHOTO_W;
  const h = root.viewBox?.baseVal?.height || PHOTO_H;
  return topLevelGroups(root)
    .filter((group) => !isSceneBackdrop(group) && !isBackdrop(group))
    .map((group, index) => {
      const box = groupInkBox(group);
      const rot = wrapDeg(group.dataset.rot || 0);
      const ry = wrapDeg(group.dataset.ry || 0);
      const rx = wrapDeg(group.dataset.rx || 0);
      const id = group.getAttribute("data-symbol") || symbolSlug(group.getAttribute("data-name") || "");
      const name = group.getAttribute("data-name") || group.id || "object";
      const drawn = DRAWN_FACING[id];
      return {
        n: index + 1,
        name,
        id,
        x: (box.x + box.width / 2) / w,
        y: (box.y + box.height / 2) / h,
        wp: box.width / w,
        hp: box.height / h,
        rot,
        ry,
        rx,
        flipped: Math.cos((ry * Math.PI) / 180) < 0,
        face: drawn ? turn3d(drawn, rot, rx, ry) : null,
      };
    });
}

function facingLine(item) {
  const bits = [];
  if (item.face) {
    const [fx, fy, fz] = item.face;
    const flat = Math.hypot(fx, fy);
    if (fz > 0.92) bits.push("facing the camera");
    else if (fz < -0.92) bits.push("facing away from the camera, we see its back");
    else {
      bits.push(`its head points to the ${compassWord((Math.atan2(fy, fx) * 180) / Math.PI)} of the frame`);
      if (fz > 0.35 && flat > 0.2) bits.push("angled partly toward the camera");
      if (fz < -0.35 && flat > 0.2) bits.push("angled partly away from the camera");
    }
  } else if (item.flipped) {
    bits.push("drawn mirrored — keep that left-right facing exactly as in the sketch");
  } else {
    bits.push("keep the left-right facing exactly as drawn");
  }
  const spin = Math.abs(item.rot);
  if (spin > 130) bits.push("upside down");
  else if (spin > 50) bits.push("lying on its side");
  if (Math.abs(item.rx) > 12) bits.push(pitchView(item.rx));
  return bits.join(", ");
}

function layoutNotes() {
  const items = stampRecords();
  if (!items.length) return "";
  const counts = {};
  for (const item of items) counts[item.name] = (counts[item.name] || 0) + 1;
  const tally = Object.entries(counts)
    .map(([name, n]) => `${n} ${name}`)
    .join(", ");
  const rows = items.map((item) => {
    const x = Math.round(item.x * 100);
    const y = Math.round(item.y * 100);
    const w = Math.round(item.wp * 100);
    const h = Math.round(item.hp * 100);
    return `The ${item.name} sits ${x} percent from the left and ${y} percent from the top, about ${w} by ${h} percent of the frame, ${facingLine(item)}`;
  });
  const rel = [];
  for (let i = 0; i < items.length; i += 1) {
    for (let j = i + 1; j < items.length; j += 1) {
      const a = items[i];
      const b = items[j];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      if (Math.hypot(dx, dy) < 0.08) continue;
      const horiz = Math.abs(dx) > 0.08 ? (dx < 0 ? "left of" : "right of") : "";
      const vert = Math.abs(dy) > 0.08 ? (dy < 0 ? "above" : "below") : "";
      const where = [vert, horiz].filter(Boolean).join(" and ");
      if (where) rel.push(`the ${a.name} is ${where} the ${b.name}`);
    }
  }
  return [
    `Keep ${tally} in these places:`,
    rows.join(". ") + ".",
    rel.length ? rel.join(". ") + "." : "",
  ]
    .filter(Boolean)
    .join(" ");
}

function poseNotes() {
  const items = stampRecords();
  if (!items.length) return "";
  return items.map((item) => `The ${item.name} is ${facingLine(item)}.`).join(" ");
}

function stampPoint(group, x, y) {
  const svg = group.ownerSVGElement;
  const point = svg.createSVGPoint();
  point.x = x;
  point.y = y;
  const screen = point.matrixTransform(group.getScreenCTM());
  return screen.matrixTransform(svg.getScreenCTM().inverse());
}

function addBananaBoxes(clone, live) {
  const liveGroups = topLevelGroups(live).filter((group) => !isSceneBackdrop(group) && !isBackdrop(group));
  const ns = "http://www.w3.org/2000/svg";
  const layer = document.createElementNS(ns, "g");
  layer.setAttribute("class", "banana-box");
  layer.setAttribute("fill", "none");
  layer.setAttribute("stroke", "#3a8f4a");
  layer.setAttribute("stroke-width", "1.8");
  layer.setAttribute("stroke-linejoin", "round");
  liveGroups.forEach((liveGroup) => {
    const box = contentBox(liveGroup);
    if (!box.width || !box.height) return;
    const rx = Number(liveGroup.dataset.rx || 0);
    const ry = Number(liveGroup.dataset.ry || 0);
    const rot = Number(liveGroup.dataset.rot || 0);
    const front = [
      stampPoint(liveGroup, box.x, box.y),
      stampPoint(liveGroup, box.x + box.width, box.y),
      stampPoint(liveGroup, box.x + box.width, box.y + box.height),
      stampPoint(liveGroup, box.x, box.y + box.height),
    ];
    const span = Math.min(Math.hypot(front[1].x - front[0].x, front[1].y - front[0].y), Math.hypot(front[3].x - front[0].x, front[3].y - front[0].y));
    const depth = Math.max(8, span * 0.4);
    const normal = turn3d([0, 0, 1], rot, rx, ry);
    const ox = -normal[0] * depth;
    const oy = -normal[1] * depth;
    const back = front.map((pt) => ({ x: pt.x + ox, y: pt.y + oy }));
    const fmt = (pt) => `${pt.x} ${pt.y}`;
    const rear = document.createElementNS(ns, "path");
    rear.setAttribute("d", `M${fmt(back[0])}L${fmt(back[1])}L${fmt(back[2])}L${fmt(back[3])}Z`);
    rear.setAttribute("stroke-opacity", "0.4");
    const ribs = document.createElementNS(ns, "path");
    ribs.setAttribute("d", front.map((pt, i) => `M${fmt(pt)}L${fmt(back[i])}`).join(""));
    ribs.setAttribute("stroke-opacity", "0.65");
    const face = document.createElementNS(ns, "path");
    face.setAttribute("d", `M${fmt(front[0])}L${fmt(front[1])}L${fmt(front[2])}L${fmt(front[3])}Z`);
    face.setAttribute("stroke-width", "2.3");
    layer.appendChild(rear);
    layer.appendChild(ribs);
    layer.appendChild(face);
  });
  clone.appendChild(layer);
}

async function svgToPng(svg, { clear = true, boxes = false } = {}) {
  const live = stage.querySelector("svg");
  let markup = wrapSvg(svg);
  if (live) {
    const clone = live.cloneNode(true);
    clone.querySelectorAll(".hit").forEach((node) => node.remove());
    if (boxes) addBananaBoxes(clone, live);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("width", String(PHOTO_W));
    clone.setAttribute("height", String(PHOTO_H));
    if (!clone.getAttribute("viewBox")) clone.setAttribute("viewBox", `0 0 ${PHOTO_W} ${PHOTO_H}`);
    markup = new XMLSerializer().serializeToString(clone);
  }
  const blob = new Blob([markup], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = PHOTO_W * 2;
    canvas.height = PHOTO_H * 2;
    const ctx = canvas.getContext("2d");
    if (clear) {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/png");
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function compositeEdit() {
  const canvas = document.createElement("canvas");
  canvas.width = PHOTO_W * 2;
  canvas.height = PHOTO_H * 2;
  const ctx = canvas.getContext("2d");
  if (sourcePhoto) {
    const photo = new Image();
    photo.src = sourcePhoto;
    await photo.decode();
    const scale = Math.max(canvas.width / photo.naturalWidth, canvas.height / photo.naturalHeight);
    const w = photo.naturalWidth * scale;
    const h = photo.naturalHeight * scale;
    ctx.drawImage(photo, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
  }
  const overlay = await svgToPng(serializeSketch(), { clear: false, boxes: true });
  const mark = new Image();
  mark.src = overlay;
  await mark.decode();
  ctx.drawImage(mark, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/png");
}

function dataPayload(dataUrl) {
  return {
    image: String(dataUrl).replace(/^data:[^;]+;base64,/, ""),
    mime: String(dataUrl).match(/^data:([^;]+);/)?.[1] || "image/png",
  };
}

function publicSceneUrl(src) {
  if (!src) return "";
  let path = String(src).split("?")[0];
  if (path.startsWith("http://") || path.startsWith("https://")) {
    try {
      path = new URL(path).pathname;
    } catch {
      return "";
    }
  }
  return path.startsWith("/sd-scenes/") ? path : "";
}

async function compactPhoto(src) {
  const image = new Image();
  image.src = src;
  await image.decode();
  const canvas = document.createElement("canvas");
  canvas.width = PHOTO_W;
  canvas.height = PHOTO_H;
  const ctx = canvas.getContext("2d");
  const scale = Math.max(canvas.width / image.naturalWidth, canvas.height / image.naturalHeight);
  const w = image.naturalWidth * scale;
  const h = image.naturalHeight * scale;
  ctx.drawImage(image, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
  return dataPayload(canvas.toDataURL("image/jpeg", 0.72));
}

async function imagePayload(src) {
  if (!src) throw new Error("Add a photo first.");
  if (src.startsWith("data:")) return dataPayload(src);
  const response = await fetch(src);
  if (!response.ok) throw new Error("Could not read the photo.");
  const blob = await response.blob();
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read the photo."));
    reader.readAsDataURL(blob);
  });
  return dataPayload(dataUrl);
}

async function coverFile(file) {
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = PHOTO_W * 2;
    canvas.height = PHOTO_H * 2;
    const ctx = canvas.getContext("2d");
    const scale = Math.max(canvas.width / image.naturalWidth, canvas.height / image.naturalHeight);
    const w = image.naturalWidth * scale;
    const h = image.naturalHeight * scale;
    ctx.drawImage(image, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
    return canvas.toDataURL("image/jpeg", 0.9);
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function runSketch() {
  const onPhoto = Boolean(sourcePhoto);
  concept = promptEl.value.trim() || concept || (onPhoto ? "Photo edit" : "");
  placeFromPrompt();
  const rest = leftoverPrompt(promptEl.value);
  const hasStamps = Boolean(stage.querySelector("g[data-symbol]"));
  if (!concept && !onPhoto && !hasStamps) {
    setStatus(copy().describeFirst, true);
    return;
  }
  if (!rest && hasStamps) {
    mode = onPhoto ? "edit" : "sketch";
    setStatus(uiLang === "zh" ? "符号已在画面上，无需重新生成。" : "Library symbols are already on the canvas.");
    bananaBtn.disabled = false;
    return;
  }
  mode = onPhoto ? "edit" : "sketch";
  setBusy(true, onPhoto ? "Marking" : "Sketching");
  setStatus(onPhoto ? copy().drawing : copy().laying);
  try {
    const placed = [...stage.querySelectorAll("g[data-symbol]")].map((node) => node.getAttribute("data-name")).filter(Boolean);
    const bodyIn = {
      concept: rest || concept,
      keep: placed,
      force: true,
    };
    if (onPhoto) {
      const packed = await compactPhoto(sourcePhoto);
      bodyIn.image = packed.image;
      bodyIn.mime = packed.mime;
    }
    const response = await fetch(onPhoto ? "/api/sd/overlay" : "/api/sd/sketch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(bodyIn),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || "Sketch failed");
    showSketch(body.svg);
    await refreshScenes();
  } catch (error) {
    setStatus(error.message || "Sketch failed", true);
  } finally {
    setBusy(false);
  }
}

async function runBanana() {
  if (!svgMarkup) return;
  concept = promptEl.value.trim() || concept;
  const edit = promptEl.value.trim();
  setBusy(true, "Nano Banana");
  setStatus("Making the photograph…");
  barSub.textContent = "Nano Banana";
  try {
    const pose = poseNotes();
    const layout = layoutNotes();
    const payload = {
      concept,
      prompt: edit,
      pose,
      layout,
      svg: serializeSketch(),
      mode,
      force: true,
    };
    if (mode === "edit" && sourcePhoto) {
      const original = await imagePayload(sourcePhoto);
      const overlay = dataPayload(await compositeEdit());
      payload.image = original.image;
      payload.mime = original.mime;
      payload.overlay = overlay.image;
      payload.overlayMime = overlay.mime;
    } else {
      const dataUrl = await svgToPng(serializeSketch(), { boxes: true });
      const packed = dataPayload(dataUrl);
      payload.image = packed.image;
      payload.mime = packed.mime;
    }
    const response = await fetch("/api/sd/banana", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || "Nano Banana failed");
    showPhoto(body.image);
    setStatus(copy().panelFrom);
    await refreshScenes();
  } catch (error) {
    setStatus(error.message || "Nano Banana failed", true);
    barSub.textContent = mode === "edit" ? "Move, resize, rotate" : "Sketch ready";
  } finally {
    setBusy(false);
  }
}

function openBlank() {
  mode = "sketch";
  concept = "";
  promptEl.value = "";
  clearRejected();
  promptEl.placeholder = copy().placeholder;
  paintPrompt();
  setUnderlay("");
  hideResult();
  svgMarkup = "";
  history = [];
  historyIndex = -1;
  selected = null;
  bananaBtn.disabled = true;
  stage.innerHTML = `<p id="empty">${copy().emptyBeat}</p>`;
  setView("work");
  barSub.textContent = "";
}

async function openScene(scene) {
  clearRejected();
  promptEl.value = scene.concept || "";
  paintPrompt();
  concept = scene.concept || "";
  const photoUrl = scene.photo || scene.source;
  mode = photoUrl ? "edit" : "sketch";
  promptEl.placeholder = photoUrl ? copy().placeholderNext : copy().placeholder;
  setUnderlay(photoUrl || "");
  hideResult();
  renderThumbs();
  setView("work");
  svgMarkup = "";
  history = [];
  historyIndex = -1;
  selected = null;
  bananaBtn.disabled = true;
  if (photoUrl) {
    stage.innerHTML = `<p id="empty">${copy().emptyNext}</p>`;
    syncEditButtons();
    barSub.textContent = "";
    return;
  }
  if (scene.sketch) {
    const svg = await fetch(`${scene.sketch}?v=${scene.updated || 1}`).then((response) => response.text());
    showSketch(svg);
    barSub.textContent = "";
    return;
  }
  stage.innerHTML = `<p id="empty">${copy().emptyBeat}</p>`;
  syncEditButtons();
  barSub.textContent = "";
}

thumbs.addEventListener("click", async (event) => {
  const add = event.target.closest("#load");
  if (add) {
    fileEl.click();
    return;
  }
  const remove = event.target.closest(".thumb-del");
  if (remove) {
    event.stopPropagation();
    const id = remove.dataset.id;
    const response = await fetch("/api/sd/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    if (!response.ok) return;
    scenes = scenes.filter((item) => item.id !== id);
    renderThumbs();
    return;
  }
  const thumb = event.target.closest(".thumb");
  if (!thumb) return;
  if (thumb.dataset.id === "blank") {
    openBlank();
    return;
  }
  const scene = scenes.find((item) => item.id === thumb.dataset.id);
  if (!scene) return;
  await openScene(scene);
});

fileEl.addEventListener("change", async () => {
  const file = fileEl.files?.[0];
  fileEl.value = "";
  if (!file) return;
  mode = "edit";
  hideResult();
  svgMarkup = "";
  bananaBtn.disabled = true;
  const dataUrl = await coverFile(file);
  setUnderlay(dataUrl);
  concept = promptEl.value.trim() || file.name.replace(/\.[^.]+$/, "").replace(/[-_]+/g, " ");
  if (!promptEl.value.trim()) promptEl.value = "";
  promptEl.placeholder = "What should change?";
  history = [];
  historyIndex = -1;
  selected = null;
  stage.innerHTML = `<p id="empty">${copy().emptyNext}</p>`;
  syncEditButtons();
  setView("work");
  barSub.textContent = "";
  await saveCurrent();
});

document.getElementById("back").addEventListener("click", () => {
  if (!photoOut.hidden) {
    hideResult();
    layoutGizmo(selected);
    syncEditButtons();
    barSub.textContent = "";
    return;
  }
  setView("open");
  selectGroup(null);
  stage.innerHTML = `<p id="empty">Choose a photo, or start empty.</p>`;
  setUnderlay("");
  hideResult();
  bananaBtn.disabled = true;
  renderThumbs();
});

undoBtn.addEventListener("click", undo);
redoBtn.addEventListener("click", redo);
deleteBtn.addEventListener("click", closeSelected);

window.addEventListener("keydown", (event) => {
  if (event.target === promptEl) return;
  const key = event.key.toLowerCase();
  if ((event.metaKey || event.ctrlKey) && key === "z") {
    event.preventDefault();
    if (event.shiftKey) redo();
    else undo();
  } else if ((event.metaKey || event.ctrlKey) && key === "y") {
    event.preventDefault();
    redo();
  } else if ((event.key === "Backspace" || event.key === "Delete") && selected) {
    event.preventDefault();
    closeSelected();
  }
});

function onTransformMove(event) {
  if (!selected || !gesture || event.pointerId !== gesture.pointer) return;
  const point = svgPoint(event);
  if (gesture.op === "move") {
    selected.dataset.tx = String(gesture.tx + (point.x - gesture.x));
    selected.dataset.ty = String(gesture.ty + (point.y - gesture.y));
  } else if (gesture.op === "scale") {
    const rad = (-gesture.rot * Math.PI) / 180;
    const dx = point.x - gesture.cx;
    const dy = point.y - gesture.cy;
    const localX = dx * Math.cos(rad) - dy * Math.sin(rad);
    const localY = dx * Math.sin(rad) + dy * Math.cos(rad);
    const uniform = gesture.shift;
    let sx = gesture.sx;
    let sy = gesture.sy;
    if (gesture.edge === "e" || gesture.edge === "w" || gesture.corner) {
      const signX = gesture.edge === "w" || gesture.corner === "nw" || gesture.corner === "sw" ? -1 : 1;
      sx = clampScale((signX * localX) / gesture.halfW);
    }
    if (gesture.edge === "n" || gesture.edge === "s" || gesture.corner) {
      const signY = gesture.edge === "n" || gesture.corner === "nw" || gesture.corner === "ne" ? -1 : 1;
      sy = clampScale((signY * localY) / gesture.halfH);
    }
    if (uniform && gesture.corner) {
      const start = Math.max(8, Math.hypot(gesture.halfW * gesture.sx, gesture.halfH * gesture.sy));
      const factor = Math.hypot(localX, localY) / start;
      sx = clampScale(gesture.sx * factor);
      sy = clampScale(gesture.sy * factor);
    }
    selected.dataset.sx = String(sx);
    selected.dataset.sy = String(sy);
  } else if (gesture.op === "rotate") {
    const ang = Math.atan2(point.y - gesture.cy, point.x - gesture.cx);
    selected.dataset.rot = String(gesture.rot + ((ang - gesture.ang) * 180) / Math.PI);
  } else if (gesture.op === "orbit") {
    selected.dataset.ry = String(gesture.ry + (point.x - gesture.x) * 0.7);
    selected.dataset.rx = String(Math.max(-70, Math.min(70, gesture.rx + (point.y - gesture.y) * 0.45)));
  }
  applyXform(selected);
  layoutGizmo(selected);
}

function onTransformUp(event) {
  if (!gesture || event.pointerId !== gesture.pointer) return;
  gesture = null;
  commitSketch();
}

gizmo.addEventListener("pointerdown", (event) => {
  if (!selected) return;
  if (event.target.closest(".gizmo-close")) {
    event.preventDefault();
    event.stopPropagation();
    closeSelected();
    return;
  }
  const handle = event.target.closest("[data-op]");
  const op = handle?.dataset.op || "move";
  event.preventDefault();
  event.stopPropagation();
  const point = svgPoint(event);
  const box = contentBox(selected);
  const { sx, sy } = scaleXY(selected);
  const cx = Number(selected.dataset.cx) + Number(selected.dataset.tx || 0);
  const cy = Number(selected.dataset.cy) + Number(selected.dataset.ty || 0);
  gesture = {
    op,
    pointer: event.pointerId,
    tx: Number(selected.dataset.tx || 0),
    ty: Number(selected.dataset.ty || 0),
    rot: Number(selected.dataset.rot || 0),
    rx: Number(selected.dataset.rx || 0),
    ry: Number(selected.dataset.ry || 0),
    sx,
    sy,
    corner: handle?.dataset.corner || "",
    edge: handle?.dataset.edge || "",
    shift: event.shiftKey,
    x: point.x,
    y: point.y,
    cx,
    cy,
    halfW: Math.max(4, box.width / 2),
    halfH: Math.max(4, box.height / 2),
    dist: Math.max(8, Math.hypot(point.x - cx, point.y - cy)),
    ang: Math.atan2(point.y - cy, point.x - cx),
  };
  try {
    (handle || gizmoBox).setPointerCapture(event.pointerId);
  } catch {
    /* synthetic events have no real pointer */
  }
});

document.getElementById("photo")?.addEventListener(
  "wheel",
  (event) => {
    if (!selected || photoOut.hidden === false) return;
    event.preventDefault();
    const { sx, sy } = scaleXY(selected);
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    selected.dataset.sx = String(clampScale(sx * factor));
    selected.dataset.sy = String(clampScale(sy * factor));
    applyXform(selected);
    layoutGizmo(selected);
  },
  { passive: false }
);

window.addEventListener("pointermove", onTransformMove);
window.addEventListener("pointerup", onTransformUp);
window.addEventListener("pointercancel", onTransformUp);

sketchBtn.addEventListener("click", runSketch);
bananaBtn.addEventListener("click", runBanana);
promptEl.addEventListener("input", () => {
  concept = promptEl.value.trim() || concept;
  paintPrompt();
  clearTimeout(typeTimer);
  typeTimer = window.setTimeout(placeFromPrompt, 140);
});
promptEl.addEventListener("scroll", () => {
  if (promptMark) promptMark.scrollTop = promptEl.scrollTop;
});
promptEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    runSketch();
  }
});

libToggle.addEventListener("click", () => {
  if (sheet.dataset.mode !== "work") setView("work");
  setLibOpen(libEl.hidden);
});
libClose.addEventListener("click", () => setLibOpen(false));
libCats.addEventListener("click", (event) => {
  const button = event.target.closest("[data-cat]");
  if (!button) return;
  libCat = button.dataset.cat;
  renderLibrary();
});
libGrid.addEventListener("click", (event) => {
  const button = event.target.closest("[data-symbol]");
  if (!button) return;
  insertSymbolWord(button.dataset.symbol);
});

window.addEventListener("message", (event) => {
  if (event.data?.type === "lang") applyLang(event.data.lang);
});
applyLang(uiLang);
refreshScenes().then(() => setView("open"));
fetch("/api/sd/library")
  .then((response) => response.json())
  .then((body) => {
    loadCustomSymbols(body.symbols || []);
    renderLibrary();
    paintPrompt();
  })
  .catch(() => {});
