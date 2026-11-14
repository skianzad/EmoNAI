const screen = document.querySelector(".screen");
const sheet = document.getElementById("sheet");
const stage = document.getElementById("stage");
const photoUnder = document.getElementById("photo-under");
const photoOut = document.getElementById("photo-out");
const busy = document.getElementById("busy");
const busyCopy = document.getElementById("busy-copy");
const promptEl = document.getElementById("prompt");
const barTitle = document.getElementById("bar-title");
const barSub = document.getElementById("bar-sub");
const sketchBtn = document.getElementById("sketch");
const bananaBtn = document.getElementById("banana");
const thumbs = document.getElementById("thumbs");
const fileEl = document.getElementById("file");
const gizmo = document.getElementById("gizmo");
const gizmoBox = gizmo.querySelector(".gizmo-box");
const editBar = document.getElementById("edit-bar");
const undoBtn = document.getElementById("undo");
const redoBtn = document.getElementById("redo");
const deleteBtn = document.getElementById("delete-obj");

const PHOTO_W = 390;
const PHOTO_H = 520;
const HIT_PAD = 36;
const HIT_MIN = 88;

let svgMarkup = "";
let concept = "";
let scenes = [];
let sourcePhoto = "";
let mode = "sketch";
let selected = null;
let gesture = null;
let history = [];
let historyIndex = -1;

function fit() {
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

function setView(view) {
  sheet.dataset.mode = view;
  if (view === "open") {
    barTitle.textContent = "Pick a photo";
    barSub.textContent = "";
    gizmo.hidden = true;
    editBar.hidden = true;
    hideResult();
    selected = null;
  } else {
    barTitle.textContent = "Sketch Director";
  }
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
  let x = Infinity;
  let y = Infinity;
  let right = -Infinity;
  let bottom = -Infinity;
  let found = false;
  for (const el of group.children) {
    if (el.classList?.contains("hit")) continue;
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

function applyXform(group) {
  const tx = Number(group.dataset.tx || 0);
  const ty = Number(group.dataset.ty || 0);
  const rot = Number(group.dataset.rot || 0);
  const scale = Number(group.dataset.scale || 1);
  const cx = Number(group.dataset.cx);
  const cy = Number(group.dataset.cy);
  group.setAttribute(
    "transform",
    `translate(${tx + cx} ${ty + cy}) rotate(${rot}) scale(${scale}) translate(${-cx} ${-cy})`
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
  const sc = /scale\(\s*([-\d.]+)/.exec(t);
  if (translates.length && rot) {
    group.dataset.tx = String(Number(translates[0][1]) - cx);
    group.dataset.ty = String(Number(translates[0][2]) - cy);
    group.dataset.rot = rot[1];
    group.dataset.scale = sc ? sc[1] : "1";
  } else if (translates.length) {
    group.dataset.tx = translates[0][1];
    group.dataset.ty = translates[0][2];
    group.dataset.rot = "0";
    group.dataset.scale = "1";
  } else {
    group.dataset.tx = "0";
    group.dataset.ty = "0";
    group.dataset.rot = "0";
    group.dataset.scale = "1";
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
  const scale = Number(group.dataset.scale || 1);
  const rot = Number(group.dataset.rot || 0);
  const center = photoPointFromGroup(group, Number(group.dataset.cx), Number(group.dataset.cy));
  const width = Math.max(36, box.width * scale);
  const height = Math.max(36, box.height * scale);
  gizmo.hidden = false;
  gizmoBox.style.width = `${width}px`;
  gizmoBox.style.height = `${height}px`;
  gizmoBox.style.left = `${center.x - width / 2}px`;
  gizmoBox.style.top = `${center.y - height / 2}px`;
  gizmoBox.style.transform = `rotate(${rot}deg)`;
  gizmo.querySelector(".gizmo-rot").style.top = center.y - height / 2 < 36 ? `${height + 12}px` : "-34px";
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

function deleteSelected() {
  if (!selected) return;
  const name = selected.getAttribute("data-name") || "object";
  selected.remove();
  selected = null;
  gizmo.hidden = true;
  commitSketch();
  bananaBtn.disabled = !stage.querySelector("svg");
  setStatus(`Deleted ${name}.`);
}

function groupArea(group) {
  const box = contentBox(group);
  const scale = Number(group.dataset.scale || 1);
  return Math.max(1, box.width * box.height * scale * scale);
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
  const width = Math.max(HIT_MIN, box.width + HIT_PAD * 2);
  const height = Math.max(HIT_MIN, box.height + HIT_PAD * 2);
  const hit = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  hit.setAttribute("class", "hit");
  hit.setAttribute("x", String(box.x + box.width / 2 - width / 2));
  hit.setAttribute("y", String(box.y + box.height / 2 - height / 2));
  hit.setAttribute("width", String(width));
  hit.setAttribute("height", String(height));
  hit.setAttribute("rx", "14");
  group.insertBefore(hit, group.firstChild);
}

function bindObjects(root) {
  const groups = [...root.querySelectorAll("g[data-name], g[id]")].filter((group) => group.closest("svg") === root);
  for (const group of groups) {
    if (!group.getAttribute("data-name")) {
      group.setAttribute("data-name", group.id || "object");
    }
    addHitTarget(group);
    readXform(group);
  }

  root.addEventListener("pointerdown", (event) => {
    if (event.button && event.button !== 0) return;
    const group = smallestAt(event, groups);
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

function showSketch(svg) {
  applySketch(svg, { reset: true });
  barSub.textContent = mode === "edit" ? "Move, resize, rotate" : "Sketch ready";
  setStatus(mode === "edit" ? "Move, resize, or rotate, then Banana." : "Move, resize, or rotate, then Nano Banana.");
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
  const response = await fetch("/api/scenes");
  const body = await response.json().catch(() => ({}));
  scenes = body.scenes || [];
  renderThumbs();
}

async function saveCurrent(image) {
  if (!concept || (!svgMarkup && !sourcePhoto)) return;
  await fetch("/api/save", {
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

async function svgToPng(svg, { clear = true } = {}) {
  const blob = new Blob([wrapSvg(svg)], { type: "image/svg+xml;charset=utf-8" });
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
  const overlay = await svgToPng(serializeSketch(), { clear: false });
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
  return path.startsWith("/scenes/") ? path : "";
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
  if (!concept && !onPhoto) {
    setStatus("Describe a scene first.", true);
    return;
  }
  mode = onPhoto ? "edit" : "sketch";
  setBusy(true, onPhoto ? "Marking" : "Sketching");
  setStatus(onPhoto ? "Drawing on the photo…" : "Laying out objects…");
  try {
    const bodyIn = { concept };
    if (onPhoto) {
      const packed = await compactPhoto(sourcePhoto);
      bodyIn.image = packed.image;
      bodyIn.mime = packed.mime;
    }
    const response = await fetch(onPhoto ? "/api/overlay" : "/api/sketch", {
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
    const payload = {
      concept,
      prompt: edit,
      svg: serializeSketch(),
      mode,
    };
    if (mode === "edit" && sourcePhoto) {
      const original = await imagePayload(sourcePhoto);
      const overlay = dataPayload(await compositeEdit());
      payload.image = original.image;
      payload.mime = original.mime;
      payload.overlay = overlay.image;
      payload.overlayMime = overlay.mime;
    } else {
      const dataUrl = await svgToPng(serializeSketch());
      const packed = dataPayload(dataUrl);
      payload.image = packed.image;
      payload.mime = packed.mime;
    }
    const response = await fetch("/api/banana", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || "Nano Banana failed");
    showPhoto(body.image);
    setStatus(body.cached ? "Photograph from this sketch." : "Photograph from the sketch.");
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
  promptEl.placeholder = "Describe the scene.";
  setUnderlay("");
  hideResult();
  svgMarkup = "";
  history = [];
  historyIndex = -1;
  selected = null;
  bananaBtn.disabled = true;
  stage.innerHTML = `<p id="empty">Type a prompt, then Sketch.</p>`;
  setView("work");
  barSub.textContent = "";
}

async function openScene(scene) {
  promptEl.value = scene.concept || "";
  concept = scene.concept || "";
  const photoUrl = scene.photo || scene.source;
  mode = photoUrl ? "edit" : "sketch";
  promptEl.placeholder = photoUrl ? "What should change?" : "Describe the scene.";
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
    stage.innerHTML = `<p id="empty">Tap Sketch to mark the edit.</p>`;
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
  stage.innerHTML = `<p id="empty">Type a prompt, then Sketch.</p>`;
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
    const response = await fetch("/api/delete", {
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
  stage.innerHTML = `<p id="empty">Tap Sketch to mark the edit.</p>`;
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
deleteBtn.addEventListener("click", deleteSelected);

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
    deleteSelected();
  }
});

function onTransformMove(event) {
  if (!selected || !gesture || event.pointerId !== gesture.pointer) return;
  const point = svgPoint(event);
  if (gesture.op === "move") {
    selected.dataset.tx = String(gesture.tx + (point.x - gesture.x));
    selected.dataset.ty = String(gesture.ty + (point.y - gesture.y));
  } else if (gesture.op === "scale") {
    const dist = Math.hypot(point.x - gesture.cx, point.y - gesture.cy);
    selected.dataset.scale = String(Math.min(4, Math.max(0.2, (dist / gesture.dist) * gesture.scale)));
  } else if (gesture.op === "rotate") {
    const ang = Math.atan2(point.y - gesture.cy, point.x - gesture.cx);
    selected.dataset.rot = String(gesture.rot + ((ang - gesture.ang) * 180) / Math.PI);
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
  const handle = event.target.closest("[data-op]");
  const op = handle?.dataset.op || "move";
  event.preventDefault();
  event.stopPropagation();
  const point = svgPoint(event);
  const cx = Number(selected.dataset.cx) + Number(selected.dataset.tx || 0);
  const cy = Number(selected.dataset.cy) + Number(selected.dataset.ty || 0);
  gesture = {
    op,
    pointer: event.pointerId,
    tx: Number(selected.dataset.tx || 0),
    ty: Number(selected.dataset.ty || 0),
    rot: Number(selected.dataset.rot || 0),
    scale: Number(selected.dataset.scale || 1),
    x: point.x,
    y: point.y,
    cx,
    cy,
    dist: Math.max(8, Math.hypot(point.x - cx, point.y - cy)),
    ang: Math.atan2(point.y - cy, point.x - cx),
  };
  try {
    (handle || gizmoBox).setPointerCapture(event.pointerId);
  } catch {
    /* synthetic events have no real pointer */
  }
});

window.addEventListener("pointermove", onTransformMove);
window.addEventListener("pointerup", onTransformUp);
window.addEventListener("pointercancel", onTransformUp);

sketchBtn.addEventListener("click", runSketch);
bananaBtn.addEventListener("click", runBanana);
promptEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (svgMarkup) runBanana();
    else runSketch();
  }
});

refreshScenes().then(() => setView("open"));
