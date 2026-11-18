"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export type Lang = "en" | "zh";

type Dict = {
  title: string;
  lab: string;
  upload: string;
  settings: string;
  language: string;
  undo: string;
  redo: string;
  delete: string;
  editor: string;
  noKey: (name: string) => string;
  ignoreSmall: string;
  off: string;
  underPx: (px: string) => string;
  eraserSize: string;
  selection: string;
  tap: string;
  box: string;
  lasso: string;
  scissors: string;
  eraser: string;
  history: string;
  newPrompt: string;
  addPrompt: string;
  update: string;
  subprompt: string;
  furtherChange: string;
  sendsEdited: string;
  regions: (n: number) => string;
  activeVersion: string;
  makeActive: string;
  hideVersion: string;
  showVersion: string;
  deleteVersion: (label: string) => string;
  changeRegion: string;
  addSubprompt: (label: string) => string;
  promptFor: (label: string) => string;
  furtherFor: (label: string) => string;
  openProcess: string;
  saveProcess: string;
  password: string;
  enter: string;
  wrongPassword: string;
  passwordCheckFailed: string;
  couldNotLoadEditors: string;
  removedRegions: (n: number) => string;
  drawLasso: string;
  keptLasso: string;
  drawCut: string;
  cutInto: (n: number) => string;
  uploadFirst: string;
  uploadBeforeSave: string;
  running: (name: string) => string;
  updating: (name: string) => string;
  subprompting: (name: string) => string;
  editFailed: string;
  compressed: (kb: number) => string;
  couldNotReadPhoto: string;
  saving: string;
  savedVersions: (n: number) => string;
  couldNotSave: string;
  opening: string;
  openedVersions: (n: number) => string;
  couldNotOpen: string;
  demoPromptBag: string;
  demoPromptPeople: string;
  approachTree: string;
  approachIslands: string;
  approachControls: string;
  approachSketch: string;
  approachRefs: string;
  approachBlurbTree: string;
  approachBlurbIslands: string;
  approachBlurbControls: string;
  approachBlurbSketch: string;
  approachBlurbRefs: string;
  approachesLabel: string;
  comboVersions: string;
  comboSliders: string;
  controlsTitle: string;
  controlsBody: string;
  colorTitle: string;
  colorSubtitle: string;
  colorToggle: string;
  coloring: string;
  colorObjects: string;
  noObjects: string;
  objectCount: (n: number) => string;
  showMap: string;
  showAdjust: string;
  colorHint: string;
  colorHintOff: string;
  colorLabel: string;
  sizeLabel: string;
  drawFailed: string;
  refsTitle: string;
  refsBody: string;
  refsBusy: string;
  refsAlt: string;
};

const en: Dict = {
  title: "GenAI Image Editor",
  lab: "HMI Lab",
  upload: "Upload a photo",
  settings: "Settings",
  language: "Language",
  undo: "Undo",
  redo: "Redo",
  delete: "Delete",
  editor: "Editor",
  noKey: (name) => `${name} — No key`,
  ignoreSmall: "Ignore small islands",
  off: "Off",
  underPx: (px) => `under ${px} px`,
  eraserSize: "Eraser size",
  selection: "Selection",
  tap: "Tap",
  box: "Box",
  lasso: "Lasso",
  scissors: "Scissors",
  eraser: "Eraser",
  history: "Version history",
  newPrompt: "New prompt",
  addPrompt: "Add prompt",
  update: "Update",
  subprompt: "Subprompt",
  furtherChange: "Further change inside this mask",
  sendsEdited: "Sends the edited image",
  regions: (n) => (n === 1 ? "1 change region" : `${n} change regions`),
  activeVersion: "Active version",
  makeActive: "Make this version active",
  hideVersion: "Hide version",
  showVersion: "Show version",
  deleteVersion: (label) => `Delete ${label}`,
  changeRegion: "Change region",
  addSubprompt: (label) => `Add subprompt to ${label}`,
  promptFor: (label) => `Prompt ${label}`,
  furtherFor: (label) => `Further change ${label}`,
  openProcess: "Open process",
  saveProcess: "Save process",
  password: "Password",
  enter: "Enter",
  wrongPassword: "Wrong password.",
  passwordCheckFailed: "Could not check the password.",
  couldNotLoadEditors: "Could not load editors.",
  removedRegions: (n) => `Removed ${n} change region${n === 1 ? "" : "s"}`,
  drawLasso: "Draw the lasso over the part to keep.",
  keptLasso: "Kept the lassoed part",
  drawCut: "Draw across the island to cut it apart.",
  cutInto: (n) => `Cut into ${n} change regions`,
  uploadFirst: "Upload a photo first.",
  uploadBeforeSave: "Upload a photo before saving.",
  running: (name) => `Running ${name}…`,
  updating: (name) => `Updating with ${name}…`,
  subprompting: (name) => `Subprompt with ${name}…`,
  editFailed: "Edit failed",
  compressed: (kb) => `Compressed to ${kb} KB`,
  couldNotReadPhoto: "Could not read the photo",
  saving: "Saving…",
  savedVersions: (n) => `Saved ${n} version${n === 1 ? "" : "s"}`,
  couldNotSave: "Could not save this process.",
  opening: "Opening…",
  openedVersions: (n) => `Opened ${n} version${n === 1 ? "" : "s"}`,
  couldNotOpen: "Could not open this process.",
  demoPromptBag: "Move the handbag from the chair to the table.",
  demoPromptPeople: "Remove the people in the back.",
  approachTree: "Version tree",
  approachIslands: "Edit islands",
  approachControls: "Structured prompt controls",
  approachSketch: "Fast preview",
  approachRefs: "Supporting references",
  approachBlurbTree: "Each prompt is a branch, and later edits build on a parent result. Open a prompt, add a subprompt, or hide a version — the photo is the history of those branches, not a single overwritten image.",
  approachBlurbIslands: "Each AI change becomes its own region you can keep, move, or undo. Regeneration does not force you to accept every pixel the model touched, so real edits stay separate from unwanted drift.",
  approachBlurbControls: "Instead of a new freeform prompt, change one axis of the scene: pick a sky, set crowd density, or restyle the background and coat for a professional look.",
  approachBlurbSketch: "Fast preview is Sketch Director: type dog, cat, bird — or pick from a PowerPoint-style symbol library — and those stamps appear immediately. Already placed symbols stay; Sketch only draws what is not in the library.",
  approachBlurbRefs: "Supporting references would use burst frames, Live Photos, or Gallery shots as evidence so an edit copies real detail. That work has not started; nothing in this demo is a references prototype.",
  approachesLabel: "Five approaches",
  comboVersions: "Version tree",
  comboSliders: "Colour, size, hide",
  controlsTitle: "Structured prompt controls",
  controlsBody: "Sky, crowd, and professional look are structured controls: one slider or preset, not a new prompt.",
  colorTitle: "Color objects",
  colorSubtitle: "First prompt paints each object a new color",
  colorToggle: "Color objects",
  coloring: "Coloring…",
  colorObjects: "Color the objects",
  noObjects: "No separate objects found.",
  objectCount: (n) => `${n} objects`,
  showMap: "Show the AI color map",
  showAdjust: "Show adjustments",
  colorHint: "The model paints every important object a flat color that is not already in the photo, and returns the list. Then change a color or a size here, without another prompt.",
  colorHintOff: "Turn this on to ask the model for a color map of the photo.",
  colorLabel: "Color",
  sizeLabel: "Size",
  drawFailed: "Could not draw the objects",
  refsTitle: "Supporting references",
  refsBody: "Not started. Burst, Live Photo, and Gallery evidence are defined as an idea only. This phone is empty on purpose.",
  refsBusy: "Matching sky…",
  refsAlt: "Desert with a referenced sky",
};

const zh: Dict = {
  title: "生成式 AI 图像编辑器",
  lab: "HMI Lab",
  upload: "上传照片",
  settings: "设置",
  language: "语言",
  undo: "撤销",
  redo: "重做",
  delete: "删除",
  editor: "编辑模型",
  noKey: (name) => `${name} — 缺少密钥`,
  ignoreSmall: "忽略小区域",
  off: "关闭",
  underPx: (px) => `小于 ${px} 像素`,
  eraserSize: "橡皮擦大小",
  selection: "选择方式",
  tap: "点选",
  box: "框选",
  lasso: "套索",
  scissors: "剪刀",
  eraser: "橡皮擦",
  history: "版本历史",
  newPrompt: "新提示词",
  addPrompt: "添加提示词",
  update: "更新",
  subprompt: "子提示词",
  furtherChange: "在此遮罩内进一步修改",
  sendsEdited: "发送已编辑的图像",
  regions: (n) => `${n} 个修改区域`,
  activeVersion: "当前版本",
  makeActive: "设为当前版本",
  hideVersion: "隐藏版本",
  showVersion: "显示版本",
  deleteVersion: (label) => `删除 ${label}`,
  changeRegion: "修改区域",
  addSubprompt: (label) => `为 ${label} 添加子提示词`,
  promptFor: (label) => `提示词 ${label}`,
  furtherFor: (label) => `进一步修改 ${label}`,
  openProcess: "打开流程",
  saveProcess: "保存流程",
  password: "密码",
  enter: "进入",
  wrongPassword: "密码错误。",
  passwordCheckFailed: "无法验证密码。",
  couldNotLoadEditors: "无法加载编辑模型。",
  removedRegions: (n) => `已移除 ${n} 个修改区域`,
  drawLasso: "用套索圈出要保留的部分。",
  keptLasso: "已保留套索内的部分",
  drawCut: "在区域上划线以切开。",
  cutInto: (n) => `已切成 ${n} 个修改区域`,
  uploadFirst: "请先上传照片。",
  uploadBeforeSave: "请先上传照片再保存。",
  running: (name) => `正在运行 ${name}…`,
  updating: (name) => `正在用 ${name} 更新…`,
  subprompting: (name) => `正在用 ${name} 生成子提示…`,
  editFailed: "编辑失败",
  compressed: (kb) => `已压缩至 ${kb} KB`,
  couldNotReadPhoto: "无法读取照片",
  saving: "正在保存…",
  savedVersions: (n) => `已保存 ${n} 个版本`,
  couldNotSave: "无法保存此流程。",
  opening: "正在打开…",
  openedVersions: (n) => `已打开 ${n} 个版本`,
  couldNotOpen: "无法打开此流程。",
  demoPromptBag: "把手提包从椅子上移到桌子上。",
  demoPromptPeople: "去掉背景中的人。",
  approachTree: "Version tree",
  approachIslands: "Edit islands",
  approachControls: "Structured prompt controls",
  approachSketch: "Fast preview",
  approachRefs: "Supporting references",
  approachBlurbTree: "每次提示都是一条分支，后续修改叠在父结果上。打开提示、加子提示或隐藏某个版本，照片就是这些分支的历史，而不是被覆盖掉的单张图。",
  approachBlurbIslands: "每一次 AI 改动都变成可保留、可移动、可撤销的独立区域。重新生成时不必接受模型碰到的每一个像素，真正的编辑和多余的漂移得以分开。",
  approachBlurbControls: "不必再写一条自由提示，只改场景的一个维度：换天空、调人群疏密，或改背景和外套做成正装。",
  approachBlurbSketch: "Fast preview 就是 Sketch Director：输入狗、猫、鸟，或从符号库点选，符号立刻出现在画面上。已经放上的符号会保留，草图只画库里没有的部分。",
  approachBlurbRefs: "Supporting references 本应用连拍、实况照片或图库原图当证据，让编辑拷贝真实细节。这项工作还没开始；当前演示里没有任何参考图原型。",
  approachesLabel: "Five approaches",
  comboVersions: "Version tree",
  comboSliders: "Colour, size, hide",
  controlsTitle: "Structured prompt controls",
  controlsBody: "天空、人群和正装是结构化控制：一个滑块或预设，而不是新的提示词。",
  colorTitle: "给物体上色",
  colorSubtitle: "第一次提示为每个物体铺上新颜色",
  colorToggle: "给物体上色",
  coloring: "正在上色…",
  colorObjects: "给物体上色",
  noObjects: "没有找到独立物体。",
  objectCount: (n) => `${n} 个物体`,
  showMap: "显示 AI 色图",
  showAdjust: "显示调整结果",
  colorHint: "模型会把每个重要物体涂成照片里没有的纯色，并返回列表。然后在这里改颜色或大小，不用再写提示词。",
  colorHintOff: "打开后即可让模型生成这张照片的色图。",
  colorLabel: "颜色",
  sizeLabel: "大小",
  drawFailed: "无法绘制物体",
  refsTitle: "Supporting references",
  refsBody: "尚未开始。连拍、实况照片和图库证据只停留在想法。这部手机里故意是空的。",
  refsBusy: "正在匹配天空…",
  refsAlt: "带参考天空的沙漠",
};

const dictionaries = { en, zh } as const;

type LangContextValue = {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: Dict;
};

const LangContext = createContext<LangContextValue>({
  lang: "zh",
  setLang: () => {},
  t: zh,
});

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>("zh");

  useEffect(() => {
    const saved = localStorage.getItem("lang");
    if (saved === "en" || saved === "zh") setLangState(saved);
  }, []);

  useEffect(() => {
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  }, [lang]);

  function setLang(next: Lang) {
    localStorage.setItem("lang", next);
    setLangState(next);
    window.dispatchEvent(new CustomEvent("app-lang", { detail: next }));
  }

  return (
    <LangContext.Provider value={{ lang, setLang, t: dictionaries[lang] }}>{children}</LangContext.Provider>
  );
}

export function useLang() {
  return useContext(LangContext);
}
