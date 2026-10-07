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
};

const dictionaries = { en, zh } as const;

type LangContextValue = {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: Dict;
};

const LangContext = createContext<LangContextValue>({
  lang: "en",
  setLang: () => {},
  t: en,
});

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>("en");

  useEffect(() => {
    const saved = localStorage.getItem("lang");
    if (saved === "en" || saved === "zh") {
      setLang(saved);
      return;
    }
    if (navigator.language.toLowerCase().startsWith("zh")) setLang("zh");
  }, []);

  useEffect(() => {
    localStorage.setItem("lang", lang);
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  }, [lang]);

  return (
    <LangContext.Provider value={{ lang, setLang, t: dictionaries[lang] }}>{children}</LangContext.Provider>
  );
}

export function useLang() {
  return useContext(LangContext);
}
