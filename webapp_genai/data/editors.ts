export const editorChoices = [
  { id: "nano_banana", name: "Nano Banana" },
  { id: "qwen_image_3", name: "Qwen" },
  { id: "seedream_5_pro", name: "Seedream" },
  { id: "gpt_image_2", name: "GPT Image" },
] as const;

export type EditorId = (typeof editorChoices)[number]["id"];
