# EditIslands — Approach 5

iPhone app for **editable mask islands**: each generative edit becomes a layer of changed-pixel islands on top of the original.

## Claim under test

`Final Image = Original + Selected Islands`

Unchanged pixels never enter a layer. Adjacent changed pixels form islands you can select, disable, erase, or combine with boolean ops.

## Open in Xcode

```bash
open edit_islands/EditIslands.xcodeproj
```

Bundle ID: `com.SensciLab.EditIslands` · iOS 18+

## App flow

1. Load the **original** photo.
2. Run a prompt in any editor (Nano Banana, Qwen, Seedream, GPT Image, …).
3. Load the **edited result**, optionally type the prompt label, tap **Diff → Islands → New Layer**.
4. In the island editor:
   - **Select** — tap an island (cursor / hand tool) to highlight it; delete to restore original pixels there.
   - **Eraser** — paint out unwanted pixels on the active layer / selected island.
   - **Boolean** — check two layers, then Unite / Intersect / Subtract / Exclude → new layer.
5. Export the composite to Photos.

Diff uses **local SSIM** (not raw RGB subtraction) to reduce VAE-noise islands.

## Viability eval (do this before trusting the UI)

Cross-tool island purity on existing `genai_image_editing` runs:

```bash
cd genai_image_editing
pip install -r requirements.txt
python island_pipeline.py --run latest --conditions 8 --expected-islands 2
```

Writes `outputs/<run>/islands/purity_summary.csv` plus tinted overlays. If every tool over-segments badly, Approach 5 is not viable regardless of UI polish.
