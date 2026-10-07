import PhotosUI
import SwiftUI
import UIKit

@MainActor
final class EditIslandsViewModel: ObservableObject {
    // MARK: - Source
    @Published var originalItem: PhotosPickerItem?
    @Published var originalImage: UIImage?
    @Published private(set) var originalBuffer: ImageBuffer?

    // MARK: - New layer intake
    @Published var pendingEditedItem: PhotosPickerItem?
    @Published var pendingEditedImage: UIImage?
    @Published var promptText: String = ""
    @Published var selectedProvider: EditProvider = .nanoBanana
    @Published var showAPIKeys: Bool = false
    /// Draft text for a follow-up edit on a version. Keyed by that version's id.
    @Published var subpromptDrafts: [UUID: String] = [:]

    // MARK: - Document
    @Published var layers: [EditLayer] = []
    @Published var selectedLayerID: UUID?
    @Published var selectedIslandIDs: Set<UUID> = []
    @Published var booleanOpsSelection: Set<UUID> = []

    /// How many largest islands to show in the side list.
    let maxListedIslands = 8

    // MARK: - Tools
    @Published var tool: EditorTool = .select
    @Published var eraserRadius: Int = 14
    @Published var showIslandOverlay: Bool = true
    @Published var ssimThreshold: Double = 0.85
    @Published var minIslandArea: Int = 64
    /// Live filter. Islands smaller than this stay in the version but are left out of the photo and the cursor.
    @Published var islandAreaFilter: Int = 0

    /// Photo zoom/pan — owned here so tool switches never reset the view.
    @Published var canvasScale: CGFloat = 1
    @Published var canvasOffset: CGSize = .zero

    // MARK: - Preview
    @Published var compositeImage: UIImage?
    @Published var overlayImage: UIImage?
    @Published var statusMessage: String = ""
    @Published var isBusy: Bool = false
    @Published var busyLabel: String = "Working…"
    @Published var showProjectExporter = false
    @Published var showProjectImporter = false
    @Published var projectDocument: EditIslandsProjectDocument?

    /// Working composite pixels — patched in place for fast island toggles.
    private var compositeBuffer: ImageBuffer?

    /// Skip heavy UIImage rebuilds while the eraser stroke is in progress.
    private var isErasing = false
    private var eraseDirty = false
    private var eraseStrokeBegan = false
    private var overlayTask: Task<Void, Never>?

    // MARK: - Undo / Redo
    private struct HistoryEntry {
        let layers: [EditLayer]
        let selectedLayerID: UUID?
        let selectedIslandIDs: Set<UUID>
    }

    private var undoStack: [HistoryEntry] = []
    private var redoStack: [HistoryEntry] = []
    private let maxHistory = 40

    @Published private(set) var canUndo = false
    @Published private(set) var canRedo = false

    var selectedProviderHasKey: Bool {
        APIKeyStore.hasKey(for: selectedProvider)
    }

    var selectedLayer: EditLayer? {
        guard let id = selectedLayerID else { return nil }
        return layers.first { $0.id == id }
    }

    var selectedLayerIndex: Int? {
        guard let id = selectedLayerID else { return nil }
        return layers.firstIndex { $0.id == id }
    }

    /// Largest islands on the active layer (for the toggle list).
    var listedIslands: [MaskIsland] {
        guard let layer = selectedLayer else { return [] }
        return Array(layer.islands.sorted { $0.area > $1.area }.prefix(maxListedIslands))
    }

    /// Full island list for the scrollable panel (largest first).
    var scrollableIslands: [MaskIsland] {
        guard let layer = selectedLayer else { return [] }
        return layer.islands.sorted { $0.area > $1.area }
    }

    var hiddenIslandCount: Int {
        max(0, (selectedLayer?.islands.count ?? 0) - listedIslands.count)
    }

    // MARK: - Load

    func loadOriginal() async {
        guard let item = originalItem else { return }
        isBusy = true
        defer { isBusy = false }
        do {
            if let data = try await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                originalImage = img
                originalBuffer = ImageBuffer.from(img)
                layers = []
                selectedLayerID = nil
                selectedIslandIDs = []
                booleanOpsSelection = []
                clearHistory()
                resetCanvasZoom()
                refreshPreview()
                statusMessage = "Original loaded (\(originalBuffer?.width ?? 0)×\(originalBuffer?.height ?? 0))"
            }
        } catch {
            statusMessage = "Failed to load original: \(error.localizedDescription)"
        }
    }

    func loadPendingEdited() async {
        guard let item = pendingEditedItem else { return }
        do {
            if let data = try await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                pendingEditedImage = img
            }
        } catch {
            statusMessage = "Failed to load edited image: \(error.localizedDescription)"
        }
    }

    /// Run the selected commercial editor, then diff → islands → new layer.
    func runSelectedProviderEdit() async {
        guard let original = originalImage else {
            statusMessage = "Load an original image first."
            return
        }
        let prompt = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            statusMessage = "Enter a prompt for this edit."
            return
        }
        guard selectedProviderHasKey else {
            statusMessage = "Add an API key for \(selectedProvider.displayName)."
            showAPIKeys = true
            return
        }

        isBusy = true
        busyLabel = "Running \(selectedProvider.displayName)…"
        defer { isBusy = false }

        do {
            let edited = try await ImageEditClient.edit(
                provider: selectedProvider,
                source: original,
                prompt: prompt
            )
            pendingEditedImage = edited
            busyLabel = "Segmenting islands…"
            addLayerFromPendingEdit(provider: selectedProvider)
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    /// Diff pending edit against original → islands → new layer.
    func addLayerFromPendingEdit(provider: EditProvider? = nil) {
        guard let originalBuffer else {
            statusMessage = "Load an original image first."
            return
        }
        guard let pending = pendingEditedImage,
              let editedBuf = ImageBuffer.from(pending, maxSide: CGFloat(max(originalBuffer.width, originalBuffer.height)))
        else {
            statusMessage = "Pick an edited result image for this prompt."
            return
        }
        guard let matched = ImageBuffer.matchSize(originalBuffer, editedBuf) else {
            statusMessage = "Could not align image sizes."
            return
        }

        isBusy = true
        busyLabel = "Segmenting islands…"
        defer { isBusy = false }

        // Keep document resolution locked to the first original buffer.
        let workOriginal: ImageBuffer
        let workEdited: ImageBuffer
        if matched.0.width == originalBuffer.width && matched.0.height == originalBuffer.height {
            workOriginal = matched.0
            workEdited = matched.1
        } else if let resizedOrig = ImageBuffer.from(originalImage ?? UIImage(), maxSide: CGFloat(max(matched.0.width, matched.0.height))),
                  let pair = ImageBuffer.matchSize(resizedOrig, matched.1) {
            // Re-lock original to matched size if needed (first layer only typically).
            self.originalBuffer = pair.0
            workOriginal = pair.0
            workEdited = pair.1
        } else {
            workOriginal = matched.0
            workEdited = matched.1
            self.originalBuffer = workOriginal
        }

        let options = PerceptualDiff.Options(
            ssimThreshold: Float(ssimThreshold),
            minIslandArea: minIslandArea
        )
        guard let diff = PerceptualDiff.changeMask(
            original: workOriginal,
            edited: workEdited,
            options: options
        ) else {
            statusMessage = "Perceptual diff failed."
            return
        }

        let islands = IslandSegmenter.segment(
            changeMask: diff.changeMask,
            width: diff.width,
            height: diff.height,
            minArea: minIslandArea
        )

        let prompt = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        let layer = EditLayer(
            prompt: prompt.isEmpty ? "Edit \(layers.count + 1)" : prompt,
            provider: provider,
            editedPixels: workEdited.rgba,
            width: workEdited.width,
            height: workEdited.height,
            islands: islands
        )

        pushUndo()
        layers.append(layer)
        selectedLayerID = layer.id
        selectedIslandIDs = []
        pendingEditedImage = nil
        pendingEditedItem = nil
        promptText = ""
        let toolTag = provider.map { " via \($0.shortName)" } ?? ""
        statusMessage = "Layer added\(toolTag) — \(islands.count) island(s), \(String(format: "%.1f", diff.changeFraction * 100))% changed"
        refreshPreview()
    }

    /// Keep the prompt text on a version without calling the editor.
    func updateLayerPrompt(_ id: UUID, text: String) {
        guard let idx = layers.firstIndex(where: { $0.id == id }) else { return }
        layers[idx].prompt = text
        selectedLayerID = id
    }

    /// Re-run this version's prompt and replace its result so the photo updates.
    func rerunLayer(_ id: UUID) async {
        guard let original = originalImage, let originalBuffer else {
            statusMessage = "Load an original image first."
            return
        }
        guard let idx = layers.firstIndex(where: { $0.id == id }) else { return }
        let prompt = layers[idx].prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            statusMessage = "Enter a prompt for this version."
            return
        }
        let provider = layers[idx].provider ?? selectedProvider
        if let parentID = layers[idx].parentID {
            await applyMaskedFollowUp(parentID: parentID, prompt: prompt, provider: provider, replacing: id)
            return
        }
        guard APIKeyStore.hasKey(for: provider) else {
            statusMessage = "Add an API key for \(provider.displayName)."
            showAPIKeys = true
            return
        }

        isBusy = true
        busyLabel = "Updating with \(provider.shortName)…"
        defer { isBusy = false }

        do {
            let edited = try await ImageEditClient.edit(
                provider: provider,
                source: original,
                prompt: prompt
            )
            guard let workEdited = bufferMatchingOriginal(originalBuffer, image: edited) else {
                statusMessage = "Could not align the new result."
                return
            }
            let options = PerceptualDiff.Options(
                ssimThreshold: Float(ssimThreshold),
                minIslandArea: minIslandArea
            )
            guard let diff = PerceptualDiff.changeMask(
                original: originalBuffer,
                edited: workEdited,
                options: options
            ) else {
                statusMessage = "Perceptual diff failed."
                return
            }
            let islands = IslandSegmenter.segment(
                changeMask: diff.changeMask,
                width: diff.width,
                height: diff.height,
                minArea: minIslandArea
            )

            pushUndo()
            var layer = layers[idx]
            layer.prompt = prompt
            layer.provider = provider
            layer.editedPixels = workEdited.rgba
            layer.width = workEdited.width
            layer.height = workEdited.height
            layer.islands = islands
            layer.isVisible = true
            layer.refreshCombinedMask()
            layers[idx] = layer
            selectedLayerID = layer.id
            selectedIslandIDs = []
            statusMessage = "Updated — \(islands.count) region(s), \(String(format: "%.1f", diff.changeFraction * 100))% changed"
            refreshPreview()
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    /// Scale an edit result onto the locked original canvas.
    private func bufferMatchingOriginal(_ original: ImageBuffer, image: UIImage) -> ImageBuffer? {
        guard let raw = ImageBuffer.from(image, maxSide: CGFloat(max(original.width, original.height))) else {
            return nil
        }
        if raw.width == original.width && raw.height == original.height {
            return raw
        }
        guard let ui = raw.toUIImage() else { return nil }
        let size = CGSize(width: original.width, height: original.height)
        let renderer = UIGraphicsImageRenderer(size: size)
        let scaled = renderer.image { _ in
            ui.draw(in: CGRect(origin: .zero, size: size))
        }
        return ImageBuffer.from(scaled, maxSide: CGFloat(max(original.width, original.height)))
    }

    /// Send this version's image and mask with a further-change prompt. Adds a child version.
    func runSubprompt(on parentID: UUID) async {
        let prompt = (subpromptDrafts[parentID] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            statusMessage = "Enter a subprompt for the further change."
            return
        }
        guard let parent = layers.first(where: { $0.id == parentID }) else { return }
        let provider = parent.provider ?? selectedProvider
        let ok = await applyMaskedFollowUp(
            parentID: parentID,
            prompt: prompt,
            provider: provider,
            replacing: nil
        )
        if ok {
            subpromptDrafts[parentID] = ""
        }
    }

    /// Image + mask follow-up. `replacing` updates an existing child; otherwise a new child is inserted.
    @discardableResult
    private func applyMaskedFollowUp(
        parentID: UUID,
        prompt: String,
        provider: EditProvider,
        replacing: UUID?
    ) async -> Bool {
        guard let originalBuffer else {
            statusMessage = "Load an original image first."
            return false
        }
        guard let parent = layers.first(where: { $0.id == parentID }) else { return false }
        guard APIKeyStore.hasKey(for: provider) else {
            statusMessage = "Add an API key for \(provider.displayName)."
            showAPIKeys = true
            return false
        }
        let region = filteredMask(for: parent)
        guard region.contains(where: { $0 != 0 }) else {
            statusMessage = "This version has no mask to refine."
            return false
        }
        let base = Compositor.composite(
            original: originalBuffer,
            layers: branchLayers(endingAt: parentID),
            minArea: islandAreaFilter
        )
        guard let sourceImage = base.toUIImage(),
              let maskImage = ImageBuffer.visibleMaskImage(mask: region, width: parent.width, height: parent.height)
        else {
            statusMessage = "Could not build the image and mask."
            return false
        }

        isBusy = true
        busyLabel = "Subprompt with \(provider.shortName)…"
        defer { isBusy = false }

        do {
            let edited = try await ImageEditClient.edit(
                provider: provider,
                source: sourceImage,
                prompt: prompt,
                mask: maskImage
            )
            guard let workEdited = bufferMatchingOriginal(originalBuffer, image: edited) else {
                statusMessage = "Could not align the new result."
                return false
            }
            let options = PerceptualDiff.Options(
                ssimThreshold: Float(ssimThreshold),
                minIslandArea: minIslandArea
            )
            guard let diff = PerceptualDiff.changeMask(
                original: base,
                edited: workEdited,
                options: options
            ) else {
                statusMessage = "Perceptual diff failed."
                return false
            }
            var constrained = diff.changeMask
            let limit = min(constrained.count, region.count)
            for i in 0..<limit where region[i] == 0 {
                constrained[i] = 0
            }
            let islands = IslandSegmenter.segment(
                changeMask: constrained,
                width: diff.width,
                height: diff.height,
                minArea: minIslandArea
            )
            guard !islands.isEmpty else {
                statusMessage = "No further change inside the mask."
                return false
            }

            pushUndo()
            if let replacing, let idx = layers.firstIndex(where: { $0.id == replacing }) {
                var layer = layers[idx]
                layer.prompt = prompt
                layer.provider = provider
                layer.editedPixels = workEdited.rgba
                layer.width = workEdited.width
                layer.height = workEdited.height
                layer.islands = islands
                layer.isVisible = true
                layer.parentID = parentID
                layer.refreshCombinedMask()
                layers[idx] = layer
                selectedLayerID = layer.id
            } else {
                let layer = EditLayer(
                    prompt: prompt,
                    provider: provider,
                    editedPixels: workEdited.rgba,
                    width: workEdited.width,
                    height: workEdited.height,
                    islands: islands,
                    parentID: parentID
                )
                layers.insert(layer, at: indexAfterBranch(parentID))
                selectedLayerID = layer.id
            }
            selectedIslandIDs = []
            statusMessage = "Subprompt — \(islands.count) region(s) inside the mask"
            refreshPreview()
            return true
        } catch {
            statusMessage = error.localizedDescription
            return false
        }
    }

    /// This version plus its parents, in paint order.
    private func branchLayers(endingAt id: UUID) -> [EditLayer] {
        var chain: [UUID] = [id]
        var current = layers.first { $0.id == id }?.parentID
        var guardCount = 0
        while let pid = current, guardCount < layers.count {
            chain.append(pid)
            current = layers.first { $0.id == pid }?.parentID
            guardCount += 1
        }
        let set = Set(chain)
        return layers.filter { set.contains($0.id) }
    }

    private func descendantIDs(of id: UUID) -> Set<UUID> {
        var out: Set<UUID> = []
        var stack = layers.filter { $0.parentID == id }.map(\.id)
        while let next = stack.popLast() {
            if out.contains(next) { continue }
            out.insert(next)
            stack.append(contentsOf: layers.filter { $0.parentID == next }.map(\.id))
        }
        return out
    }

    private func indexAfterBranch(_ id: UUID) -> Int {
        let ids = descendantIDs(of: id).union([id])
        if let last = layers.lastIndex(where: { ids.contains($0.id) }) {
            return last + 1
        }
        return layers.count
    }

    // MARK: - Interaction

    /// Cursor tool — tap the island that is visible on top, on any version.
    func handleTap(imageX: Int, imageY: Int) {
        switch tool {
        case .hand:
            return
        case .select:
            guard let hit = topmostIsland(atX: imageX, y: imageY) else {
                clearIslandSelection()
                statusMessage = "No island at that point"
                return
            }
            selectIsland(hit.islandID, on: hit.layerID)
        case .eraser:
            commitEraseStroke(points: [(imageX, imageY)])
        }
    }

    func toggleIslandSelection(id: UUID) {
        if selectedIslandIDs.contains(id) {
            selectedIslandIDs.remove(id)
            statusMessage = selectedIslandIDs.isEmpty
                ? "Nothing selected"
                : "\(selectedIslandIDs.count) selected — Delete removes edge pixels"
        } else {
            selectedIslandIDs.insert(id)
            statusMessage = "\(selectedIslandIDs.count) selected — Delete removes edge pixels"
        }
        scheduleOverlayRefresh()
    }

    func activateLayer(_ id: UUID) {
        selectedLayerID = id
        selectedIslandIDs = []
        statusMessage = "Cursor targets this version"
        scheduleOverlayRefresh()
    }

    /// Cursor marquee — select every island the blue box covers on that version.
    func selectIslandsInImageRect(_ rect: CGRect) {
        let normalized = rect.standardized
        guard normalized.width > 2 || normalized.height > 2 else {
            clearIslandSelection()
            statusMessage = "Selection cleared"
            return
        }

        struct Candidate {
            let layerID: UUID
            let hits: [MaskIsland]
            let frontRank: Int
            let isActive: Bool
        }

        var best: Candidate?
        for (rank, item) in visibleLayersFrontToBack().enumerated() {
            let hits = item.layer.islands.filter {
                $0.isEnabled && $0.area >= islandAreaFilter && $0.containsAnyPixel(in: normalized)
            }
            guard !hits.isEmpty else { continue }
            let candidate = Candidate(
                layerID: item.layer.id,
                hits: hits,
                frontRank: rank,
                isActive: item.layer.id == selectedLayerID
            )
            guard let current = best else {
                best = candidate
                continue
            }
            if candidate.hits.count > current.hits.count
                || (candidate.hits.count == current.hits.count && candidate.isActive && !current.isActive)
                || (candidate.hits.count == current.hits.count && candidate.isActive == current.isActive && candidate.frontRank < current.frontRank) {
                best = candidate
            }
        }

        guard let best else {
            clearIslandSelection()
            statusMessage = "No islands in selection"
            return
        }
        selectedLayerID = best.layerID
        selectedIslandIDs = Set(best.hits.map(\.id))
        statusMessage = "Selected \(best.hits.count) island\(best.hits.count == 1 ? "" : "s") — Delete to restore"
        scheduleOverlayRefresh()
    }

    func setIslandAreaFilter(_ value: Int) {
        let next = max(0, value)
        guard next != islandAreaFilter else { return }
        islandAreaFilter = next
        if let layer = selectedLayer {
            let allowed = Set(layer.islands.filter { $0.area >= next }.map(\.id))
            selectedIslandIDs.formIntersection(allowed)
        }
        refreshPreview()
    }

    /// Change mask with islands under the size filter left out.
    func filteredMask(for layer: EditLayer) -> [UInt8] {
        guard islandAreaFilter > 0 else { return layer.combinedMask }
        let kept = layer.islands.filter { $0.isEnabled && $0.area >= islandAreaFilter }
        return EditLayer.rebuildCombinedMask(islands: kept, width: layer.width, height: layer.height)
    }

    /// Later versions paint on top, so search those first. Hidden versions are skipped.
    private func visibleLayersFrontToBack() -> [(index: Int, layer: EditLayer)] {
        layers.enumerated().reversed().compactMap { offset, layer in
            layer.isVisible ? (index: offset, layer: layer) : nil
        }
    }

    private func topmostIsland(atX x: Int, y: Int) -> (layerID: UUID, islandID: UUID)? {
        for item in visibleLayersFrontToBack() {
            if let id = IslandSegmenter.hitTest(
                islands: item.layer.islands.filter { $0.area >= islandAreaFilter },
                x: x,
                y: y
            ) {
                return (item.layer.id, id)
            }
        }
        return nil
    }

    private func selectIsland(_ islandID: UUID, on layerID: UUID) {
        if selectedLayerID != layerID {
            selectedLayerID = layerID
            selectedIslandIDs = []
        }
        toggleIslandSelection(id: islandID)
    }

    func activateHandTool() {
        tool = .hand
        statusMessage = "Hand: drag to pan — pinch to zoom"
    }

    func activateCursorTool() {
        tool = .select
        statusMessage = "Cursor: drag a blue box or tap islands, then Delete"
    }

    func activateEraserTool() {
        tool = .eraser
        statusMessage = "Eraser: paint under the orange ring"
    }

    func resetCanvasZoom() {
        canvasScale = 1
        canvasOffset = .zero
    }

    func handleDrag(imageX: Int, imageY: Int) {
        // Kept for API compatibility — realtime erase disabled; use commitEraseStroke.
        _ = imageX
        _ = imageY
    }

    func endEraseStroke() {
        // No-op: strokes commit via commitEraseStroke.
    }

    /// Apply a full eraser stroke at once after the finger lifts (fast interaction).
    func commitEraseStroke(points: [(x: Int, y: Int)]) {
        guard tool == .eraser || !points.isEmpty,
              let idx = selectedLayerIndex,
              !points.isEmpty
        else { return }

        pushUndo()
        for p in points {
            eraseAt(layerIndex: idx, x: p.x, y: p.y, live: true)
        }
        for i in layers[idx].islands.indices {
            layers[idx].islands[i].area = BooleanMaskOps.area(of: layers[idx].islands[i].mask)
            layers[idx].islands[i].bbox = MaskIsland.computeBBox(
                mask: layers[idx].islands[i].mask,
                width: layers[idx].width,
                height: layers[idx].height
            )
        }
        layers[idx].refreshCombinedMask()
        eraseDirty = false
        isErasing = false
        eraseStrokeBegan = false
        refreshPreview()
        statusMessage = "Eraser applied — Undo available"
    }

    private func beginEraseStroke() {
        if !eraseStrokeBegan {
            pushUndo()
            eraseStrokeBegan = true
        }
        isErasing = true
        eraseDirty = false
    }

    private func eraseAt(layerIndex idx: Int, x: Int, y: Int, live: Bool) {
        let targets: [UUID]
        if !selectedIslandIDs.isEmpty {
            targets = Array(selectedIslandIDs)
        } else {
            targets = layers[idx].islands.filter(\.isEnabled).map(\.id)
        }

        for sid in targets {
            guard let iIdx = layers[idx].islands.firstIndex(where: { $0.id == sid }) else { continue }
            BooleanMaskOps.erase(
                mask: &layers[idx].islands[iIdx].mask,
                width: layers[idx].width,
                height: layers[idx].height,
                atX: x,
                atY: y,
                radius: eraserRadius
            )
        }
        eraseDirty = true

        if !live {
            layers[idx].refreshCombinedMask()
            refreshPreview()
        }
    }

    func deleteSelectedIslands() {
        guard let lid = selectedLayerID,
              !selectedIslandIDs.isEmpty,
              let lIdx = layers.firstIndex(where: { $0.id == lid })
        else { return }
        pushUndo()
        let removed = selectedIslandIDs.count
        layers[lIdx].islands.removeAll { selectedIslandIDs.contains($0.id) }
        layers[lIdx].refreshCombinedMask()
        selectedIslandIDs = []
        statusMessage = "Removed \(removed) island\(removed == 1 ? "" : "s") (restored original)"
        refreshPreview()
    }

    func toggleIslandEnabled(_ islandID: UUID) {
        guard let originalBuffer,
              let lIdx = selectedLayerIndex,
              let iIdx = layers[lIdx].islands.firstIndex(where: { $0.id == islandID })
        else { return }

        pushUndo()
        layers[lIdx].islands[iIdx].isEnabled.toggle()
        let enabled = layers[lIdx].islands[iIdx].isEnabled
        let mask = layers[lIdx].islands[iIdx].mask
        layers[lIdx].refreshCombinedMask()

        // Patch only this island's pixels — no full recomposite.
        if compositeBuffer == nil {
            compositeBuffer = Compositor.composite(
                original: originalBuffer,
                layers: layers,
                minArea: islandAreaFilter
            )
        } else if var buffer = compositeBuffer {
            let edited = layers[lIdx].editedPixels
            let orig = originalBuffer.rgba
            let n = mask.count
            for i in 0..<n where mask[i] != 0 {
                let o = i * 4
                if enabled {
                    buffer.rgba[o] = edited[o]
                    buffer.rgba[o + 1] = edited[o + 1]
                    buffer.rgba[o + 2] = edited[o + 2]
                    buffer.rgba[o + 3] = 255
                } else {
                    buffer.rgba[o] = orig[o]
                    buffer.rgba[o + 1] = orig[o + 1]
                    buffer.rgba[o + 2] = orig[o + 2]
                    buffer.rgba[o + 3] = 255
                    for layer in layers where layer.isVisible && filteredMask(for: layer)[i] != 0 {
                        buffer.rgba[o] = layer.editedPixels[o]
                        buffer.rgba[o + 1] = layer.editedPixels[o + 1]
                        buffer.rgba[o + 2] = layer.editedPixels[o + 2]
                    }
                }
            }
            compositeBuffer = buffer
        }

        if let img = compositeBuffer?.toUIImage() {
            compositeImage = img
        }
        scheduleOverlayRefresh()
    }

    func toggleIslandListSelection(_ islandID: UUID) {
        if selectedIslandIDs.contains(islandID) {
            selectedIslandIDs.remove(islandID)
        } else {
            selectedIslandIDs.insert(islandID)
        }
        scheduleOverlayRefresh()
    }

    func clearIslandSelection() {
        selectedIslandIDs = []
        scheduleOverlayRefresh()
    }

    func toggleLayerVisibility(_ layerID: UUID) {
        guard let idx = layers.firstIndex(where: { $0.id == layerID }) else { return }
        layers[idx].isVisible.toggle()
        refreshPreview()
    }

    func deleteLayer(_ layerID: UUID) {
        pushUndo()
        let remove = descendantIDs(of: layerID).union([layerID])
        layers.removeAll { remove.contains($0.id) }
        for id in remove {
            subpromptDrafts[id] = nil
            booleanOpsSelection.remove(id)
        }
        if let selected = selectedLayerID, remove.contains(selected) {
            selectedLayerID = layers.last?.id
            selectedIslandIDs = []
        }
        refreshPreview()
    }

    // MARK: - Boolean ops on layers

    func toggleBooleanSelection(_ layerID: UUID) {
        if booleanOpsSelection.contains(layerID) {
            booleanOpsSelection.remove(layerID)
        } else if booleanOpsSelection.count < 2 {
            booleanOpsSelection.insert(layerID)
        } else {
            if let first = booleanOpsSelection.first {
                booleanOpsSelection.remove(first)
            }
            booleanOpsSelection.insert(layerID)
        }
    }

    func applyBoolean(_ op: BooleanOp) {
        let ids = Array(booleanOpsSelection)
        guard ids.count == 2,
              let aIdx = layers.firstIndex(where: { $0.id == ids[0] }),
              let bIdx = layers.firstIndex(where: { $0.id == ids[1] })
        else {
            statusMessage = "Select exactly two layers for boolean ops."
            return
        }

        let a = layers[aIdx]
        let b = layers[bIdx]
        guard a.width == b.width, a.height == b.height else {
            statusMessage = "Layers must share the same resolution."
            return
        }

        let mask = BooleanMaskOps.apply(
            op,
            a: filteredMask(for: a),
            b: filteredMask(for: b)
        )
        let area = BooleanMaskOps.area(of: mask)
        guard area > 0 else {
            statusMessage = "Boolean result is empty."
            return
        }

        pushUndo()
        let island = MaskIsland(
            label: "\(op.title) result",
            mask: mask,
            width: a.width,
            height: a.height,
            area: area,
            tint: IslandPalette.tint(for: layers.count + 3)
        )
        let layer = EditLayer(
            prompt: "\(op.title): \(a.prompt) ⊕ \(b.prompt)",
            editedPixels: a.editedPixels,
            width: a.width,
            height: a.height,
            islands: [island]
        )
        layers.append(layer)
        selectedLayerID = layer.id
        selectedIslandIDs = [island.id]
        booleanOpsSelection = []
        statusMessage = "Boolean \(op.title) → new layer (\(area) px)"
        refreshPreview()
    }

    // MARK: - Undo / Redo

    private func pushUndo() {
        undoStack.append(
            HistoryEntry(
                layers: layers,
                selectedLayerID: selectedLayerID,
                selectedIslandIDs: selectedIslandIDs
            )
        )
        if undoStack.count > maxHistory {
            undoStack.removeFirst(undoStack.count - maxHistory)
        }
        redoStack.removeAll()
        updateHistoryFlags()
    }

    func undo() {
        guard let entry = undoStack.popLast() else { return }
        redoStack.append(
            HistoryEntry(
                layers: layers,
                selectedLayerID: selectedLayerID,
                selectedIslandIDs: selectedIslandIDs
            )
        )
        layers = entry.layers
        selectedLayerID = entry.selectedLayerID
        selectedIslandIDs = entry.selectedIslandIDs
        updateHistoryFlags()
        refreshPreview()
        statusMessage = "Undo"
    }

    func redo() {
        guard let entry = redoStack.popLast() else { return }
        undoStack.append(
            HistoryEntry(
                layers: layers,
                selectedLayerID: selectedLayerID,
                selectedIslandIDs: selectedIslandIDs
            )
        )
        layers = entry.layers
        selectedLayerID = entry.selectedLayerID
        selectedIslandIDs = entry.selectedIslandIDs
        updateHistoryFlags()
        refreshPreview()
        statusMessage = "Redo"
    }

    private func updateHistoryFlags() {
        canUndo = !undoStack.isEmpty
        canRedo = !redoStack.isEmpty
    }

    private func clearHistory() {
        undoStack.removeAll()
        redoStack.removeAll()
        updateHistoryFlags()
    }

    // MARK: - Preview / export

    func refreshPreview() {
        guard let originalBuffer else {
            compositeImage = originalImage
            compositeBuffer = nil
            overlayImage = nil
            return
        }
        guard originalBuffer.width > 0, originalBuffer.height > 0 else { return }
        let comp = Compositor.composite(original: originalBuffer, layers: layers, minArea: islandAreaFilter)
        compositeBuffer = comp
        compositeImage = comp.toUIImage()
        scheduleOverlayRefresh()
    }

    func refreshOverlayOnly() {
        scheduleOverlayRefresh(immediate: true)
    }

    private func scheduleOverlayRefresh(immediate: Bool = false) {
        overlayTask?.cancel()
        let work = {
            guard let originalBuffer = self.originalBuffer, self.showIslandOverlay else {
                self.overlayImage = nil
                return
            }
            guard originalBuffer.width > 0, originalBuffer.height > 0 else {
                self.overlayImage = nil
                return
            }
            // Only tint largest + selected islands (keeps UI responsive).
            let overlay = Compositor.islandOverlay(
                size: (originalBuffer.width, originalBuffer.height),
                layers: self.layers,
                selectedIslandIDs: self.selectedIslandIDs,
                maxIslandsPerLayer: self.maxListedIslands,
                minArea: self.islandAreaFilter
            )
            self.overlayImage = overlay.toUIImage()
        }
        if immediate {
            work()
        } else {
            overlayTask = Task { @MainActor in
                try? await Task.sleep(nanoseconds: 40_000_000)
                guard !Task.isCancelled else { return }
                work()
            }
        }
    }

    func saveCompositeToPhotos() {
        guard let image = compositeImage else { return }
        UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil)
        statusMessage = "Saved composite to Photos"
    }

    func beginProjectExport() {
        guard let originalBuffer else {
            statusMessage = "Load an original image before saving a project."
            return
        }
        do {
            let data = try ProjectArchive.encode(
                ProjectArchive.Snapshot(
                    original: originalBuffer,
                    layers: layers,
                    selectedLayerID: selectedLayerID,
                    selectedIslandIDs: selectedIslandIDs,
                    ssimThreshold: ssimThreshold,
                    minIslandArea: minIslandArea,
                    islandAreaFilter: islandAreaFilter
                )
            )
            projectDocument = EditIslandsProjectDocument(data: data)
            showProjectExporter = true
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    func loadProject(from url: URL) {
        let didAccess = url.startAccessingSecurityScopedResource()
        defer {
            if didAccess { url.stopAccessingSecurityScopedResource() }
        }
        do {
            let data = try Data(contentsOf: url)
            let snapshot = try ProjectArchive.decode(data)
            originalImage = snapshot.original.toUIImage()
            originalBuffer = snapshot.original
            originalItem = nil
            pendingEditedItem = nil
            pendingEditedImage = nil
            layers = snapshot.layers
            selectedLayerID = snapshot.selectedLayerID ?? snapshot.layers.last?.id
            selectedIslandIDs = snapshot.selectedIslandIDs
            booleanOpsSelection = []
            subpromptDrafts = [:]
            ssimThreshold = snapshot.ssimThreshold
            minIslandArea = snapshot.minIslandArea
            islandAreaFilter = snapshot.islandAreaFilter
            clearHistory()
            resetCanvasZoom()
            refreshPreview()
            statusMessage = "Opened project — \(layers.count) version\(layers.count == 1 ? "" : "s")"
        } catch {
            statusMessage = error.localizedDescription
        }
    }
}
