import SwiftUI

struct EditorView: View {
    @ObservedObject var vm: EditIslandsViewModel

    var body: some View {
        GeometryReader { geo in
            let canvasHeight = max(220, geo.size.height * 0.48)
            VStack(spacing: 0) {
                canvasWithTools
                    .frame(height: canvasHeight)
                    .clipped()
                toolDock
                    .padding(.vertical, 8)
                    .frame(maxWidth: .infinity)
                    .background(Color(.systemBackground))
                islandSizeFilter
                Divider()
                lists
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("Edit")
        .navigationBarTitleDisplayMode(.inline)
        .scrollDismissesKeyboard(.immediately)
        .background(KeyboardDismissInstaller())
        .overlay {
            if vm.isBusy {
                ProgressView(vm.busyLabel)
                    .padding()
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
        }
        .sheet(isPresented: $vm.showAPIKeys) {
            APIKeysView()
        }
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                if !vm.selectedIslandIDs.isEmpty {
                    Button("Clear") { vm.clearIslandSelection() }
                }
                Button {
                    vm.saveCompositeToPhotos()
                } label: {
                    Image(systemName: "square.and.arrow.down")
                }
                .disabled(vm.compositeImage == nil)
                .accessibilityLabel("Save photo")
                Button {
                    vm.beginProjectExport()
                } label: {
                    Image(systemName: "square.and.arrow.up")
                }
                .disabled(vm.originalBuffer == nil)
                .accessibilityLabel("Save project")
                Button {
                    vm.showProjectImporter = true
                } label: {
                    Image(systemName: "folder")
                }
                .accessibilityLabel("Open project")
            }
        }
    }

    // MARK: - Photo + floating tool icons (one canvas — tools don't remount it)

    private var canvasWithTools: some View {
        GeometryReader { geo in
            ZStack {
                Color.black
                if geo.size.width > 1, geo.size.height > 1, let composite = vm.compositeImage,
                   composite.size.width > 0, composite.size.height > 0 {
                    ZoomableEditorCanvas(
                        composite: composite,
                        overlay: vm.showIslandOverlay ? vm.overlayImage : nil,
                        scale: $vm.canvasScale,
                        offset: $vm.canvasOffset,
                        tool: vm.tool,
                        eraserRadius: vm.eraserRadius,
                        onTap: { x, y in vm.handleTap(imageX: x, imageY: y) },
                        onMarqueeSelect: { rect in vm.selectIslandsInImageRect(rect) },
                        onEraseStroke: { points in vm.commitEraseStroke(points: points) }
                    )
                    .frame(width: geo.size.width, height: geo.size.height)
                } else if vm.compositeImage == nil {
                    ContentUnavailableView("No composite", systemImage: "photo")
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var toolDock: some View {
        VStack(spacing: 6) {
            if !vm.statusMessage.isEmpty {
                Text(vm.statusMessage)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 4)
                    .background(.ultraThinMaterial, in: Capsule())
                    .lineLimit(1)
            }

            HStack(spacing: 4) {
                toolIcon(
                    systemImage: "hand.draw",
                    selected: vm.tool == .hand,
                    tint: .gray,
                    label: "Hand — pan"
                ) {
                    vm.activateHandTool()
                }

                toolIcon(
                    systemImage: "cursorarrow",
                    selected: vm.tool == .select,
                    tint: .accentColor,
                    label: "Cursor — select islands"
                ) {
                    vm.activateCursorTool()
                }

                toolIcon(
                    systemImage: "eraser",
                    selected: vm.tool == .eraser,
                    tint: .orange,
                    label: "Eraser"
                ) {
                    vm.activateEraserTool()
                }

                Divider().frame(height: 22)

                toolIcon(
                    systemImage: "arrow.uturn.backward",
                    selected: false,
                    tint: .secondary,
                    label: "Undo",
                    enabled: vm.canUndo
                ) {
                    vm.undo()
                }

                toolIcon(
                    systemImage: "arrow.uturn.forward",
                    selected: false,
                    tint: .secondary,
                    label: "Redo",
                    enabled: vm.canRedo
                ) {
                    vm.redo()
                }

                toolIcon(
                    systemImage: vm.showIslandOverlay ? "paintpalette.fill" : "paintpalette",
                    selected: vm.showIslandOverlay,
                    tint: .purple,
                    label: "Island tint"
                ) {
                    vm.showIslandOverlay.toggle()
                    vm.refreshOverlayOnly()
                }

                toolIcon(
                    systemImage: "trash",
                    selected: false,
                    tint: .red,
                    label: "Delete selected",
                    enabled: !vm.selectedIslandIDs.isEmpty
                ) {
                    vm.deleteSelectedIslands()
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())
            .padding(.horizontal, 12)
        }
    }

    private func toolIcon(
        systemImage: String,
        selected: Bool,
        tint: Color,
        label: String,
        enabled: Bool = true,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.body.weight(.semibold))
                .foregroundStyle(selected ? Color.white : (enabled ? Color.primary : Color.secondary.opacity(0.4)))
                .frame(width: 36, height: 36)
                .background(
                    Circle().fill(selected ? tint : Color.clear)
                )
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
        .accessibilityLabel(label)
    }

    private var islandSizeFilter: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Ignore small islands")
                    .font(.caption.weight(.semibold))
                Spacer()
                Text(vm.islandAreaFilter == 0 ? "Off" : "under \(vm.islandAreaFilter) px")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
            Slider(
                value: Binding(
                    get: { Double(vm.islandAreaFilter) },
                    set: { vm.setIslandAreaFilter(Int($0)) }
                ),
                in: 0...8000,
                step: 32
            )
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 8)
        .background(Color(.systemBackground))
    }

    // MARK: - Version history (one layer per prompt)

    private var lists: some View {
        List {
            Section {
                if vm.layers.isEmpty {
                    Text("Each prompt becomes a layer version here.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(versionNodes) { node in
                        versionRow(node.layer, versionLabel: node.label, depth: node.depth)
                            .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                                Button(role: .destructive) {
                                    vm.deleteLayer(node.layer.id)
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                    }
                }
            } header: {
                Text("Version history")
            } footer: {
                Text("Update regenerates that prompt. Subprompt sends this version’s image and mask. Swipe any version left to delete it.")
            }

            Section {
                Text("Select two versions, then run an op.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack {
                    ForEach(BooleanOp.allCases) { op in
                        Button {
                            vm.applyBoolean(op)
                        } label: {
                            VStack(spacing: 2) {
                                Image(systemName: op.systemImage)
                                Text(op.title)
                                    .font(.caption2)
                            }
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .disabled(vm.booleanOpsSelection.count != 2)
                    }
                }
            } header: {
                Text("Boolean Ops")
            }
        }
        .listStyle(.insetGrouped)
        .scrollDismissesKeyboard(.immediately)
        .scrollIndicators(.visible)
    }

    private struct VersionNode: Identifiable {
        let layer: EditLayer
        let label: String
        let depth: Int
        var id: UUID { layer.id }
    }

    /// Newest roots first, with each version’s subprompts nested underneath.
    private var versionNodes: [VersionNode] {
        let roots = vm.layers.filter { $0.parentID == nil }
        var nodes: [VersionNode] = []
        for (offset, root) in roots.reversed().enumerated() {
            let number = roots.count - offset
            appendVersion(root, label: "v\(number)", depth: 0, into: &nodes)
        }
        return nodes
    }

    private func appendVersion(_ layer: EditLayer, label: String, depth: Int, into nodes: inout [VersionNode]) {
        nodes.append(VersionNode(layer: layer, label: label, depth: depth))
        let children = vm.layers.filter { $0.parentID == layer.id }
        for (offset, child) in children.reversed().enumerated() {
            let childLabel = "\(label).\(children.count - offset)"
            appendVersion(child, label: childLabel, depth: depth + 1, into: &nodes)
        }
    }

    private func regionCountLabel(for layer: EditLayer) -> String {
        let total = layer.islands.count
        let shown = layer.islands.filter { $0.area >= vm.islandAreaFilter }.count
        if vm.islandAreaFilter > 0, shown != total {
            return "\(shown) of \(total) regions"
        }
        return "\(total) change region\(total == 1 ? "" : "s")"
    }

    private func versionRow(_ layer: EditLayer, versionLabel: String, depth: Int) -> some View {
        let promptBinding = Binding<String>(
            get: { vm.layers.first { $0.id == layer.id }?.prompt ?? layer.prompt },
            set: { vm.updateLayerPrompt(layer.id, text: $0) }
        )
        let prompt = promptBinding.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines)
        let subpromptBinding = Binding<String>(
            get: { vm.subpromptDrafts[layer.id] ?? "" },
            set: { vm.subpromptDrafts[layer.id] = $0 }
        )
        let subprompt = subpromptBinding.wrappedValue.trimmingCharacters(in: .whitespacesAndNewlines)

        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Button {
                    vm.toggleBooleanSelection(layer.id)
                } label: {
                    Image(systemName: vm.booleanOpsSelection.contains(layer.id) ? "checkmark.circle.fill" : "circle")
                }
                .buttonStyle(.borderless)

                Button {
                    vm.activateLayer(layer.id)
                } label: {
                    Text(versionLabel)
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
                .buttonStyle(.borderless)
                if depth > 0 {
                    Text("Sub")
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.orange.opacity(0.18)))
                }
                if let provider = layer.provider {
                    Text(provider.shortName)
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Capsule().fill(Color.secondary.opacity(0.15)))
                }
                if vm.selectedLayerID == layer.id {
                    Text("Active")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.tint)
                }
                Spacer()
                Button {
                    vm.selectedLayerID = layer.id
                    vm.selectedIslandIDs = []
                    vm.toggleLayerVisibility(layer.id)
                } label: {
                    Image(systemName: layer.isVisible ? "eye" : "eye.slash")
                }
                .buttonStyle(.borderless)
            }

            TextField("Prompt", text: promptBinding, axis: .vertical)
                .font(.subheadline)
                .lineLimit(2...4)
                .textFieldStyle(.roundedBorder)

            HStack {
                Text(regionCountLabel(for: layer))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    Task { await vm.rerunLayer(layer.id) }
                } label: {
                    Label("Update", systemImage: "wand.and.stars")
                        .font(.caption.weight(.semibold))
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(vm.isBusy || prompt.isEmpty)
            }

            TextField("Further change inside this mask", text: subpromptBinding, axis: .vertical)
                .font(.subheadline)
                .lineLimit(2...4)
                .textFieldStyle(.roundedBorder)

            HStack {
                Text("Sends this image + mask")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    Task { await vm.runSubprompt(on: layer.id) }
                } label: {
                    Label("Subprompt", systemImage: "arrow.turn.down.right")
                        .font(.caption.weight(.semibold))
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(vm.isBusy || subprompt.isEmpty)
            }
        }
        .padding(.leading, CGFloat(depth) * 14)
        .padding(.vertical, 4)
    }
}

/// Dismisses the keyboard on any tap that is not inside a text field, without stealing the tap.
private struct KeyboardDismissInstaller: UIViewRepresentable {
    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        view.isUserInteractionEnabled = false
        context.coordinator.install()
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        context.coordinator.install()
    }

    static func dismantleUIView(_ uiView: UIView, coordinator: Coordinator) {
        coordinator.remove()
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        private let tap = UITapGestureRecognizer()
        private weak var installedOn: UIWindow?

        func install() {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                guard let window = Self.keyWindow, self.installedOn !== window else { return }
                self.remove()
                self.tap.addTarget(self, action: #selector(self.dismiss))
                self.tap.cancelsTouchesInView = false
                self.tap.delegate = self
                window.addGestureRecognizer(self.tap)
                self.installedOn = window
            }
        }

        func remove() {
            if let installedOn {
                installedOn.removeGestureRecognizer(tap)
            }
            installedOn = nil
        }

        @objc private func dismiss() {
            UIApplication.shared.sendAction(
                #selector(UIResponder.resignFirstResponder),
                to: nil,
                from: nil,
                for: nil
            )
        }

        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
            var view = touch.view
            while let current = view {
                if current is UITextField || current is UITextView {
                    return false
                }
                view = current.superview
            }
            return true
        }

        func gestureRecognizer(
            _ gestureRecognizer: UIGestureRecognizer,
            shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer
        ) -> Bool {
            true
        }

        private static var keyWindow: UIWindow? {
            UIApplication.shared.connectedScenes
                .compactMap { $0 as? UIWindowScene }
                .flatMap(\.windows)
                .first(where: \.isKeyWindow)
        }
    }
}
