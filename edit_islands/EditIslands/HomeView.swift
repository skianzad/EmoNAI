import PhotosUI
import SwiftUI

struct HomeView: View {
    @StateObject private var vm = EditIslandsViewModel()

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    HStack {
                        PhotosPicker(selection: $vm.originalItem, matching: .images) {
                            Label("Choose Original", systemImage: "photo")
                        }
                        .onChange(of: vm.originalItem) { _, _ in
                            Task { await vm.loadOriginal() }
                        }
                        Spacer()
                        if let img = vm.originalImage {
                            Image(uiImage: img)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 64, height: 64)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                } header: {
                    Text("Base Image")
                } footer: {
                    Text("Final image = original + enabled islands from each edit layer.")
                }

                Section {
                    Picker("Editor", selection: $vm.selectedProvider) {
                        ForEach(EditProvider.allCases) { provider in
                            Label(provider.displayName, systemImage: provider.systemImage)
                                .tag(provider)
                        }
                    }
                    .pickerStyle(.navigationLink)

                    HStack {
                        Image(systemName: vm.selectedProviderHasKey ? "checkmark.seal.fill" : "key.slash")
                            .foregroundStyle(vm.selectedProviderHasKey ? .green : .orange)
                        Text(vm.selectedProviderHasKey
                             ? "\(vm.selectedProvider.displayName) key ready"
                             : "No key for \(vm.selectedProvider.displayName)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("API Keys") { vm.showAPIKeys = true }
                            .font(.caption)
                    }

                    TextField("Prompt for this edit", text: $vm.promptText, axis: .vertical)
                        .lineLimit(2...4)

                    Button {
                        Task { await vm.runSelectedProviderEdit() }
                    } label: {
                        Label(
                            "Run \(vm.selectedProvider.shortName) → Islands",
                            systemImage: "wand.and.stars"
                        )
                    }
                    .disabled(
                        vm.originalImage == nil
                            || vm.promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || vm.isBusy
                    )
                } header: {
                    Text("Generate Edit Layer")
                } footer: {
                    Text("Switch between Nano Banana, Qwen, Seedream, and GPT. Each run diffs the result into a new island layer.")
                }

                Section {
                    HStack {
                        PhotosPicker(selection: $vm.pendingEditedItem, matching: .images) {
                            Label("Import Edited Result", systemImage: "photo.badge.plus")
                        }
                        .onChange(of: vm.pendingEditedItem) { _, _ in
                            Task { await vm.loadPendingEdited() }
                        }
                        Spacer()
                        if let img = vm.pendingEditedImage {
                            Image(uiImage: img)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 64, height: 64)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }

                    Button {
                        vm.addLayerFromPendingEdit(provider: nil)
                    } label: {
                        Label("Diff Import → New Layer", systemImage: "square.stack.3d.up")
                    }
                    .disabled(vm.originalImage == nil || vm.pendingEditedImage == nil || vm.isBusy)
                } header: {
                    Text("Or Import Result")
                } footer: {
                    Text("Optional: paste a result from outside the app, then segment islands locally.")
                }

                Section("Diff Sensitivity") {
                    HStack {
                        Text("SSIM threshold")
                        Spacer()
                        Text(String(format: "%.2f", vm.ssimThreshold))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    Slider(value: $vm.ssimThreshold, in: 0.5...0.98, step: 0.01)

                    Stepper("Min island area: \(vm.minIslandArea)px", value: $vm.minIslandArea, in: 16...2000, step: 16)
                }

                if !vm.layers.isEmpty {
                    Section("Layers (\(vm.layers.count))") {
                        ForEach(vm.layers) { layer in
                            NavigationLink {
                                EditorView(vm: vm)
                                    .onAppear {
                                        vm.selectedLayerID = layer.id
                                        vm.selectedIslandIDs = []
                                        vm.refreshOverlayOnly()
                                    }
                            } label: {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(layer.displayTitle)
                                        .lineLimit(2)
                                    Text("\(layer.islands.count) islands · \(layer.isVisible ? "visible" : "hidden")")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                        .onDelete { indexSet in
                            for i in indexSet {
                                vm.deleteLayer(vm.layers[i].id)
                            }
                        }
                    }
                }

                if vm.compositeImage != nil {
                    Section {
                        NavigationLink {
                            EditorView(vm: vm)
                        } label: {
                            Label("Open Island Editor", systemImage: "square.stack.3d.forward.dottedline")
                        }
                    }
                }

                Section {
                    Button {
                        vm.showProjectImporter = true
                    } label: {
                        Label("Open Project", systemImage: "folder")
                    }
                    Button {
                        vm.beginProjectExport()
                    } label: {
                        Label("Save Project", systemImage: "square.and.arrow.up")
                    }
                    .disabled(vm.originalBuffer == nil)
                } header: {
                    Text("Project")
                } footer: {
                    Text("Saves the original and every version: prompts, masks, islands, and subprompts. The down-arrow in the editor saves only the flattened photo.")
                }

                if !vm.statusMessage.isEmpty {
                    Section {
                        Text(vm.statusMessage)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("EditIslands")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        vm.showAPIKeys = true
                    } label: {
                        Image(systemName: "key.fill")
                    }
                }
            }
            .sheet(isPresented: $vm.showAPIKeys) {
                APIKeysView()
            }
            .fileExporter(
                isPresented: $vm.showProjectExporter,
                document: vm.projectDocument,
                contentType: .data,
                defaultFilename: "EditIslands.editislands"
            ) { result in
                if case .failure(let error) = result {
                    vm.statusMessage = error.localizedDescription
                } else if case .success = result {
                    vm.statusMessage = "Project saved — \(vm.layers.count) version\(vm.layers.count == 1 ? "" : "s")"
                }
            }
            .fileImporter(
                isPresented: $vm.showProjectImporter,
                allowedContentTypes: [.item]
            ) { result in
                switch result {
                case .success(let url):
                    vm.loadProject(from: url)
                case .failure(let error):
                    vm.statusMessage = error.localizedDescription
                }
            }
            .overlay {
                if vm.isBusy {
                    ProgressView(vm.busyLabel)
                        .padding()
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                }
            }
        }
    }
}

#Preview {
    HomeView()
}
