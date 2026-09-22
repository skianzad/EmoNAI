import PhotosUI
import SwiftUI

struct HomeView: View {
    @StateObject private var vm = SmartPhotoViewModel()
    @State private var showResults = false

    var body: some View {
        NavigationStack {
            Form {
                // MARK: - Reference Photo
                Section {
                    HStack {
                        PhotosPicker(
                            selection: $vm.referenceItem,
                            matching: .images
                        ) {
                            Label("Choose Reference", systemImage: "person.crop.square")
                        }
                        .onChange(of: vm.referenceItem) { _, _ in
                            Task { await vm.loadReference() }
                        }

                        Spacer()

                        if let img = vm.referenceFaceCrop ?? vm.referenceImage {
                            Image(uiImage: img)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 60, height: 60)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                } header: {
                    Text("Reference Photo")
                }

                // MARK: - Candidate Photos
                Section {
                    Toggle(isOn: $vm.livePhotoMode) {
                        Label("Live Photo mode", systemImage: "livephoto")
                    }
                    .onChange(of: vm.livePhotoMode) { _, enabled in
                        vm.candidateItems = []
                        vm.livePhotoItem = nil
                        vm.candidateImages = []
                        vm.livePhotoError = nil
                        if !enabled { vm.statusMessage = "" }
                    }

                    if vm.livePhotoMode {
                        HStack {
                            PhotosPicker(
                                selection: $vm.livePhotoItem,
                                matching: .livePhotos,
                                photoLibrary: .shared()
                            ) {
                                Label("Choose Live Photo", systemImage: "livephoto")
                            }
                            .onChange(of: vm.livePhotoItem) { _, _ in
                                Task { await vm.loadLivePhotoFrames() }
                            }

                            Spacer()

                            if vm.isExtracting {
                                ProgressView()
                                    .controlSize(.small)
                            } else if !vm.candidateImages.isEmpty {
                                Text("\(vm.candidateImages.count) frames")
                                    .foregroundStyle(.secondary)
                            }
                        }

                        if let err = vm.livePhotoError {
                            Text(err)
                                .font(.caption)
                                .foregroundStyle(.red)
                        }
                    } else {
                        HStack {
                            PhotosPicker(
                                selection: $vm.candidateItems,
                                maxSelectionCount: 50,
                                matching: .images
                            ) {
                                Label("Choose Candidates", systemImage: "photo.on.rectangle.angled")
                            }
                            .onChange(of: vm.candidateItems) { _, _ in
                                Task { await vm.loadCandidates() }
                            }

                            Spacer()

                            if !vm.candidateImages.isEmpty {
                                Text("\(vm.candidateImages.count) photos")
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    if !vm.candidateImages.isEmpty {
                        ScrollView(.horizontal, showsIndicators: false) {
                            LazyHStack(spacing: 6) {
                                ForEach(vm.candidateImages, id: \.id) { cand in
                                    Image(uiImage: cand.preview)
                                        .resizable()
                                        .scaledToFill()
                                        .frame(width: 50, height: 50)
                                        .clipShape(RoundedRectangle(cornerRadius: 6))
                                }
                            }
                            .padding(.vertical, 4)
                        }
                    }
                } header: {
                    Text(vm.livePhotoMode ? "Live Photo Frames" : "Candidate Photos")
                } footer: {
                    if vm.livePhotoMode {
                        Text("Pick one Live Photo. Every video frame is extracted and shown here, then ranked against the reference.")
                    }
                }

                // MARK: - Approaches
                Section {
                    Toggle(isOn: $vm.approaches.landmarks) {
                        VStack(alignment: .leading) {
                            Text("A — Landmarks")
                                .font(.subheadline.bold())
                            Text("Head pose + geometric ratios (no model)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    Toggle(isOn: $vm.approaches.posterV2) {
                        VStack(alignment: .leading) {
                            HStack {
                                Text("B — POSTER V2")
                                    .font(.subheadline.bold())
                                if !ApproachB.isAvailable {
                                    Text("(no model)")
                                        .font(.caption2)
                                        .foregroundStyle(.red)
                                }
                            }
                            Text("Learned expression embedding (768-d)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .disabled(!ApproachB.isAvailable)

                    Toggle(isOn: $vm.approaches.fecNet) {
                        VStack(alignment: .leading) {
                            HStack {
                                Text("C — FECNet")
                                    .font(.subheadline.bold())
                                if !ApproachC.isAvailable {
                                    Text("(no model)")
                                        .font(.caption2)
                                        .foregroundStyle(.red)
                                }
                            }
                            Text("Similarity embedding (16-d)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .disabled(!ApproachC.isAvailable)
                } header: {
                    Text("Approaches")
                }

                // MARK: - Run
                Section {
                    Button {
                        Task {
                            await vm.run()
                            showResults = true
                        }
                    } label: {
                        HStack {
                            Spacer()
                            if vm.isRunning {
                                ProgressView()
                                    .controlSize(.small)
                                    .padding(.trailing, 8)
                                Text(vm.statusMessage)
                            } else {
                                Label(vm.livePhotoMode ? "Find Closest Frame" : "Run Comparison", systemImage: "play.fill")
                                    .font(.headline)
                            }
                            Spacer()
                        }
                    }
                    .disabled(!canRun)

                    if vm.isRunning {
                        ProgressView(value: vm.progress)
                            .progressViewStyle(.linear)
                    }
                }
            }
            .navigationTitle("SmartPhoto")
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(isPresented: $showResults) {
                ResultsView(
                    referenceImage: vm.referenceFaceCrop ?? vm.referenceImage,
                    results: vm.results)
            }
        }
    }

    private var canRun: Bool {
        vm.referenceImage != nil
        && !vm.candidateImages.isEmpty
        && vm.approaches.any
        && !vm.isRunning
        && !vm.isExtracting
    }
}

#Preview {
    HomeView()
}
