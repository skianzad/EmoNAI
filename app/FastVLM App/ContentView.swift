//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import AVFoundation
import MLXLMCommon
import SwiftUI
import Video

// support swift 6
extension CVImageBuffer: @unchecked @retroactive Sendable {}
extension CMSampleBuffer: @unchecked @retroactive Sendable {}

// delay between frames -- controls the frame rate of the updates
let FRAME_DELAY = Duration.milliseconds(1)

struct ContentView: View {
    @State private var camera = CameraController()
    @State private var model = FastVLMModel()

    /// stream of frames -> VideoFrameView, see distributeVideoFrames
    @State private var framesToDisplay: AsyncStream<CVImageBuffer>?

    @State private var prompt = "Describe the image in English."
    @State private var promptSuffix = "Output should be brief, about 15 words or less."
    @State private var arrowsOnFacePrompt = "Rate this face's arousal (calm to excited) and valence (negative to positive)."
    @State private var arrowsOnFaceSuffix = "Format: arousal <low/medium/high>, valence <negative/neutral/positive>. Under 10 words."

    @State private var isShowingInfo: Bool = false

    @State private var selectedCameraType: CameraType = .continuous
    @State private var isEditingPrompt: Bool = false

    /// When enabled, each frame is first passed through the face landmarker;
    /// only the cropped face region is sent to the VLM.  Frames with no
    /// detected face are skipped (continuous mode) or fall back to the full
    /// frame (single mode).
    @State private var faceLandmarkModeEnabled: Bool = false
    private let faceLandmarker = FaceLandmarkerService()

    /// Rolling history of landmark snapshots for the ghost trail.
    @State private var displayFaceHistory: [FaceLandmarkDisplayResult] = []
    /// How many ghost snapshots to keep (user-adjustable, 1–10).
    @State private var maxGhostCount: Int = 5
    /// Seconds between freezing a new ghost snapshot (user-adjustable).
    @State private var snapshotInterval: Double = 1.0

    /// Rolling history of VLM emotion responses for change-detection prompting.
    @State private var emotionHistory: [String] = []

    /// Face landmark display mode when face landmark mode is enabled.
    enum FaceOverlayMode: Int, CaseIterable {
        case ghosts = 0      // ghost trail of past landmark snapshots
        case arrows = 1      // current landmarks + movement arrows
        case arrowsOnFace = 2 // faded camera frame + movement arrows (sent to VLM)
    }
    @State private var faceOverlayMode: FaceOverlayMode = .ghosts

    /// Last image sent to the VLM, shown as a debug thumbnail.
    @State private var lastVLMInputImage: CGImage? = nil
    @State private var showVLMDebugThumbnail: Bool = false

    var toolbarItemPlacement: ToolbarItemPlacement {
        var placement: ToolbarItemPlacement = .navigation
        #if os(iOS)
        placement = .topBarLeading
        #endif
        return placement
    }
    
    var statusTextColor : Color {
        return model.evaluationState == .processingPrompt ? .black : .white
    }
    
    var statusBackgroundColor : Color {
        switch model.evaluationState {
        case .idle:
            return .gray
        case .generatingResponse:
            return .green
        case .processingPrompt:
            return .yellow
        }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    VStack(alignment: .leading, spacing: 10.0) {
                        Picker("Camera Type", selection: $selectedCameraType) {
                            ForEach(CameraType.allCases, id: \.self) { cameraType in
                                Text(cameraType.rawValue.capitalized).tag(cameraType)
                            }
                        }
                        // Prevent macOS from adding a text label for the picker
                        .labelsHidden()
                        .pickerStyle(.segmented)
                        .onChange(of: selectedCameraType) { _, _ in
                            // Cancel any in-flight requests when switching modes
                            model.cancel()
                        }

                        Toggle(isOn: $faceLandmarkModeEnabled) {
                            Label("Face Landmark Mode", systemImage: "face.dashed")
                                .font(.subheadline)
                        }
                        .onChange(of: faceLandmarkModeEnabled) { _, enabled in
                            if !enabled {
                                displayFaceHistory.removeAll()
                                emotionHistory.removeAll()
                                model.maxTokens = 240
                            }
                            if enabled {
                                prompt = "What emotion is this face showing?"
                                promptSuffix = "One word: happy, sad, angry, surprised, fearful, disgusted, or neutral. Under 15 words."
                                model.maxTokens = 20
                            }
                        }

                        if faceLandmarkModeEnabled {
                            HStack {
                                Text("Ghosts: \(maxGhostCount)")
                                    .font(.caption).monospacedDigit()
                                Stepper("", value: $maxGhostCount, in: 1...10)
                                    .labelsHidden()

                                Spacer()

                                Text("Interval: \(String(format: "%.1fs", snapshotInterval))")
                                    .font(.caption).monospacedDigit()
                                Stepper("", value: $snapshotInterval, in: 0.5...5.0, step: 0.5)
                                    .labelsHidden()

                                Spacer()

                                Picker("Overlay", selection: $faceOverlayMode) {
                                    Image(systemName: "circle.grid.3x3")
                                        .tag(FaceOverlayMode.ghosts)
                                    Image(systemName: "arrow.up.and.down.and.arrow.left.and.right")
                                        .tag(FaceOverlayMode.arrows)
                                    Image(systemName: "person.fill.viewfinder")
                                        .tag(FaceOverlayMode.arrowsOnFace)
                                }
                                .pickerStyle(.segmented)
                                .frame(maxWidth: 140)

                                Button {
                                    showVLMDebugThumbnail.toggle()
                                } label: {
                                    Image(systemName: "eye.circle")
                                        .foregroundStyle(showVLMDebugThumbnail ? .blue : .gray)
                                        .font(.title3)
                                }
                                .buttonStyle(.plain)
                            }
                            .onChange(of: maxGhostCount) { _, newMax in
                                if displayFaceHistory.count > newMax {
                                    displayFaceHistory.removeFirst(
                                        displayFaceHistory.count - newMax)
                                }
                            }
                        }

                        if let framesToDisplay {
                            VideoFrameView(
                                frames: framesToDisplay,
                                cameraType: selectedCameraType,
                                action: { frame in
                                    processSingleFrame(frame)
                                })
                                // Because we're using the AVCaptureSession preset
                                // `.vga640x480`, we can assume this aspect ratio
                                .aspectRatio(4/3, contentMode: .fit)
                                #if os(macOS)
                                .frame(maxWidth: 750)
                                #endif
                                .overlay {
                                    if faceLandmarkModeEnabled,
                                       faceOverlayMode != .arrowsOnFace {
                                        Color.white
                                    }
                                }
                                .overlay {
                                    if faceLandmarkModeEnabled,
                                       faceOverlayMode == .arrowsOnFace {
                                        Color.black.opacity(0.3)
                                    }
                                }
                                .overlay {
                                    if faceLandmarkModeEnabled,
                                       !displayFaceHistory.isEmpty {
                                        switch faceOverlayMode {
                                        case .ghosts:
                                            FaceLandmarkOverlay(
                                                history: displayFaceHistory)
                                        case .arrows:
                                            FaceLandmarkOverlay(
                                                history: [displayFaceHistory.last!],
                                                referenceFace: displayFaceHistory.count >= 2
                                                    ? displayFaceHistory.first!.landmarks.first
                                                    : nil)
                                        case .arrowsOnFace:
                                            EmptyView()
                                        }
                                    }
                                }
                                .overlay {
                                    if faceLandmarkModeEnabled,
                                       faceOverlayMode != .ghosts,
                                       displayFaceHistory.count >= 2 {
                                        FaceMovementArrowsOverlay(
                                            oldSnapshot: displayFaceHistory.first!,
                                            newSnapshot: displayFaceHistory.last!)
                                    }
                                }
                                .overlay(alignment: .bottomTrailing) {
                                    if showVLMDebugThumbnail,
                                       let cgImg = lastVLMInputImage {
                                        Image(decorative: cgImg, scale: 1)
                                            .resizable()
                                            .aspectRatio(contentMode: .fit)
                                            .frame(width: 120, height: 120)
                                            .border(Color.white, width: 2)
                                            .shadow(radius: 4)
                                            .padding(8)
                                    }
                                }
                                .overlay(alignment: .top) {
                                    if !model.promptTime.isEmpty {
                                        Text("TTFT \(model.promptTime)")
                                            .font(.caption)
                                            .foregroundStyle(.white)
                                            .monospaced()
                                            .padding(.vertical, 4.0)
                                            .padding(.horizontal, 6.0)
                                            .background(alignment: .center) {
                                                RoundedRectangle(cornerRadius: 8)
                                                    .fill(Color.black.opacity(0.6))
                                            }
                                            .padding(.top)
                                    }
                                }
                                #if !os(macOS)
                                .overlay(alignment: .topTrailing) {
                                    CameraControlsView(
                                        backCamera: $camera.backCamera,
                                        device: $camera.device,
                                        devices: $camera.devices)
                                    .padding()
                                }
                                #endif
                                .overlay(alignment: .bottom) {
                                    if selectedCameraType == .continuous {
                                        Group {
                                            if model.evaluationState == .processingPrompt {
                                                HStack {
                                                    ProgressView()
                                                        .tint(self.statusTextColor)
                                                        .controlSize(.small)

                                                    Text(model.evaluationState.rawValue)
                                                }
                                            } else if model.evaluationState == .idle {
                                                HStack(spacing: 6.0) {
                                                    Image(systemName: "clock.fill")
                                                        .font(.caption)

                                                    Text(model.evaluationState.rawValue)
                                                }
                                            }
                                            else {
                                                // I'm manually tweaking the spacing to
                                                // better match the spacing with ProgressView
                                                HStack(spacing: 6.0) {
                                                    Image(systemName: "lightbulb.fill")
                                                        .font(.caption)

                                                    Text(model.evaluationState.rawValue)
                                                }
                                            }
                                        }
                                        .foregroundStyle(self.statusTextColor)
                                        .font(.caption)
                                        .bold()
                                        .padding(.vertical, 6.0)
                                        .padding(.horizontal, 8.0)
                                        .background(self.statusBackgroundColor)
                                        .clipShape(.capsule)
                                        .padding(.bottom)
                                    }
                                }
                                #if os(macOS)
                                .frame(maxWidth: .infinity)
                                .frame(minWidth: 500)
                                .frame(minHeight: 375)
                                #endif
                        }
                    }
                }
                .listRowInsets(EdgeInsets())
                .listRowBackground(Color.clear)
                .listRowSeparator(.hidden)

                promptSections

                Section {
                    if model.output.isEmpty && model.running {
                        ProgressView()
                            .controlSize(.large)
                            .frame(maxWidth: .infinity)
                    } else {
                        ScrollView {
                            Text(model.output)
                                .foregroundStyle(isEditingPrompt ? .secondary : .primary)
                                .textSelection(.enabled)
                                #if os(macOS)
                                .font(.headline)
                                .fontWeight(.regular)
                                #endif
                        }
                        .frame(minHeight: 50.0, maxHeight: 200.0)
                    }
                } header: {
                    Text("Response")
                        #if os(macOS)
                        .font(.headline)
                        .padding(.bottom, 2.0)
                        #endif
                }

                #if os(macOS)
                Spacer()
                #endif
            }
            
            #if os(iOS)
            .listSectionSpacing(0)
            #elseif os(macOS)
            .padding()
            #endif
            .task {
                camera.start()
            }
            .task {
                await model.load()
            }

            #if !os(macOS)
            .onAppear {
                // Prevent the screen from dimming or sleeping due to inactivity
                UIApplication.shared.isIdleTimerDisabled = true
            }
            .onDisappear {
                // Resumes normal idle timer behavior
                UIApplication.shared.isIdleTimerDisabled = false
            }
            #endif

            // task to distribute video frames -- this will cancel
            // and restart when the view is on/off screen.  note: it is
            // important that this is here (attached to the VideoFrameView)
            // rather than the outer view because this has the correct lifecycle
            .task {
                if Task.isCancelled {
                    return
                }

                await distributeVideoFrames()
            }

            .navigationTitle("FastVLM")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: toolbarItemPlacement) {
                    Button {
                        isShowingInfo.toggle()
                    }
                    label: {
                        Image(systemName: "info.circle")
                    }
                }

                ToolbarItem(placement: .primaryAction) {
                    if isEditingPrompt {
                        Button {
                            isEditingPrompt.toggle()
                        }
                        label: {
                            Text("Done")
                                .fontWeight(.bold)
                        }
                    }
                    else {
                        Menu {
                            Button("Describe image") {
                                prompt = "Describe the image in English."
                                promptSuffix = "Output should be brief, about 15 words or less."
                            }
                            Button("Facial expression") {
                                prompt = "What is this person's facial expression?"
                                promptSuffix = "Output only one or two words."
                            }
                            Button("Face landmarks — emotion") {
                                faceLandmarkModeEnabled = true
                                prompt = "Name the exact emotion on this face."
                                promptSuffix = "One or two words only. Example: happy. Do not describe the image."
                            }
                            Button("Face landmarks — movement") {
                                faceLandmarkModeEnabled = true
                                prompt = "Name the exact emotion on this face and any head tilt or turn."
                                promptSuffix = "Format: <emotion>, <movement>. Example: happy, tilting left. Do not describe the image."
                            }
                            Button("Read text") {
                                prompt = "What is written in this image?"
                                promptSuffix = "Output only the text in the image."
                            }
                            #if !os(macOS)
                            Button("Customize...") {
                                isEditingPrompt.toggle()
                            }
                            #endif
                        } label: { Text("Prompts") }
                    }
                }
            }
            .sheet(isPresented: $isShowingInfo) {
                InfoView()
            }
        }
    }

    var promptSummary: some View {
        Section("Prompt") {
            VStack(alignment: .leading, spacing: 4.0) {
                let isAOF = faceLandmarkModeEnabled && faceOverlayMode == .arrowsOnFace
                let activePrompt = isAOF ? arrowsOnFacePrompt : prompt
                let activeSuffix = isAOF ? arrowsOnFaceSuffix : promptSuffix
                let trimmedPrompt = activePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedPrompt.isEmpty {
                    Text(trimmedPrompt)
                        .foregroundStyle(.secondary)
                }

                let trimmedSuffix = activeSuffix.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedSuffix.isEmpty {
                    Text(trimmedSuffix)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    var promptForm: some View {
        Group {
            #if os(iOS)
            if faceLandmarkModeEnabled && faceOverlayMode == .arrowsOnFace {
                Section("Face+Arrows Prompt") {
                    TextEditor(text: $arrowsOnFacePrompt)
                        .frame(minHeight: 38)
                }
                Section("Face+Arrows Suffix") {
                    TextEditor(text: $arrowsOnFaceSuffix)
                        .frame(minHeight: 38)
                }
            } else {
                Section("Prompt") {
                    TextEditor(text: $prompt)
                        .frame(minHeight: 38)
                }
                Section("Prompt Suffix") {
                    TextEditor(text: $promptSuffix)
                        .frame(minHeight: 38)
                }
            }
            #elseif os(macOS)
            Section {
                if faceLandmarkModeEnabled && faceOverlayMode == .arrowsOnFace {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading) {
                            Text("Face+Arrows Prompt")
                                .font(.headline)

                            TextEditor(text: $arrowsOnFacePrompt)
                                .frame(height: 38)
                                .padding(.horizontal, 8.0)
                                .padding(.vertical, 10.0)
                                .background(Color(.textBackgroundColor))
                                .cornerRadius(10.0)
                        }

                        VStack(alignment: .leading) {
                            Text("Face+Arrows Suffix")
                                .font(.headline)

                            TextEditor(text: $arrowsOnFaceSuffix)
                                .frame(height: 38)
                                .padding(.horizontal, 8.0)
                                .padding(.vertical, 10.0)
                                .background(Color(.textBackgroundColor))
                                .cornerRadius(10.0)
                        }
                    }
                } else {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading) {
                            Text("Prompt")
                                .font(.headline)

                            TextEditor(text: $prompt)
                                .frame(height: 38)
                                .padding(.horizontal, 8.0)
                                .padding(.vertical, 10.0)
                                .background(Color(.textBackgroundColor))
                                .cornerRadius(10.0)
                        }

                        VStack(alignment: .leading) {
                            Text("Prompt Suffix")
                                .font(.headline)

                            TextEditor(text: $promptSuffix)
                                .frame(height: 38)
                                .padding(.horizontal, 8.0)
                                .padding(.vertical, 10.0)
                                .background(Color(.textBackgroundColor))
                                .cornerRadius(10.0)
                        }
                    }
                }
            }
            .padding(.vertical)
            #endif
        }
    }

    var promptSections: some View {
        Group {
            #if os(iOS)
            if isEditingPrompt {
                promptForm
            }
            else {
                promptSummary
            }
            #elseif os(macOS)
            promptForm
            #endif
        }
    }

    func analyzeVideoFrames(_ frames: AsyncStream<CVImageBuffer>) async {
        var vlmLandmarkHistory: [FaceLandmarkDisplayResult] = []
        var lastVLMSnapshotDate = Date.distantPast

        for await frame in frames {
            let imageForVLM: CIImage
            let isFaceMode = await MainActor.run { faceLandmarkModeEnabled }
            let currentMaxGhosts = await MainActor.run { maxGhostCount }
            let currentInterval = await MainActor.run { snapshotInterval }

            if isFaceMode {
                guard let detection = faceLandmarker.detectObservation(in: frame) else {
                    print("[VLM DEBUG] no face detected — skipping")
                    continue
                }

                let now = Date()
                let shouldFreeze = now.timeIntervalSince(lastVLMSnapshotDate) >= currentInterval
                if shouldFreeze { lastVLMSnapshotDate = now }

                if shouldFreeze || vlmLandmarkHistory.isEmpty {
                    vlmLandmarkHistory.append(detection)
                    if vlmLandmarkHistory.count > currentMaxGhosts {
                        vlmLandmarkHistory.removeFirst(
                            vlmLandmarkHistory.count - currentMaxGhosts)
                    }
                } else {
                    vlmLandmarkHistory[vlmLandmarkHistory.count - 1] = detection
                }

                let overlayMode = await MainActor.run { faceOverlayMode }
                let arrowsOn = overlayMode == .arrows || overlayMode == .arrowsOnFace
                if arrowsOn && vlmLandmarkHistory.count >= 2 {
                    let hasMovement = FaceMovementArrowsOverlay.hasMeaningfulMovement(
                        old: vlmLandmarkHistory.first!,
                        new: vlmLandmarkHistory.last!)
                    if !hasMovement {
                        print("[VLM DEBUG] no meaningful movement — skipping VLM")
                        continue
                    }
                }

                let rendered: CIImage?
                if overlayMode == .arrowsOnFace {
                    rendered = FaceLandmarkOverlay.renderCroppedFace(
                        frame: frame,
                        snapshot: vlmLandmarkHistory.last!)
                } else {
                    rendered = FaceLandmarkOverlay.renderToImage(
                        history: vlmLandmarkHistory,
                        drawArrows: arrowsOn)
                }

                guard let rendered else {
                    print("[VLM DEBUG] renderToImage failed")
                    continue
                }
                imageForVLM = rendered
                print("[VLM DEBUG] rendered mode=\(overlayMode), \(vlmLandmarkHistory.count) ghosts, extent: \(rendered.extent)")
                debugSaveImage(rendered, tag: "vlm_input")
                let ciCtx = CIContext()
                if let cg = ciCtx.createCGImage(rendered, from: rendered.extent) {
                    await MainActor.run { lastVLMInputImage = cg }
                }
            } else {
                vlmLandmarkHistory.removeAll()
                imageForVLM = CIImage(cvPixelBuffer: frame)
            }

            let fullPrompt: String
            if isFaceMode {
                let emotions = await MainActor.run { emotionHistory }
                let overlayMode2 = await MainActor.run { faceOverlayMode }
                let aofPrompt = await MainActor.run { arrowsOnFacePrompt }
                let aofSuffix = await MainActor.run { arrowsOnFaceSuffix }
                var basePrompt: String
                if overlayMode2 == .arrowsOnFace, vlmLandmarkHistory.count >= 2 {
                    let movement = FaceMovementArrowsOverlay.describeMovements(
                        old: vlmLandmarkHistory.first!,
                        new: vlmLandmarkHistory.last!)
                    basePrompt = "Detected muscle movement: \(movement). \(aofPrompt)"
                } else {
                    basePrompt = prompt
                }
                let suffix = overlayMode2 == .arrowsOnFace ? aofSuffix : promptSuffix
                if !emotions.isEmpty {
                    let prev = emotions.suffix(3).joined(separator: ", ")
                    fullPrompt = "\(basePrompt) Previous: \(prev). \(suffix)"
                } else {
                    fullPrompt = "\(basePrompt) \(suffix)"
                }
                print("[VLM DEBUG] prompt: \(fullPrompt)")
            } else {
                fullPrompt = "\(prompt) \(promptSuffix)"
            }

            let userInput = UserInput(
                prompt: .text(fullPrompt),
                images: [.ciImage(imageForVLM)]
            )

            let t = await model.generate(userInput)
            _ = await t.result

            if isFaceMode {
                let rawEmotion = await MainActor.run {
                    model.output.trimmingCharacters(in: .whitespacesAndNewlines)
                }
                print("[VLM DEBUG] response: \(rawEmotion)")
                let emotion = Self.extractEmotion(from: rawEmotion)
                if !emotion.isEmpty {
                    await MainActor.run {
                        emotionHistory.append(emotion)
                        if emotionHistory.count > 5 {
                            emotionHistory.removeFirst(emotionHistory.count - 5)
                        }
                    }
                }
            }

            do {
                try await Task.sleep(for: FRAME_DELAY)
            } catch { return }
        }
    }

    /// Runs MediaPipe face-landmark detection on display frames and maintains
    /// a rolling history for the ghost-trail overlay.  A new ghost is frozen
    /// every `snapshotInterval` seconds; the newest entry is live-updated
    /// every frame.
    func detectDisplayLandmarks(_ frames: AsyncStream<CVImageBuffer>) async {
        var lastSnapshotDate = Date.distantPast

        for await frame in frames {
            guard faceLandmarkModeEnabled,
                  let result = faceLandmarker.detectObservation(in: frame) else {
                continue
            }

            let now = Date()
            let interval = await MainActor.run { snapshotInterval }
            let shouldFreeze = now.timeIntervalSince(lastSnapshotDate) >= interval

            if shouldFreeze {
                lastSnapshotDate = now
            }

            await MainActor.run {
                let maxCount = maxGhostCount
                if shouldFreeze || displayFaceHistory.isEmpty {
                    displayFaceHistory.append(result)
                    if displayFaceHistory.count > maxCount {
                        displayFaceHistory.removeFirst(
                            displayFaceHistory.count - maxCount)
                    }
                } else {
                    displayFaceHistory[displayFaceHistory.count - 1] = result
                }
            }
        }
    }

    func distributeVideoFrames() async {
        // attach a stream to the camera -- this code will read this
        let frames = AsyncStream<CMSampleBuffer>(bufferingPolicy: .bufferingNewest(1)) {
            camera.attach(continuation: $0)
        }

        let (framesToDisplay, framesToDisplayContinuation) = AsyncStream.makeStream(
            of: CVImageBuffer.self,
            bufferingPolicy: .bufferingNewest(1)
        )
        self.framesToDisplay = framesToDisplay

        // Only create analysis stream if in continuous mode
        let (framesToAnalyze, framesToAnalyzeContinuation) = AsyncStream.makeStream(
            of: CVImageBuffer.self,
            bufferingPolicy: .bufferingNewest(1)
        )

        // Always feed landmark detection regardless of camera mode
        let (framesToLandmark, framesToLandmarkContinuation) = AsyncStream.makeStream(
            of: CVImageBuffer.self,
            bufferingPolicy: .bufferingNewest(1)
        )

        // set up structured tasks (important -- this means the child tasks
        // are cancelled when the parent is cancelled)
        async let distributeFrames: () = {
            for await sampleBuffer in frames {
                if let frame = sampleBuffer.imageBuffer {
                    framesToDisplayContinuation.yield(frame)
                    framesToLandmarkContinuation.yield(frame)
                    // Only send frames for analysis in continuous mode
                    if await selectedCameraType == .continuous {
                        framesToAnalyzeContinuation.yield(frame)
                    }
                }
            }

            // detach from the camera controller and feed to the video view
            await MainActor.run {
                self.framesToDisplay = nil
                self.camera.detatch()
            }

            framesToDisplayContinuation.finish()
            framesToAnalyzeContinuation.finish()
            framesToLandmarkContinuation.finish()
        }()

        async let detectLandmarks: () = detectDisplayLandmarks(framesToLandmark)

        // Only analyze frames if in continuous mode
        if selectedCameraType == .continuous {
            async let analyze: () = analyzeVideoFrames(framesToAnalyze)
            await distributeFrames
            await analyze
        } else {
            await distributeFrames
        }
        await detectLandmarks
    }

    /// Perform FastVLM inference on a single frame.
    func processSingleFrame(_ frame: CVImageBuffer) {
        Task { @MainActor in
            model.output = ""
        }

        let isFaceMode = faceLandmarkModeEnabled
        let currentOverlayMode = faceOverlayMode
        let imageForVLM: CIImage
        if isFaceMode {
            let history = displayFaceHistory
            let arrowsOn = currentOverlayMode == .arrows || currentOverlayMode == .arrowsOnFace

            var rendered: CIImage? = nil
            if currentOverlayMode == .arrowsOnFace, !history.isEmpty {
                rendered = FaceLandmarkOverlay.renderCroppedFace(
                    frame: frame,
                    snapshot: history.last!)
            }
            if rendered == nil {
                rendered = FaceLandmarkOverlay.renderToImage(
                    history: history, drawArrows: arrowsOn)
            }

            if let rendered {
                imageForVLM = rendered
                debugSaveImage(rendered, tag: "vlm_input_single")
                print("[VLM DEBUG single] rendered mode=\(currentOverlayMode), \(history.count) ghosts")
                let ciCtx = CIContext()
                if let cg = ciCtx.createCGImage(rendered, from: rendered.extent) {
                    lastVLMInputImage = cg
                }
            } else if let detection = faceLandmarker.detectObservation(in: frame) {
                if let r = FaceLandmarkOverlay.renderToImage(history: [detection]) {
                    imageForVLM = r
                } else {
                    imageForVLM = CIImage(cvPixelBuffer: frame)
                }
            } else {
                imageForVLM = CIImage(cvPixelBuffer: frame)
            }
        } else {
            imageForVLM = CIImage(cvPixelBuffer: frame)
        }

        let fullPrompt: String
        let singleBasePrompt: String
        let isAOF = isFaceMode && currentOverlayMode == .arrowsOnFace
        if isAOF && displayFaceHistory.count >= 2 {
            let movement = FaceMovementArrowsOverlay.describeMovements(
                old: displayFaceHistory.first!,
                new: displayFaceHistory.last!)
            singleBasePrompt = "Detected muscle movement: \(movement). \(arrowsOnFacePrompt)"
        } else {
            singleBasePrompt = prompt
        }
        let activeSuffix = isAOF ? arrowsOnFaceSuffix : promptSuffix
        if isFaceMode && !emotionHistory.isEmpty {
            let prev = emotionHistory.suffix(3).joined(separator: ", ")
            fullPrompt = "\(singleBasePrompt) Previous: \(prev). \(activeSuffix)"
        } else if isFaceMode {
            fullPrompt = "\(singleBasePrompt) \(activeSuffix)"
        } else {
            fullPrompt = "\(prompt) \(promptSuffix)"
        }

        let userInput = UserInput(
            prompt: .text(fullPrompt),
            images: [.ciImage(imageForVLM)]
        )

        Task {
            let t = await model.generate(userInput)
            _ = await t.result
            if isFaceMode {
                let raw = model.output.trimmingCharacters(in: .whitespacesAndNewlines)
                let emotion = Self.extractEmotion(from: raw)
                if !emotion.isEmpty {
                    emotionHistory.append(emotion)
                    if emotionHistory.count > 5 {
                        emotionHistory.removeFirst(emotionHistory.count - 5)
                    }
                }
            }
        }
    }

    // MARK: - Emotion extraction

    private static let knownEmotions: Set<String> = [
        "happy", "happiness", "sad", "sadness", "angry", "anger",
        "surprised", "surprise", "fearful", "fear", "disgusted", "disgust",
        "neutral", "contempt", "joy", "anxious", "confused", "bored",
        "excited", "calm", "smiling", "frowning", "worried"
    ]

    private static let junkPrefixes = [
        "the image depicts", "the image appears", "the image shows",
        "this image depicts", "this image appears", "this image shows",
        "the image is", "this is a", "i see a",
    ]

    /// Extracts a short emotion label from a VLM response.
    /// Always tries to find a known emotion keyword first and returns just that word.
    private static let arousalValenceKeywords: Set<String> = [
        "arousal", "valence", "low", "medium", "high",
        "negative", "neutral", "positive"
    ]

    static func extractEmotion(from raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: ".,;"))
        let lower = trimmed.lowercased()

        for prefix in junkPrefixes {
            if lower.hasPrefix(prefix) { return "" }
        }

        // Check for arousal/valence format
        if lower.contains("arousal") || lower.contains("valence") {
            let words = lower.split(separator: " ")
                .map { String($0).trimmingCharacters(in: .punctuationCharacters) }
                .filter { arousalValenceKeywords.contains($0) }
            if words.count >= 2 { return words.joined(separator: " ") }
        }

        for w in lower.split(separator: " ") {
            let clean = String(w).trimmingCharacters(in: .punctuationCharacters)
            if knownEmotions.contains(clean) {
                return clean
            }
        }

        let allWords = trimmed.split(separator: " ")
        if allWords.count <= 6 { return trimmed.lowercased() }

        return ""
    }
}

// MARK: - Debug helpers

/// Saves a CIImage to Desktop (macOS) or Documents (iOS) for inspection.
private func debugSaveImage(_ ciImage: CIImage, tag: String) {
    let ctx = CIContext()
    guard let cgImage = ctx.createCGImage(ciImage, from: ciImage.extent) else {
        print("[VLM DEBUG] failed to create CGImage")
        return
    }

    #if os(iOS)
    let uiImage = UIImage(cgImage: cgImage)
    guard let data = uiImage.pngData() else { return }
    let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        ?? FileManager.default.temporaryDirectory
    #else
    let nsImage = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    guard let tiff = nsImage.tiffRepresentation,
          let rep = NSBitmapImageRep(data: tiff),
          let data = rep.representation(using: .png, properties: [:]) else { return }
    let dir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop")
    #endif

    let url = dir.appendingPathComponent("\(tag).png")
    do {
        try data.write(to: url)
        print("[VLM DEBUG] saved → \(url.path) (\(cgImage.width)×\(cgImage.height))")
    } catch {
        print("[VLM DEBUG] save failed: \(error)")
    }
}

// MARK: - FaceLandmarkOverlay

/// Draws up to 5 face-mesh snapshots with a ghost-trail effect.
/// Newest snapshot is fully bright; older ones fade out progressively.
///
/// Coordinate mapping accounts for `resizeAspectFill` — the image may be
/// cropped when it doesn't match the 4:3 view ratio.
private struct FaceLandmarkOverlay: View {
    let history: [FaceLandmarkDisplayResult]
    var referenceFace: [CGPoint]? = nil

    var body: some View {
        Canvas { ctx, size in
            draw(ctx: ctx, size: size)
        }
        .allowsHitTesting(false)
    }

    // MARK: - Strain helpers

    static func edgeDistance(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        hypot(a.x - b.x, a.y - b.y)
    }

    /// Maps a distance ratio (current / reference) to an RGB color.
    /// ratio < 1 → red (compression), ratio > 1 → blue (stretch), ~1 → gray.
    static func strainColor(ratio: CGFloat) -> (r: CGFloat, g: CGFloat, b: CGFloat) {
        let clamped = min(max(ratio, 0.7), 1.3)
        let t = (clamped - 1.0) / 0.3  // -1…+1, 0 = no change
        if t < 0 {
            let intensity = -t  // 0…1
            return (r: 0.5 + 0.5 * intensity, g: 0.5 * (1 - intensity), b: 0.5 * (1 - intensity))
        } else {
            let intensity = t   // 0…1
            return (r: 0.5 * (1 - intensity), g: 0.5 * (1 - intensity), b: 0.5 + 0.5 * intensity)
        }
    }

    static func strainSwiftUIColor(ratio: CGFloat) -> Color {
        let c = strainColor(ratio: ratio)
        return Color(red: Double(c.r), green: Double(c.g), blue: Double(c.b))
    }

    // MARK: - Drawing

    private func draw(ctx: GraphicsContext, size: CGSize) {
        let count = history.count
        guard count > 0 else { return }

        let ref: [CGPoint]? = referenceFace ?? (count > 1 ? history.first?.landmarks.first : nil)

        for (idx, snapshot) in history.enumerated() {
            let age = CGFloat(idx + 1) / CGFloat(count)
            let alpha = age * age
            let isNewest = idx == count - 1
            drawSnapshot(ctx: ctx, size: size, result: snapshot, alpha: alpha,
                         referenceFace: isNewest ? ref : nil)
        }
    }

    private func drawSnapshot(
        ctx: GraphicsContext, size: CGSize,
        result: FaceLandmarkDisplayResult, alpha: CGFloat,
        referenceFace: [CGPoint]? = nil
    ) {
        let imgW = result.imageSize.width
        let imgH = result.imageSize.height
        guard imgW > 0, imgH > 0 else { return }

        let scaleX = size.width  / imgW
        let scaleY = size.height / imgH
        let scale  = max(scaleX, scaleY)

        let scaledW = imgW * scale
        let scaledH = imgH * scale
        let offX = (scaledW - size.width)  / 2
        let offY = (scaledH - size.height) / 2

        for face in result.landmarks {
            guard face.count >= 468 else { continue }

            func toView(_ p: CGPoint) -> CGPoint {
                CGPoint(x: p.x * scaledW - offX,
                        y: p.y * scaledH - offY)
            }

            drawConnections(ctx: ctx, face: face, toView: toView, alpha: alpha,
                            referenceFace: referenceFace)

            let dotRadius: CGFloat = alpha < 1 ? 1.0 : 1.5
            for i in 0..<min(face.count, 468) {
                let vp = toView(face[i])
                ctx.fill(
                    Path(ellipseIn: CGRect(
                        x: vp.x - dotRadius, y: vp.y - dotRadius,
                        width: dotRadius * 2, height: dotRadius * 2)),
                    with: .color(.blue.opacity(0.6 * alpha)))
            }

            if face.count >= 478 {
                let r: CGFloat = alpha < 1 ? 2.5 : 4
                for i in 468..<478 {
                    let vp = toView(face[i])
                    let rect = CGRect(x: vp.x - r, y: vp.y - r,
                                      width: 2 * r, height: 2 * r)
                    ctx.fill(Path(ellipseIn: rect),
                             with: .color(.black.opacity(0.8 * alpha)))
                    ctx.stroke(Path(ellipseIn: rect),
                               with: .color(.blue.opacity(alpha)),
                               lineWidth: 1.5)
                }
            }
        }
    }

    // MARK: - Mesh connections

    private func drawConnections(
        ctx: GraphicsContext,
        face: [CGPoint],
        toView: (CGPoint) -> CGPoint,
        alpha: CGFloat,
        referenceFace: [CGPoint]? = nil
    ) {
        let lw: CGFloat = alpha < 1 ? 0.8 : 1.0

        func strokePath(
            _ indices: [Int], color: Color,
            lineWidth: CGFloat = 1.5, closed: Bool = false
        ) {
            guard indices.count >= 2,
                  indices.allSatisfy({ $0 < face.count }) else { return }

            let useStrain = referenceFace != nil
                && indices.allSatisfy({ $0 < (referenceFace?.count ?? 0) })

            if useStrain, let ref = referenceFace {
                let allIdx = closed ? indices + [indices[0]] : indices
                for seg in 0..<(allIdx.count - 1) {
                    let iA = allIdx[seg], iB = allIdx[seg + 1]
                    let curDist = Self.edgeDistance(face[iA], face[iB])
                    let refDist = Self.edgeDistance(ref[iA], ref[iB])
                    let ratio = refDist > 1e-8 ? curDist / refDist : 1.0
                    let sc = Self.strainSwiftUIColor(ratio: ratio)
                    var path = Path()
                    path.move(to: toView(face[iA]))
                    path.addLine(to: toView(face[iB]))
                    ctx.stroke(path, with: .color(sc.opacity(0.9 * Double(alpha))),
                               lineWidth: lineWidth * lw)
                }
            } else {
                var path = Path()
                path.move(to: toView(face[indices[0]]))
                for i in indices.dropFirst() {
                    path.addLine(to: toView(face[i]))
                }
                if closed { path.closeSubpath() }
                ctx.stroke(path, with: .color(color.opacity(0.8 * alpha)),
                           lineWidth: lineWidth * lw)
            }
        }

        strokePath(Self.faceOval,          color: .gray,   closed: true)
        strokePath(Self.leftEye,           color: .blue,   closed: true)
        strokePath(Self.rightEye,          color: .blue,   closed: true)
        strokePath(Self.leftEyebrowUpper,  color: .brown)
        strokePath(Self.leftEyebrowLower,  color: .brown)
        strokePath(Self.rightEyebrowUpper, color: .brown)
        strokePath(Self.rightEyebrowLower, color: .brown)
        strokePath(Self.lipsOuter,         color: .red,    closed: true)
        strokePath(Self.lipsInner,         color: .red,    lineWidth: 1, closed: true)
        strokePath(Self.noseBridge,        color: .gray)
        strokePath(Self.noseBottom,        color: .gray)

        if face.count >= 478 {
            strokePath(Self.leftIris,  color: .black, lineWidth: 1.5, closed: true)
            strokePath(Self.rightIris, color: .black, lineWidth: 1.5, closed: true)
        }
    }

    // MARK: - Render to CIImage (for VLM input)

    /// Rasterises the landmark ghost-trail into a 512×512 CIImage on a black
    /// background — the same visualisation the user sees on screen.
    static func renderToImage(
        history: [FaceLandmarkDisplayResult],
        referenceFace: [CGPoint]? = nil,
        drawArrows: Bool = false
    ) -> CIImage? {
        let count = history.count
        guard count > 0 else { return nil }

        let side = 512
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: side, height: side,
            bitsPerComponent: 8, bytesPerRow: side * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        ctx.translateBy(x: 0, y: CGFloat(side))
        ctx.scaleBy(x: 1, y: -1)

        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: side, height: side))

        let ref = referenceFace ?? (count > 1 ? history.first?.landmarks.first : nil)

        let s = CGFloat(side)
        for (idx, snapshot) in history.enumerated() {
            let age = CGFloat(idx + 1) / CGFloat(count)
            let alpha = age * age
            let isNewest = idx == count - 1

            for face in snapshot.landmarks {
                guard face.count >= 468 else { continue }

                func toPixel(_ p: CGPoint) -> CGPoint {
                    CGPoint(x: p.x * s, y: p.y * s)
                }

                drawCGConnections(ctx: ctx, face: face,
                                  toPixel: toPixel, alpha: alpha,
                                  referenceFace: isNewest ? ref : nil)

                let dotR: CGFloat = alpha < 1 ? 1.5 : 2.5
                ctx.setFillColor(red: 0, green: 0, blue: 0.8, alpha: 0.6 * alpha)
                for i in 0..<min(face.count, 468) {
                    let p = toPixel(face[i])
                    ctx.fillEllipse(in: CGRect(x: p.x - dotR, y: p.y - dotR,
                                               width: dotR * 2, height: dotR * 2))
                }

                if face.count >= 478 {
                    let r: CGFloat = alpha < 1 ? 3 : 5
                    for i in 468..<478 {
                        let p = toPixel(face[i])
                        let rect = CGRect(x: p.x - r, y: p.y - r,
                                          width: 2 * r, height: 2 * r)
                        ctx.setFillColor(red: 0, green: 0, blue: 0,
                                         alpha: 0.8 * alpha)
                        ctx.fillEllipse(in: rect)
                        ctx.setStrokeColor(red: 0, green: 0, blue: 0.8,
                                           alpha: alpha)
                        ctx.setLineWidth(1.5)
                        ctx.strokeEllipse(in: rect)
                    }
                }
            }
        }

        if drawArrows, count >= 2,
           let oldFace = history.first?.landmarks.first,
           let newFace = history.last?.landmarks.first {
            FaceMovementArrowsOverlay.drawArrowsCG(
                ctx: ctx, oldFace: oldFace, newFace: newFace, size: s)
        }

        guard let cgImage = ctx.makeImage() else { return nil }
        return CIImage(cgImage: cgImage)
    }

    /// Crops the face from the camera frame and scales to 512×512.
    static func renderCroppedFace(
        frame: CVPixelBuffer,
        snapshot: FaceLandmarkDisplayResult
    ) -> CIImage? {
        guard let face = snapshot.landmarks.first,
              face.count >= 468 else { return nil }

        let xs = face.map(\.x)
        let ys = face.map(\.y)
        guard let minX = xs.min(), let maxX = xs.max(),
              let minY = ys.min(), let maxY = ys.max() else { return nil }

        let faceW = maxX - minX
        let faceH = maxY - minY
        let pad: CGFloat = 0.25
        let cropNorm = CGRect(
            x: max(0, minX - faceW * pad),
            y: max(0, minY - faceH * pad),
            width: min(1.0, faceW * (1 + 2 * pad)),
            height: min(1.0, faceH * (1 + 2 * pad)))

        let imgW = CGFloat(CVPixelBufferGetWidth(frame))
        let imgH = CGFloat(CVPixelBufferGetHeight(frame))
        let pixelRect = CGRect(
            x: cropNorm.minX * imgW,
            y: (1.0 - cropNorm.maxY) * imgH,
            width: cropNorm.width * imgW,
            height: cropNorm.height * imgH)
            .intersection(CGRect(x: 0, y: 0, width: imgW, height: imgH))

        guard pixelRect.width > 1, pixelRect.height > 1 else { return nil }

        let s: CGFloat = 512
        var ciImage = CIImage(cvPixelBuffer: frame)
            .cropped(to: pixelRect)
            .transformed(by: CGAffineTransform(
                translationX: -pixelRect.minX, y: -pixelRect.minY))
        let scaleX = s / ciImage.extent.width
        let scaleY = s / ciImage.extent.height
        ciImage = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        return ciImage
    }

    private static func drawCGConnections(
        ctx: CGContext, face: [CGPoint],
        toPixel: (CGPoint) -> CGPoint, alpha: CGFloat,
        referenceFace: [CGPoint]? = nil
    ) {
        let lw: CGFloat = alpha < 1 ? 0.8 : 1.0

        func strokePath(
            _ indices: [Int],
            r: CGFloat, g: CGFloat, b: CGFloat,
            lineWidth: CGFloat = 1.5, closed: Bool = false
        ) {
            guard indices.count >= 2,
                  indices.allSatisfy({ $0 < face.count }) else { return }

            let useStrain = referenceFace != nil
                && indices.allSatisfy({ $0 < (referenceFace?.count ?? 0) })

            if useStrain, let ref = referenceFace {
                let allIdx = closed ? indices + [indices[0]] : indices
                for seg in 0..<(allIdx.count - 1) {
                    let iA = allIdx[seg], iB = allIdx[seg + 1]
                    let curDist = edgeDistance(face[iA], face[iB])
                    let refDist = edgeDistance(ref[iA], ref[iB])
                    let ratio = refDist > 1e-8 ? curDist / refDist : 1.0
                    let sc = strainColor(ratio: ratio)
                    ctx.setStrokeColor(red: sc.r, green: sc.g, blue: sc.b,
                                       alpha: 0.9 * alpha)
                    ctx.setLineWidth(lineWidth * lw)
                    ctx.beginPath()
                    ctx.move(to: toPixel(face[iA]))
                    ctx.addLine(to: toPixel(face[iB]))
                    ctx.strokePath()
                }
            } else {
                ctx.setStrokeColor(red: r, green: g, blue: b,
                                   alpha: 0.8 * alpha)
                ctx.setLineWidth(lineWidth * lw)
                ctx.beginPath()
                ctx.move(to: toPixel(face[indices[0]]))
                for i in indices.dropFirst() {
                    ctx.addLine(to: toPixel(face[i]))
                }
                if closed { ctx.closePath() }
                ctx.strokePath()
            }
        }

        strokePath(faceOval,          r: 0.5,  g: 0.5,  b: 0.5,  closed: true)
        strokePath(leftEye,           r: 0,    g: 0,    b: 0.8,  closed: true)
        strokePath(rightEye,          r: 0,    g: 0,    b: 0.8,  closed: true)
        strokePath(leftEyebrowUpper,  r: 0.45, g: 0.3,  b: 0.15)
        strokePath(leftEyebrowLower,  r: 0.45, g: 0.3,  b: 0.15)
        strokePath(rightEyebrowUpper, r: 0.45, g: 0.3,  b: 0.15)
        strokePath(rightEyebrowLower, r: 0.45, g: 0.3,  b: 0.15)
        strokePath(lipsOuter,         r: 0.8,  g: 0,    b: 0,    closed: true)
        strokePath(lipsInner,         r: 0.8,  g: 0,    b: 0,    lineWidth: 1, closed: true)
        strokePath(noseBridge,        r: 0.5,  g: 0.5,  b: 0.5)
        strokePath(noseBottom,        r: 0.5,  g: 0.5,  b: 0.5)

        if face.count >= 478 {
            strokePath(leftIris,  r: 0, g: 0, b: 0, lineWidth: 1.5, closed: true)
            strokePath(rightIris, r: 0, g: 0, b: 0, lineWidth: 1.5, closed: true)
        }
    }

    // MARK: - Standard MediaPipe face mesh landmark indices

    static let faceOval = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]

    static let leftEye = [
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        173, 157, 158, 159, 160, 161, 246
    ]
    static let rightEye = [
        362, 382, 381, 380, 374, 373, 390, 249, 263,
        466, 388, 387, 386, 385, 384, 398
    ]

    static let leftEyebrowUpper  = [46, 53, 52, 65, 55]
    static let leftEyebrowLower  = [107, 66, 105, 63, 70]
    static let rightEyebrowUpper = [276, 283, 282, 295, 285]
    static let rightEyebrowLower = [336, 296, 334, 293, 300]

    static let lipsOuter = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 409, 270, 269, 267, 0, 37, 39, 40, 185
    ]
    static let lipsInner = [
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        308, 415, 310, 311, 312, 13, 82, 81, 80, 191
    ]

    static let noseBridge = [168, 6, 197, 195, 5]
    static let noseBottom = [48, 115, 220, 45, 4, 275, 440, 344, 278]

    static let leftIris  = [468, 469, 470, 471, 472]
    static let rightIris = [473, 474, 475, 476, 477]
}

// MARK: - FaceMovementArrowsOverlay

/// Draws arrows showing local feature movement between the oldest and newest
/// ghost snapshots.  A similarity transform (translation + rotation + uniform
/// scale) derived from the eye centres is used to map old landmarks into the
/// new head pose, so only genuine feature displacement (brow raise, lip curl,
/// cheek puff, etc.) produces arrows.  Arrows are suppressed entirely when
/// head rotation or scale change exceeds safe thresholds.
private struct FaceMovementArrowsOverlay: View {
    let oldSnapshot: FaceLandmarkDisplayResult
    let newSnapshot: FaceLandmarkDisplayResult

    var body: some View {
        Canvas { ctx, size in
            draw(ctx: ctx, size: size)
        }
        .allowsHitTesting(false)
    }

    // MARK: - Geometry helpers

    private static func centroid(of indices: [Int], in face: [CGPoint]) -> CGPoint? {
        let valid = indices.filter { $0 < face.count }
        guard !valid.isEmpty else { return nil }
        let sum = valid.reduce(CGPoint.zero) {
            CGPoint(x: $0.x + face[$1].x, y: $0.y + face[$1].y)
        }
        return CGPoint(x: sum.x / CGFloat(valid.count),
                       y: sum.y / CGFloat(valid.count))
    }

    private static func eyeCenters(_ face: [CGPoint]) -> (left: CGPoint, right: CGPoint)? {
        guard let l = centroid(of: [33, 133], in: face),
              let r = centroid(of: [362, 263], in: face) else { return nil }
        return (l, r)
    }

    /// Builds a similarity transform that maps old eye centres → new eye
    /// centres.  Returns nil if head rotation or scale change is too large
    /// for reliable local-feature comparison.
    private static func headTransform(
        oldFace: [CGPoint], newFace: [CGPoint]
    ) -> CGAffineTransform? {
        guard let oldEyes = eyeCenters(oldFace),
              let newEyes = eyeCenters(newFace) else { return nil }

        let oldMid = CGPoint(x: (oldEyes.left.x + oldEyes.right.x) / 2,
                             y: (oldEyes.left.y + oldEyes.right.y) / 2)
        let newMid = CGPoint(x: (newEyes.left.x + newEyes.right.x) / 2,
                             y: (newEyes.left.y + newEyes.right.y) / 2)

        let oldDx = oldEyes.right.x - oldEyes.left.x
        let oldDy = oldEyes.right.y - oldEyes.left.y
        let newDx = newEyes.right.x - newEyes.left.x
        let newDy = newEyes.right.y - newEyes.left.y

        let oldDist = sqrt(oldDx * oldDx + oldDy * oldDy)
        let newDist = sqrt(newDx * newDx + newDy * newDy)
        guard oldDist > 1e-6, newDist > 1e-6 else { return nil }

        let oldAngle = atan2(oldDy, oldDx)
        let newAngle = atan2(newDy, newDx)
        let dAngle = newAngle - oldAngle
        let dScale = newDist / oldDist

        // Suppress arrows for large head movements
        if abs(dAngle) > 0.18 { return nil }       // >~10°
        if dScale < 0.75 || dScale > 1.33 { return nil }  // >25% scale change

        // Similarity transform: translate to origin → rotate → scale → translate
        let cosA = cos(dAngle) * dScale
        let sinA = sin(dAngle) * dScale

        // M maps old normalised-coords to new normalised-coords:
        //   p' = R*s*(p - oldMid) + newMid
        let tx = newMid.x - cosA * oldMid.x + sinA * oldMid.y
        let ty = newMid.y - sinA * oldMid.x - cosA * oldMid.y

        return CGAffineTransform(a: cosA, b: sinA, c: -sinA, d: cosA, tx: tx, ty: ty)
    }

    /// Feature regions: (label, indices, color)
    private static let featureGroups: [(String, [Int], Color)] = [
        // Eyebrows
        ("L.Brow",     [46, 53, 52, 65, 55, 107, 66, 105, 63, 70],              .brown),
        ("R.Brow",     [276, 283, 282, 295, 285, 336, 296, 334, 293, 300],       .brown),
        // Glabella (between eyebrows) — frown/furrow indicator
        ("Glabella",   [9, 151, 108, 69, 104, 68, 71],                           .purple),
        // Cheeks
        ("L.Cheek",    [116, 117, 118, 119, 100, 126, 142, 36, 205],             .pink),
        ("R.Cheek",    [345, 346, 347, 348, 329, 355, 371, 266, 425],            .pink),
        // Lips
        ("Upper Lip",  [13, 82, 81, 80, 191, 78, 312, 311, 310, 415, 308],      .red),
        ("Lower Lip",  [14, 87, 178, 88, 95, 317, 402, 318, 324],               .red),
        ("Lip L",      [61, 146, 91],                                             .orange),
        ("Lip R",      [291, 375, 321],                                           .orange),
        // Chin
        ("Chin",       [152, 377, 400, 378, 379, 365, 397, 288, 361, 150, 149, 176, 148], .indigo),
        // Jaw sides
        ("L.Jaw",      [172, 58, 132, 93, 234, 127],                             .teal),
        ("R.Jaw",      [401, 288, 361, 323, 454, 356],                           .teal),
    ]

    // MARK: - CGContext arrow rendering (for VLM input image)

    private static let cgFeatureColors: [(String, [Int], (r: CGFloat, g: CGFloat, b: CGFloat))] = [
        ("L.Brow",     [46, 53, 52, 65, 55, 107, 66, 105, 63, 70],              (0.45, 0.3, 0.15)),
        ("R.Brow",     [276, 283, 282, 295, 285, 336, 296, 334, 293, 300],       (0.45, 0.3, 0.15)),
        ("Glabella",   [9, 151, 108, 69, 104, 68, 71],                           (0.5, 0.0, 0.5)),
        ("L.Cheek",    [116, 117, 118, 119, 100, 126, 142, 36, 205],             (0.9, 0.4, 0.5)),
        ("R.Cheek",    [345, 346, 347, 348, 329, 355, 371, 266, 425],            (0.9, 0.4, 0.5)),
        ("Upper Lip",  [13, 82, 81, 80, 191, 78, 312, 311, 310, 415, 308],      (0.8, 0.0, 0.0)),
        ("Lower Lip",  [14, 87, 178, 88, 95, 317, 402, 318, 324],               (0.8, 0.0, 0.0)),
        ("Lip L",      [61, 146, 91],                                             (0.9, 0.5, 0.0)),
        ("Lip R",      [291, 375, 321],                                           (0.9, 0.5, 0.0)),
        ("Chin",       [152, 377, 400, 378, 379, 365, 397, 288, 361, 150, 149, 176, 148], (0.3, 0.0, 0.5)),
        ("L.Jaw",      [172, 58, 132, 93, 234, 127],                             (0.0, 0.5, 0.5)),
        ("R.Jaw",      [401, 288, 361, 323, 454, 356],                           (0.0, 0.5, 0.5)),
    ]

    /// Draws movement arrows into a CGContext (for VLM input image rendering).
    static func drawArrowsCG(
        ctx: CGContext,
        oldFace: [CGPoint],
        newFace: [CGPoint],
        size: CGFloat
    ) {
        guard oldFace.count >= 468, newFace.count >= 468 else { return }
        guard let xform = headTransform(oldFace: oldFace, newFace: newFace) else { return }
        guard let newEyes = eyeCenters(newFace) else { return }
        let eyeDist = hypot(newEyes.right.x - newEyes.left.x,
                            newEyes.right.y - newEyes.left.y)
        guard eyeDist > 1e-6 else { return }

        func toPixel(_ p: CGPoint) -> CGPoint {
            CGPoint(x: p.x * size, y: p.y * size)
        }

        let arrowR: CGFloat = 0.0, arrowG: CGFloat = 0.9, arrowB: CGFloat = 0.2

        for (_, indices, _) in cgFeatureColors {
            guard let oldCtr = centroid(of: indices, in: oldFace),
                  let newCtr = centroid(of: indices, in: newFace) else { continue }
            let expected = oldCtr.applying(xform)
            let dx = newCtr.x - expected.x
            let dy = newCtr.y - expected.y
            let normMag = hypot(dx, dy) / eyeDist
            guard normMag > 0.02 else { continue }

            let arrowScale: CGFloat = size * 3.0
            let origin = toPixel(newCtr)
            let tip = CGPoint(x: origin.x + dx * arrowScale,
                              y: origin.y + dy * arrowScale)

            let shaftLen = hypot(tip.x - origin.x, tip.y - origin.y)
            guard shaftLen > 3 else { continue }

            ctx.setStrokeColor(red: arrowR, green: arrowG, blue: arrowB, alpha: 1.0)
            ctx.setLineWidth(2.5)
            ctx.beginPath()
            ctx.move(to: origin)
            ctx.addLine(to: tip)
            ctx.strokePath()

            let headLen: CGFloat = min(12, shaftLen * 0.35)
            let angle = atan2(tip.y - origin.y, tip.x - origin.x)
            let spread: CGFloat = .pi / 6
            ctx.setFillColor(red: arrowR, green: arrowG, blue: arrowB, alpha: 1.0)
            ctx.beginPath()
            ctx.move(to: tip)
            ctx.addLine(to: CGPoint(x: tip.x - headLen * cos(angle - spread),
                                    y: tip.y - headLen * sin(angle - spread)))
            ctx.addLine(to: CGPoint(x: tip.x - headLen * cos(angle + spread),
                                    y: tip.y - headLen * sin(angle + spread)))
            ctx.closePath()
            ctx.fillPath()
        }
    }

    // MARK: - Movement detection (used to gate VLM inference)

    /// Returns true if at least one feature shows meaningful local movement
    /// between two snapshots (i.e. arrows would be drawn).
    static func hasMeaningfulMovement(
        old: FaceLandmarkDisplayResult,
        new: FaceLandmarkDisplayResult
    ) -> Bool {
        guard let oldFace = old.landmarks.first,
              let newFace = new.landmarks.first,
              oldFace.count >= 468, newFace.count >= 468 else { return false }

        guard let xform = headTransform(oldFace: oldFace, newFace: newFace) else {
            return false
        }
        guard let newEyes = eyeCenters(newFace) else { return false }
        let eyeDist = hypot(newEyes.right.x - newEyes.left.x,
                            newEyes.right.y - newEyes.left.y)
        guard eyeDist > 1e-6 else { return false }

        for (_, indices, _) in featureGroups {
            guard let oldCtr = centroid(of: indices, in: oldFace),
                  let newCtr = centroid(of: indices, in: newFace) else { continue }
            let expected = oldCtr.applying(xform)
            let dx = newCtr.x - expected.x
            let dy = newCtr.y - expected.y
            if hypot(dx, dy) / eyeDist > 0.02 { return true }
        }
        return false
    }

    /// Produces a human-readable description of facial movements
    /// (e.g. "lip corners up (smile), eyebrows raised, mouth opening").
    static func describeMovements(
        old: FaceLandmarkDisplayResult,
        new: FaceLandmarkDisplayResult
    ) -> String {
        guard let oldFace = old.landmarks.first,
              let newFace = new.landmarks.first,
              oldFace.count >= 468, newFace.count >= 468 else { return "" }

        guard let xform = headTransform(oldFace: oldFace, newFace: newFace) else {
            return "large head movement"
        }
        guard let newEyes = eyeCenters(newFace) else { return "" }
        let eyeDist = hypot(newEyes.right.x - newEyes.left.x,
                            newEyes.right.y - newEyes.left.y)
        guard eyeDist > 1e-6 else { return "" }

        func residual(for indices: [Int]) -> (dx: CGFloat, dy: CGFloat, mag: CGFloat)? {
            guard let oldCtr = centroid(of: indices, in: oldFace),
                  let newCtr = centroid(of: indices, in: newFace) else { return nil }
            let expected = oldCtr.applying(xform)
            let dx = newCtr.x - expected.x
            let dy = newCtr.y - expected.y
            let mag = hypot(dx, dy) / eyeDist
            return mag > 0.02 ? (dx, dy, mag) : nil
        }

        var parts: [String] = []

        // Eyebrows
        let lBrow = [46, 53, 52, 65, 55, 107, 66, 105, 63, 70]
        let rBrow = [276, 283, 282, 295, 285, 336, 296, 334, 293, 300]
        if let lb = residual(for: lBrow), let rb = residual(for: rBrow) {
            let avgDy = (lb.dy + rb.dy) / 2
            if avgDy < -0.01 {
                parts.append("eyebrows raised")
            } else if avgDy > 0.01 {
                parts.append("eyebrows lowered")
            }
        }

        // Glabella (between eyebrows)
        let glabella = [9, 151, 108, 69, 104, 68, 71]
        if let g = residual(for: glabella) {
            if g.dy > 0.01 {
                parts.append("brow furrowing")
            }
        }

        // Lip corners
        let lipL = [61, 146, 91]
        let lipR = [291, 375, 321]
        if let ll = residual(for: lipL), let lr = residual(for: lipR) {
            let avgDy = (ll.dy + lr.dy) / 2
            if avgDy < -0.01 {
                parts.append("lip corners up (smile)")
            } else if avgDy > 0.01 {
                parts.append("lip corners down (frown)")
            }
        }

        // Upper/lower lip (mouth open/close)
        let upperLip = [13, 82, 81, 80, 191, 78, 312, 311, 310, 415, 308]
        let lowerLip = [14, 87, 178, 88, 95, 317, 402, 318, 324]
        if let ul = residual(for: upperLip), let ll = residual(for: lowerLip) {
            let spread = ll.dy - ul.dy
            if spread > 0.02 {
                parts.append("mouth opening")
            } else if spread < -0.02 {
                parts.append("lips pressing together")
            }
        }

        // Cheeks
        let lCheek = [116, 117, 118, 119, 100, 126, 142, 36, 205]
        let rCheek = [345, 346, 347, 348, 329, 355, 371, 266, 425]
        if let lc = residual(for: lCheek), let rc = residual(for: rCheek) {
            let avgDy = (lc.dy + rc.dy) / 2
            if avgDy < -0.01 {
                parts.append("cheeks raised")
            }
        }

        // Chin
        let chin = [152, 377, 400, 378, 379, 365, 397, 288, 361, 150, 149, 176, 148]
        if let c = residual(for: chin) {
            if c.dy > 0.02 {
                parts.append("jaw dropping")
            }
        }

        return parts.isEmpty ? "no significant movement" : parts.joined(separator: ", ")
    }

    // MARK: - Drawing

    private func draw(ctx: GraphicsContext, size: CGSize) {
        guard let oldFace = oldSnapshot.landmarks.first,
              let newFace = newSnapshot.landmarks.first,
              oldFace.count >= 468, newFace.count >= 468 else { return }

        let imgW = newSnapshot.imageSize.width
        let imgH = newSnapshot.imageSize.height
        guard imgW > 0, imgH > 0 else { return }

        guard let xform = Self.headTransform(oldFace: oldFace, newFace: newFace) else {
            return  // head moved too much — suppress arrows
        }

        // Inter-eye distance in the new frame for thresholding
        guard let newEyes = Self.eyeCenters(newFace) else { return }
        let eyeDist = hypot(newEyes.right.x - newEyes.left.x,
                            newEyes.right.y - newEyes.left.y)
        guard eyeDist > 1e-6 else { return }

        // resizeAspectFill mapping
        let scaleX = size.width / imgW
        let scaleY = size.height / imgH
        let viewScale = max(scaleX, scaleY)
        let scaledW = imgW * viewScale
        let scaledH = imgH * viewScale
        let offX = (scaledW - size.width) / 2
        let offY = (scaledH - size.height) / 2

        func toView(_ p: CGPoint) -> CGPoint {
            CGPoint(x: p.x * scaledW - offX, y: p.y * scaledH - offY)
        }

        let arrowColor = Color(red: 0, green: 0.9, blue: 0.2)

        for (_, indices, _) in Self.featureGroups {
            guard let oldCtr = Self.centroid(of: indices, in: oldFace),
                  let newCtr = Self.centroid(of: indices, in: newFace) else { continue }

            let expected = oldCtr.applying(xform)
            let dx = newCtr.x - expected.x
            let dy = newCtr.y - expected.y

            let normMag = hypot(dx, dy) / eyeDist
            guard normMag > 0.02 else { continue }

            let arrowScale: CGFloat = min(size.width, size.height) * 3.0
            let origin = toView(newCtr)
            let tip = CGPoint(x: origin.x + dx * arrowScale,
                              y: origin.y + dy * arrowScale)

            drawArrow(ctx: ctx, from: origin, to: tip,
                      color: arrowColor, lineWidth: 2.5)
        }
    }

    private func drawArrow(
        ctx: GraphicsContext, from: CGPoint, to: CGPoint,
        color: Color, lineWidth: CGFloat
    ) {
        let dx = to.x - from.x
        let dy = to.y - from.y
        let len = hypot(dx, dy)
        guard len > 3 else { return }

        var shaft = Path()
        shaft.move(to: from)
        shaft.addLine(to: to)
        ctx.stroke(shaft, with: .color(color), lineWidth: lineWidth)

        let headLen: CGFloat = min(12, len * 0.35)
        let angle = atan2(dy, dx)
        let spread: CGFloat = .pi / 6
        var head = Path()
        head.move(to: to)
        head.addLine(to: CGPoint(x: to.x - headLen * cos(angle - spread),
                                 y: to.y - headLen * sin(angle - spread)))
        head.addLine(to: CGPoint(x: to.x - headLen * cos(angle + spread),
                                 y: to.y - headLen * sin(angle + spread)))
        head.closeSubpath()
        ctx.fill(head, with: .color(color))
    }
}

#Preview {
    ContentView()
}
