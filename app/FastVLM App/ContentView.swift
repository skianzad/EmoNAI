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
                                prompt = "Name the exact emotion on this face and any head tilt or turn."
                                promptSuffix = "Format: <emotion>, <movement>. Example: happy, tilting left. Do not describe the image."
                                model.maxTokens = 25
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
                                    if faceLandmarkModeEnabled {
                                        Color.white
                                    }
                                }
                                .overlay {
                                    if faceLandmarkModeEnabled,
                                       !displayFaceHistory.isEmpty {
                                        FaceLandmarkOverlay(
                                            history: displayFaceHistory)
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
                let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedPrompt.isEmpty {
                    Text(trimmedPrompt)
                        .foregroundStyle(.secondary)
                }

                let trimmedSuffix = promptSuffix.trimmingCharacters(in: .whitespacesAndNewlines)
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
            Section("Prompt") {
                TextEditor(text: $prompt)
                    .frame(minHeight: 38)
            }

            Section("Prompt Suffix") {
                TextEditor(text: $promptSuffix)
                    .frame(minHeight: 38)
            }
            #elseif os(macOS)
            Section {
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

                guard let rendered = FaceLandmarkOverlay.renderToImage(
                    history: vlmLandmarkHistory
                ) else {
                    print("[VLM DEBUG] renderToImage failed")
                    continue
                }
                imageForVLM = rendered
                print("[VLM DEBUG] rendered \(vlmLandmarkHistory.count) ghosts, extent: \(rendered.extent)")
                debugSaveImage(rendered, tag: "vlm_input")
            } else {
                vlmLandmarkHistory.removeAll()
                imageForVLM = CIImage(cvPixelBuffer: frame)
            }

            let fullPrompt: String
            if isFaceMode {
                let emotions = await MainActor.run { emotionHistory }
                if !emotions.isEmpty {
                    let prev = emotions.joined(separator: " → ")
                    fullPrompt = "\(prompt) Previous readings were: [\(prev)]. State the current emotion, movement, and whether it differs from previous. \(promptSuffix)"
                } else {
                    fullPrompt = "\(prompt) \(promptSuffix)"
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
                let emotion = await MainActor.run {
                    model.output.trimmingCharacters(in: .whitespacesAndNewlines)
                }
                print("[VLM DEBUG] response: \(emotion)")
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
        let imageForVLM: CIImage
        if isFaceMode {
            // Use the display history which is already populated by detectDisplayLandmarks
            let history = displayFaceHistory
            if let rendered = FaceLandmarkOverlay.renderToImage(history: history) {
                imageForVLM = rendered
                debugSaveImage(rendered, tag: "vlm_input_single")
                print("[VLM DEBUG single] rendered \(history.count) ghosts")
            } else if let detection = faceLandmarker.detectObservation(in: frame) {
                // Fallback: render just the current detection
                if let rendered = FaceLandmarkOverlay.renderToImage(history: [detection]) {
                    imageForVLM = rendered
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
        if isFaceMode && !emotionHistory.isEmpty {
            let prev = emotionHistory.joined(separator: " → ")
            fullPrompt = "\(prompt) Previous readings were: [\(prev)]. State the current emotion, movement, and whether it differs from previous. \(promptSuffix)"
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
                let emotion = model.output.trimmingCharacters(in: .whitespacesAndNewlines)
                if !emotion.isEmpty {
                    emotionHistory.append(emotion)
                    if emotionHistory.count > 5 {
                        emotionHistory.removeFirst(emotionHistory.count - 5)
                    }
                }
            }
        }
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

    var body: some View {
        Canvas { ctx, size in
            draw(ctx: ctx, size: size)
        }
        .allowsHitTesting(false)
    }

    // MARK: - Drawing

    private func draw(ctx: GraphicsContext, size: CGSize) {
        let count = history.count
        guard count > 0 else { return }

        for (idx, snapshot) in history.enumerated() {
            let age = CGFloat(idx + 1) / CGFloat(count)  // 0…1, newest = 1
            let alpha = age * age  // quadratic fade: [0.04, 0.16, 0.36, 0.64, 1.0] for 5 items
            drawSnapshot(ctx: ctx, size: size, result: snapshot, alpha: alpha)
        }
    }

    private func drawSnapshot(
        ctx: GraphicsContext, size: CGSize,
        result: FaceLandmarkDisplayResult, alpha: CGFloat
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

            drawConnections(ctx: ctx, face: face, toView: toView, alpha: alpha)

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
        alpha: CGFloat
    ) {
        func strokePath(
            _ indices: [Int], color: Color,
            lineWidth: CGFloat = 1.5, closed: Bool = false
        ) {
            guard indices.count >= 2,
                  indices.allSatisfy({ $0 < face.count }) else { return }
            var path = Path()
            path.move(to: toView(face[indices[0]]))
            for i in indices.dropFirst() {
                path.addLine(to: toView(face[i]))
            }
            if closed { path.closeSubpath() }
            ctx.stroke(path, with: .color(color.opacity(0.8 * alpha)),
                       lineWidth: lineWidth * (alpha < 1 ? 0.8 : 1.0))
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
    static func renderToImage(history: [FaceLandmarkDisplayResult]) -> CIImage? {
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

        // Flip so (0,0) is top-left, matching MediaPipe normalised coords.
        ctx.translateBy(x: 0, y: CGFloat(side))
        ctx.scaleBy(x: 1, y: -1)

        // White background
        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: side, height: side))

        let s = CGFloat(side)
        for (idx, snapshot) in history.enumerated() {
            let age = CGFloat(idx + 1) / CGFloat(count)
            let alpha = age * age

            for face in snapshot.landmarks {
                guard face.count >= 468 else { continue }

                func toPixel(_ p: CGPoint) -> CGPoint {
                    CGPoint(x: p.x * s, y: p.y * s)
                }

                drawCGConnections(ctx: ctx, face: face,
                                  toPixel: toPixel, alpha: alpha)

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

        guard let cgImage = ctx.makeImage() else { return nil }
        return CIImage(cgImage: cgImage)
    }

    private static func drawCGConnections(
        ctx: CGContext, face: [CGPoint],
        toPixel: (CGPoint) -> CGPoint, alpha: CGFloat
    ) {
        func strokePath(
            _ indices: [Int],
            r: CGFloat, g: CGFloat, b: CGFloat,
            lineWidth: CGFloat = 1.5, closed: Bool = false
        ) {
            guard indices.count >= 2,
                  indices.allSatisfy({ $0 < face.count }) else { return }
            ctx.setStrokeColor(red: r, green: g, blue: b,
                               alpha: 0.8 * alpha)
            ctx.setLineWidth(lineWidth * (alpha < 1 ? 0.8 : 1.0))
            ctx.beginPath()
            ctx.move(to: toPixel(face[indices[0]]))
            for i in indices.dropFirst() {
                ctx.addLine(to: toPixel(face[i]))
            }
            if closed { ctx.closePath() }
            ctx.strokePath()
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

#Preview {
    ContentView()
}
