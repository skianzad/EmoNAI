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

    /// Latest MediaPipe face result used to render the landmark overlay on the video.
    @State private var displayFaceResult: FaceLandmarkDisplayResult? = nil

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
                            if !enabled { displayFaceResult = nil }
                            if enabled {
                                prompt = "What is this person's facial expression?"
                                promptSuffix = "Output only one or two words."
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
                                       let result = displayFaceResult {
                                        FaceLandmarkOverlay(result: result)
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
                                prompt = "Describe the emotion shown in this face."
                                promptSuffix = "Output only one or two words."
                            }
                            Button("Face landmarks — describe") {
                                faceLandmarkModeEnabled = true
                                prompt = "Describe the face shown in this image."
                                promptSuffix = "Output should be brief, about 10 words or less."
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
        for await frame in frames {
            let imageForVLM: CIImage

            if faceLandmarkModeEnabled {
                guard let result = faceLandmarker.detectAndCrop(in: frame) else {
                    // No face detected — skip this frame.
                    continue
                }
                imageForVLM = result.croppedImage
            } else {
                imageForVLM = CIImage(cvPixelBuffer: frame)
            }

            let userInput = UserInput(
                prompt: .text("\(prompt) \(promptSuffix)"),
                images: [.ciImage(imageForVLM)]
            )
            
            // generate output for a frame and wait for generation to complete
            let t = await model.generate(userInput)
            _ = await t.result

            do {
                try await Task.sleep(for: FRAME_DELAY)
            } catch { return }
        }
    }

    /// Runs MediaPipe face-landmark detection on display frames and updates
    /// `displayFaceResult` for the overlay.  Drops frames automatically
    /// when detection is slower than the camera frame rate (bufferingNewest(1)).
    func detectDisplayLandmarks(_ frames: AsyncStream<CVImageBuffer>) async {
        for await frame in frames {
            let result: FaceLandmarkDisplayResult?
            if faceLandmarkModeEnabled {
                result = faceLandmarker.detectObservation(in: frame)
            } else {
                result = nil
            }
            await MainActor.run { displayFaceResult = result }
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
    /// - Parameter frame: The frame to analyze.
    func processSingleFrame(_ frame: CVImageBuffer) {
        // Reset Response UI (show spinner)
        Task { @MainActor in
            model.output = ""
        }

        let imageForVLM: CIImage
        if faceLandmarkModeEnabled,
           let result = faceLandmarker.detectAndCrop(in: frame) {
            imageForVLM = result.croppedImage
        } else {
            // No face detected or mode disabled — use the full frame.
            imageForVLM = CIImage(cvPixelBuffer: frame)
        }

        // Construct request to model
        let userInput = UserInput(
            prompt: .text("\(prompt) \(promptSuffix)"),
            images: [.ciImage(imageForVLM)]
        )

        // Post request to FastVLM
        Task {
            await model.generate(userInput)
        }
    }
}

// MARK: - FaceLandmarkOverlay

/// Draws MediaPipe's 478-point face mesh over the video view.
///
/// Coordinate mapping accounts for `resizeAspectFill` —  the image may be
/// cropped when it doesn't match the 4:3 view ratio.
///
/// MediaPipe normalised coordinates use a **top-left** origin, matching
/// SwiftUI's Canvas coordinate system (no Y-flip needed).
private struct FaceLandmarkOverlay: View {
    let result: FaceLandmarkDisplayResult

    var body: some View {
        Canvas { ctx, size in
            draw(ctx: ctx, size: size)
        }
        .allowsHitTesting(false)
    }

    // MARK: - Drawing

    private func draw(ctx: GraphicsContext, size: CGSize) {
        let imgW = result.imageSize.width
        let imgH = result.imageSize.height
        guard imgW > 0, imgH > 0 else { return }

        // resizeAspectFill: scale until the image *covers* the view.
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

            // Feature connections
            drawConnections(ctx: ctx, face: face, toView: toView)

            // All 468 base landmarks as small dots
            for i in 0..<min(face.count, 468) {
                let vp = toView(face[i])
                ctx.fill(
                    Path(ellipseIn: CGRect(x: vp.x - 1.5, y: vp.y - 1.5,
                                           width: 3, height: 3)),
                    with: .color(.cyan.opacity(0.6)))
            }

            // Iris points — larger, brighter
            if face.count >= 478 {
                for i in 468..<478 {
                    let vp = toView(face[i])
                    let r: CGFloat = 4
                    let rect = CGRect(x: vp.x - r, y: vp.y - r,
                                      width: 2 * r, height: 2 * r)
                    ctx.fill(Path(ellipseIn: rect),
                             with: .color(.white.opacity(0.9)))
                    ctx.stroke(Path(ellipseIn: rect),
                               with: .color(.cyan), lineWidth: 1.5)
                }
            }
        }
    }

    // MARK: - Mesh connections

    private func drawConnections(
        ctx: GraphicsContext,
        face: [CGPoint],
        toView: (CGPoint) -> CGPoint
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
            ctx.stroke(path, with: .color(color.opacity(0.8)),
                       lineWidth: lineWidth)
        }

        strokePath(Self.faceOval,          color: .yellow, closed: true)
        strokePath(Self.leftEye,           color: .cyan,   closed: true)
        strokePath(Self.rightEye,          color: .cyan,   closed: true)
        strokePath(Self.leftEyebrowUpper,  color: .yellow)
        strokePath(Self.leftEyebrowLower,  color: .yellow)
        strokePath(Self.rightEyebrowUpper, color: .yellow)
        strokePath(Self.rightEyebrowLower, color: .yellow)
        strokePath(Self.lipsOuter,         color: .orange, closed: true)
        strokePath(Self.lipsInner,         color: .orange, lineWidth: 1, closed: true)
        strokePath(Self.noseBridge,        color: .yellow)
        strokePath(Self.noseBottom,        color: .yellow)

        if face.count >= 478 {
            strokePath(Self.leftIris,  color: .white, lineWidth: 1.5, closed: true)
            strokePath(Self.rightIris, color: .white, lineWidth: 1.5, closed: true)
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
