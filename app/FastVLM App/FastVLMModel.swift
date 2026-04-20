//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import CoreImage
import FastVLM
import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import MLXVLM

enum FastVLMModelSize: String, CaseIterable, Identifiable {
    case small = "0.5B"
    case medium = "1.5B"
    case large = "7B"

    var id: String { rawValue }

    var directoryName: String {
        switch self {
        case .small:  return "model_0.5b"
        case .medium: return "model_1.5b"
        case .large:  return "model_7b"
        }
    }

    var label: String { rawValue }
}

@Observable
@MainActor
class FastVLMModel {

    public var running = false
    public var modelInfo = ""
    public var output = ""
    public var promptTime: String = ""

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }

    /// parameters controlling the output
    let generateParameters = GenerateParameters(temperature: 0.0)
    var maxTokens = 240

    /// update the display every N tokens -- 4 looks like it updates continuously
    /// and is low overhead.  observed ~15% reduction in tokens/s when updating
    /// on every token
    let displayEveryNTokens = 4

    private var loadState = LoadState.idle
    private var currentTask: Task<Void, Never>?

    /// The currently selected model size.
    public var selectedModelSize: FastVLMModelSize = .small

    /// Model sizes that are actually present in the app bundle.
    public private(set) var availableModelSizes: [FastVLMModelSize] = []

    enum EvaluationState: String, CaseIterable {
        case idle = "Idle"
        case processingPrompt = "Processing Prompt"
        case generatingResponse = "Generating Response"
    }

    public var evaluationState = EvaluationState.idle

    public init() {
        FastVLM.register(modelFactory: VLMModelFactory.shared)
        refreshAvailableModels()
        if let first = availableModelSizes.first {
            selectedModelSize = first
        }
    }

    public func refreshAvailableModels() {
        let dirs = FastVLM.availableModelDirectories()
        availableModelSizes = FastVLMModelSize.allCases.filter { size in
            dirs.contains(size.directoryName)
        }
    }

    /// Unloads the current model and loads the given size.
    public func switchModel(to size: FastVLMModelSize) async {
        cancel()
        loadState = .idle
        selectedModelSize = size
        modelInfo = "Switching to \(size.label)…"
        output = ""
        promptTime = ""
        MLX.GPU.clearCache()
        await load()
    }

    private func _load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let config = FastVLM.modelConfiguration(directory: selectedModelSize.directoryName)
            let modelContainer = try await VLMModelFactory.shared.loadContainer(
                configuration: config
            ) { [config] progress in
                Task { @MainActor in
                    self.modelInfo =
                        "Loading \(config.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            self.modelInfo = "Loaded \(selectedModelSize.label)"
            loadState = .loaded(modelContainer)
            return modelContainer

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    public func load() async {
        do {
            _ = try await _load()
        } catch {
            self.modelInfo = "Error loading model: \(error)"
        }
    }

    public func generate(_ userInput: UserInput) async -> Task<Void, Never> {
        if let currentTask, running {
            return currentTask
        }

        running = true
        
        currentTask?.cancel()

        let task = Task {
            do {
                let modelContainer = try await _load()

                MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))
                
                if Task.isCancelled { return }

                let currentMaxTokens = self.maxTokens
                let result = try await modelContainer.perform { context in
                    Task { @MainActor in
                        evaluationState = .processingPrompt
                    }

                    let llmStart = Date()
                    let input = try await context.processor.prepare(input: userInput)
                    
                    var seenFirstToken = false

                    let result = try MLXLMCommon.generate(
                        input: input, parameters: generateParameters, context: context
                    ) { tokens in
                        if Task.isCancelled {
                            return .stop
                        }

                        if !seenFirstToken {
                            seenFirstToken = true
                            let llmDuration = Date().timeIntervalSince(llmStart)
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                evaluationState = .generatingResponse
                                self.output = text
                                self.promptTime = "\(Int(llmDuration * 1000)) ms"
                            }
                        }

                        if tokens.count % displayEveryNTokens == 0 {
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                self.output = text
                            }
                        }

                        if tokens.count >= currentMaxTokens {
                            return .stop
                        } else {
                            return .more
                        }
                    }
                    
                    return result
                }
                
                if !Task.isCancelled {
                    self.output = result.output
                }

            } catch {
                if !Task.isCancelled {
                    output = "Failed: \(error)"
                }
            }

            if evaluationState == .generatingResponse {
                evaluationState = .idle
            }

            running = false
        }
        
        currentTask = task
        return task
    }
    
    public func cancel() {
        currentTask?.cancel()
        currentTask = nil
        running = false
        output = ""
        promptTime = ""
    }
}
