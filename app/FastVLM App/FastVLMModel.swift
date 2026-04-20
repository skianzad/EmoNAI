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

    private let modelConfiguration = FastVLM.modelConfiguration

    /// parameters controlling the output
    let generateParameters = GenerateParameters(temperature: 0.0)
    var maxTokens = 240

    /// update the display every N tokens -- 4 looks like it updates continuously
    /// and is low overhead.  observed ~15% reduction in tokens/s when updating
    /// on every token
    let displayEveryNTokens = 4

    private var loadState = LoadState.idle
    private var currentTask: Task<Void, Never>?

    enum EvaluationState: String, CaseIterable {
        case idle = "Idle"
        case processingPrompt = "Processing Prompt"
        case generatingResponse = "Generating Response"
    }

    public var evaluationState = EvaluationState.idle

    public init() {
        FastVLM.register(modelFactory: VLMModelFactory.shared)
    }

    private func _load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            // limit the buffer cache
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let modelContainer = try await VLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) {
                [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            self.modelInfo = "Loaded"
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
        
        // Cancel any existing task
        currentTask?.cancel()

        // Create new task and store reference
        let task = Task {
            do {
                let modelContainer = try await _load()

                // each time you generate you will get something new
                MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))
                
                // Check if task was cancelled
                if Task.isCancelled { return }

                let currentMaxTokens = self.maxTokens
                let outputText: String = try await modelContainer.perform { context in

                    Task { @MainActor in
                        evaluationState = .processingPrompt
                    }

                    let llmStart = Date()
                    let input = try await context.processor.prepare(input: userInput)
                    let prepareEnd = Date()
                    let prepMs = Int(prepareEnd.timeIntervalSince(llmStart) * 1000)
                    print("[PERF] prepare (vision+preprocess): \(prepMs) ms")

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
                            let prefillMs = Int(llmDuration * 1000) - prepMs
                            let tokenText = context.tokenizer.decode(tokens: tokens)
                            print("[PERF] LLM prefill: \(prefillMs) ms, total TTFT: \(Int(llmDuration * 1000)) ms")
                            Task { @MainActor in
                                evaluationState = .generatingResponse
                                self.output = tokenText
                                self.promptTime = "\(prepMs)+\(prefillMs) ms"
                            }
                        }

                        if tokens.count % displayEveryNTokens == 0 {
                            let tokenText = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor in
                                self.output = tokenText
                            }
                        }

                        if tokens.count >= currentMaxTokens {
                            return .stop
                        } else {
                            return .more
                        }
                    }
                    let text = result.output

                    // Aggressively free GPU memory between generations
                    MLX.GPU.set(cacheLimit: 0)
                    MLX.GPU.clearCache()
                    MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

                    return text
                }

                if !Task.isCancelled {
                    self.output = outputText
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
