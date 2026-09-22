import Foundation

/// Commercial image editors from the evaluation harness comparison set.
enum EditProvider: String, CaseIterable, Identifiable, Codable {
    case nanoBanana = "nano_banana"
    case qwenImage3 = "qwen_image_3"
    case seedream5Pro = "seedream_5_pro"
    case gptImage2 = "gpt_image_2"

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .nanoBanana: return "Nano Banana"
        case .qwenImage3: return "Qwen-Image-3"
        case .seedream5Pro: return "Seedream 5.0 Pro"
        case .gptImage2: return "GPT Image 2.5"
        }
    }

    var shortName: String {
        switch self {
        case .nanoBanana: return "Nano Banana"
        case .qwenImage3: return "Qwen"
        case .seedream5Pro: return "Seedream"
        case .gptImage2: return "GPT"
        }
    }

    var systemImage: String {
        switch self {
        case .nanoBanana: return "leaf.fill"
        case .qwenImage3: return "q.circle.fill"
        case .seedream5Pro: return "sparkles"
        case .gptImage2: return "brain.head.profile"
        }
    }

    var apiKeyEnvName: String {
        switch self {
        case .nanoBanana: return "GEMINI_API_KEY"
        case .qwenImage3: return "DASHSCOPE_API_KEY"
        case .seedream5Pro: return "ARK_API_KEY"
        case .gptImage2: return "OPENAI_API_KEY"
        }
    }

    var keyHelpURL: URL {
        switch self {
        case .nanoBanana: return URL(string: "https://aistudio.google.com/apikey")!
        case .qwenImage3: return URL(string: "https://modelstudio.console.alibabacloud.com/")!
        case .seedream5Pro: return URL(string: "https://console.byteplus.com/ark")!
        case .gptImage2: return URL(string: "https://platform.openai.com/api-keys")!
        }
    }

    var defaultModel: String {
        switch self {
        case .nanoBanana: return "gemini-3.1-flash-image"
        case .qwenImage3: return "qwen-image-3.0-pro"
        case .seedream5Pro: return "seedream-5-0-pro-260628"
        case .gptImage2: return "gpt-image-2.5-sunburst"
        }
    }
}
