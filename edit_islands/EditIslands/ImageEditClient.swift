import Foundation
import UIKit

enum ImageEditError: LocalizedError {
    case missingAPIKey(EditProvider)
    case badImage
    case http(Int, String)
    case provider(String)
    case timeout

    var errorDescription: String? {
        switch self {
        case .missingAPIKey(let p):
            return "Missing API key for \(p.displayName). Add it in API Keys."
        case .badImage:
            return "Could not encode the source image."
        case .http(let code, let body):
            return "HTTP \(code): \(body.prefix(280))"
        case .provider(let message):
            return message
        case .timeout:
            return "Edit timed out."
        }
    }
}

/// Calls the same four commercial editors as `genai_image_editing/run_edits.py`.
enum ImageEditClient {
    static func edit(
        provider: EditProvider,
        source: UIImage,
        prompt: String,
        mask: UIImage? = nil
    ) async throws -> UIImage {
        let key = APIKeyStore.key(for: provider)
        guard !key.isEmpty else { throw ImageEditError.missingAPIKey(provider) }
        guard let png = source.pngData() else { throw ImageEditError.badImage }
        let maskPNG = mask?.pngData()
        let guided = maskPNG == nil ? prompt : maskedPrompt(prompt)

        switch provider {
        case .nanoBanana:
            return try await editNanoBanana(apiKey: key, png: png, maskPNG: maskPNG, prompt: guided)
        case .qwenImage3:
            return try await editQwen(apiKey: key, png: png, maskPNG: maskPNG, prompt: guided)
        case .seedream5Pro:
            return try await editSeedream(apiKey: key, png: png, maskPNG: maskPNG, prompt: guided)
        case .gptImage2:
            let inpaint = mask.flatMap { openAIMaskPNG(visibleMask: $0) }
            return try await editGPTImage(apiKey: key, png: png, maskPNG: inpaint, prompt: guided)
        }
    }

    /// Tell the model the second image is a white=edit / black=keep mask.
    private static func maskedPrompt(_ prompt: String) -> String {
        """
        The first image is the photo. The second image is a mask of the same size: white pixels are the only region you may change, and black pixels must stay identical to the photo.
        Further change, only inside the white region: \(prompt)
        """
    }

    // MARK: - Nano Banana (Gemini)

    private static func editNanoBanana(apiKey: String, png: Data, maskPNG: Data?, prompt: String) async throws -> UIImage {
        let model = APIKeyStore.geminiImageModel
        let url = URL(string: "https://generativelanguage.googleapis.com/v1beta/models/\(model):generateContent?key=\(apiKey)")!
        var parts: [[String: Any]] = [
            ["text": prompt],
            ["inline_data": ["mime_type": "image/png", "data": png.base64EncodedString()]],
        ]
        if let maskPNG {
            parts.append(["inline_data": ["mime_type": "image/png", "data": maskPNG.base64EncodedString()]])
        }
        let payload: [String: Any] = [
            "contents": [
                [
                    "parts": parts
                ]
            ],
            "generationConfig": [
                "responseModalities": ["TEXT", "IMAGE"],
            ],
        ]
        let (data, http) = try await postJSON(url: url, headers: [:], payload: payload)
        guard (200..<300).contains(http.statusCode) else {
            throw ImageEditError.http(http.statusCode, String(data: data, encoding: .utf8) ?? "")
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let candidates = json["candidates"] as? [[String: Any]],
              let content = candidates.first?["content"] as? [String: Any],
              let parts = content["parts"] as? [[String: Any]]
        else {
            throw ImageEditError.provider("Nano Banana returned no candidates")
        }
        for part in parts {
            if let inline = part["inlineData"] as? [String: Any]
                ?? part["inline_data"] as? [String: Any],
               let b64Out = inline["data"] as? String,
               let bytes = Data(base64Encoded: b64Out),
               let image = UIImage(data: bytes) {
                return image
            }
        }
        throw ImageEditError.provider("Nano Banana returned no image")
    }

    // MARK: - Qwen

    private static func editQwen(apiKey: String, png: Data, maskPNG: Data?, prompt: String) async throws -> UIImage {
        let base = APIKeyStore.dashscopeBaseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let model = APIKeyStore.qwenImageModel
        let dataURI = "data:image/png;base64,\(png.base64EncodedString())"
        var content: [[String: Any]] = [
            ["image": dataURI],
        ]
        if let maskPNG {
            content.append(["image": "data:image/png;base64,\(maskPNG.base64EncodedString())"])
        }
        content.append(["text": prompt])
        let payload: [String: Any] = [
            "model": model,
            "input": [
                "messages": [
                    [
                        "role": "user",
                        "content": content,
                    ]
                ]
            ],
            "parameters": [
                "n": 1,
                "watermark": false,
                "prompt_extend": true,
            ],
        ]
        let url = URL(string: "\(base)/services/aigc/multimodal-generation/generation")!
        var headers = [
            "Authorization": "Bearer \(apiKey)",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        ]
        var (data, http) = try await postJSON(url: url, headers: headers, payload: payload)
        if !(200..<300).contains(http.statusCode) {
            headers.removeValue(forKey: "X-DashScope-Async")
            (data, http) = try await postJSON(url: url, headers: headers, payload: payload)
        }
        guard (200..<300).contains(http.statusCode) else {
            throw ImageEditError.http(http.statusCode, String(data: data, encoding: .utf8) ?? "")
        }
        let body = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        if let imageURL = qwenImageURL(from: body) {
            return try await downloadImage(url: imageURL)
        }
        let taskID = (body["output"] as? [String: Any])?["task_id"] as? String
            ?? body["task_id"] as? String
        guard let taskID else {
            throw ImageEditError.provider("Qwen did not return an image or task id")
        }
        let deadline = Date().addingTimeInterval(300)
        while Date() < deadline {
            try await Task.sleep(nanoseconds: 2_000_000_000)
            let statusURL = URL(string: "\(base)/tasks/\(taskID)")!
            var req = URLRequest(url: statusURL)
            req.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            let (sData, sHTTP) = try await URLSession.shared.data(for: req)
            guard let sHTTP = sHTTP as? HTTPURLResponse, (200..<300).contains(sHTTP.statusCode) else {
                continue
            }
            let task = try JSONSerialization.jsonObject(with: sData) as? [String: Any] ?? [:]
            let state = (
                (task["output"] as? [String: Any])?["task_status"] as? String
                    ?? task["task_status"] as? String
                    ?? ""
            ).uppercased()
            if ["SUCCEEDED", "SUCCESS"].contains(state) {
                guard let imageURL = qwenImageURL(from: task) else {
                    throw ImageEditError.provider("Qwen succeeded with no image URL")
                }
                return try await downloadImage(url: imageURL)
            }
            if ["FAILED", "CANCELED", "CANCELLED", "UNKNOWN"].contains(state) {
                throw ImageEditError.provider("Qwen task failed (\(state))")
            }
        }
        throw ImageEditError.timeout
    }

    private static func qwenImageURL(from body: [String: Any]) -> URL? {
        guard let choices = (body["output"] as? [String: Any])?["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? [[String: Any]],
              let image = content.first?["image"] as? String
        else { return nil }
        return URL(string: image)
    }

    // MARK: - Seedream

    private static func editSeedream(apiKey: String, png: Data, maskPNG: Data?, prompt: String) async throws -> UIImage {
        let base = APIKeyStore.arkBaseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let model = APIKeyStore.arkModel
        let dataURI = "data:image/png;base64,\(png.base64EncodedString())"
        let imageField: Any
        if let maskPNG {
            imageField = [dataURI, "data:image/png;base64,\(maskPNG.base64EncodedString())"]
        } else {
            imageField = dataURI
        }
        let payload: [String: Any] = [
            "model": model,
            "prompt": prompt,
            "image": imageField,
            "size": "2K",
            "output_format": "png",
            "response_format": "url",
            "watermark": false,
        ]
        let url = URL(string: "\(base)/images/generations")!
        let (data, http) = try await postJSON(
            url: url,
            headers: [
                "Authorization": "Bearer \(apiKey)",
                "Content-Type": "application/json",
            ],
            payload: payload
        )
        guard (200..<300).contains(http.statusCode) else {
            throw ImageEditError.http(http.statusCode, String(data: data, encoding: .utf8) ?? "")
        }
        let body = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        guard let items = body["data"] as? [[String: Any]], let item = items.first else {
            throw ImageEditError.provider("Seedream returned no image")
        }
        if let b64 = item["b64_json"] as? String,
           let bytes = Data(base64Encoded: b64),
           let image = UIImage(data: bytes) {
            return image
        }
        guard let urlString = item["url"] as? String, let imageURL = URL(string: urlString) else {
            throw ImageEditError.provider("Seedream returned no url")
        }
        return try await downloadImage(url: imageURL)
    }

    // MARK: - GPT Image

    private static func editGPTImage(apiKey: String, png: Data, maskPNG: Data?, prompt: String) async throws -> UIImage {
        let model = APIKeyStore.openaiImageModel
        let boundary = "Boundary-\(UUID().uuidString)"
        var body = Data()
        func appendField(_ name: String, _ value: String) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(value)\r\n".data(using: .utf8)!)
        }
        appendField("model", model)
        appendField("prompt", prompt)
        appendField("output_format", "png")
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"image\"; filename=\"source.png\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
        body.append(png)
        body.append("\r\n".data(using: .utf8)!)
        if let maskPNG {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"mask\"; filename=\"mask.png\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
            body.append(maskPNG)
            body.append("\r\n".data(using: .utf8)!)
        }
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        var request = URLRequest(url: URL(string: "https://api.openai.com/v1/images/edits")!)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        request.timeoutInterval = 300

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ImageEditError.provider("No HTTP response from GPT Image")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw ImageEditError.http(http.statusCode, String(data: data, encoding: .utf8) ?? "")
        }
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        guard let items = json["data"] as? [[String: Any]],
              let b64 = items.first?["b64_json"] as? String,
              let bytes = Data(base64Encoded: b64),
              let image = UIImage(data: bytes)
        else {
            throw ImageEditError.provider("GPT Image returned no image payload")
        }
        return image
    }

    /// OpenAI edits: transparent pixels are the region that may change.
    private static func openAIMaskPNG(visibleMask: UIImage) -> Data? {
        guard let cg = visibleMask.cgImage else { return nil }
        let width = cg.width
        let height = cg.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        for i in 0..<(width * height) {
            let o = i * 4
            let editable = pixels[o] > 128 || pixels[o + 1] > 128 || pixels[o + 2] > 128
            if editable {
                pixels[o] = 0
                pixels[o + 1] = 0
                pixels[o + 2] = 0
                pixels[o + 3] = 0
            } else {
                pixels[o] = 255
                pixels[o + 1] = 255
                pixels[o + 2] = 255
                pixels[o + 3] = 255
            }
        }
        guard let outCtx = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let out = outCtx.makeImage() else { return nil }
        return UIImage(cgImage: out).pngData()
    }

    // MARK: - HTTP helpers

    private static func postJSON(
        url: URL,
        headers: [String: String],
        payload: [String: Any]
    ) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 300
        for (k, v) in headers {
            request.setValue(v, forHTTPHeaderField: k)
        }
        if request.value(forHTTPHeaderField: "Content-Type") == nil {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ImageEditError.provider("No HTTP response")
        }
        return (data, http)
    }

    private static func downloadImage(url: URL) async throws -> UIImage {
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw ImageEditError.provider("Failed to download result image")
        }
        guard let image = UIImage(data: data) else {
            throw ImageEditError.badImage
        }
        return image
    }
}
