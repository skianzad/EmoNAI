import Foundation
import Security

/// Persists API keys in the Keychain (one entry per env-style name).
enum APIKeyStore {
    private static let service = "com.SensciLab.EditIslands.keys"
    private static let seededFlag = "EditIslands.secretsSeededFromBundle"

    /// Load `Secrets.plist` from the app bundle (copied from genai_image_editing/.env) into Keychain.
    static func seedFromBundledSecretsIfNeeded(force: Bool = false) {
        guard force || !UserDefaults.standard.bool(forKey: seededFlag) else { return }
        guard let url = Bundle.main.url(forResource: "Secrets", withExtension: "plist"),
              let dict = NSDictionary(contentsOf: url) as? [String: String]
        else { return }

        let keyNames: Set<String> = [
            "GEMINI_API_KEY",
            "DASHSCOPE_API_KEY",
            "ARK_API_KEY",
            "OPENAI_API_KEY",
            "BFL_API_KEY",
            "IDEOGRAM_API_KEY",
        ]
        for (key, value) in dict {
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            if keyNames.contains(key) {
                set(key, value: trimmed)
            } else if key == "DASHSCOPE_BASE_URL" {
                dashscopeBaseURL = trimmed
            } else if key == "ARK_BASE_URL" {
                arkBaseURL = trimmed
            } else if key == "ARK_MODEL" {
                arkModel = trimmed
            } else if key == "OPENAI_IMAGE_MODEL" {
                openaiImageModel = trimmed
            } else if key == "GEMINI_IMAGE_MODEL" {
                geminiImageModel = trimmed
            } else if key == "QWEN_IMAGE_MODEL" {
                qwenImageModel = trimmed
            }
        }
        UserDefaults.standard.set(true, forKey: seededFlag)
    }

    /// Re-read Secrets.plist and overwrite Keychain (useful after updating .env → Secrets).
    static func reseedFromBundledSecrets() {
        UserDefaults.standard.set(false, forKey: seededFlag)
        seedFromBundledSecretsIfNeeded(force: true)
    }

    static func get(_ envName: String) -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: envName,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess,
              let data = item as? Data,
              let value = String(data: data, encoding: .utf8)
        else { return "" }
        return value
    }

    static func set(_ envName: String, value: String) {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let deleteQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: envName,
        ]
        SecItemDelete(deleteQuery as CFDictionary)
        guard !trimmed.isEmpty else { return }
        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: envName,
            kSecValueData as String: Data(trimmed.utf8),
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        ]
        SecItemAdd(addQuery as CFDictionary, nil)
    }

    static func key(for provider: EditProvider) -> String {
        get(provider.apiKeyEnvName)
    }

    static func hasKey(for provider: EditProvider) -> Bool {
        !key(for: provider).isEmpty
    }

    // Optional overrides (UserDefaults — non-secret)
    static var dashscopeBaseURL: String {
        get { UserDefaults.standard.string(forKey: "DASHSCOPE_BASE_URL") ?? "https://dashscope-intl.aliyuncs.com/api/v1" }
        set { UserDefaults.standard.set(newValue, forKey: "DASHSCOPE_BASE_URL") }
    }

    static var arkBaseURL: String {
        get { UserDefaults.standard.string(forKey: "ARK_BASE_URL") ?? "https://ark.ap-southeast.bytepluses.com/api/v3" }
        set { UserDefaults.standard.set(newValue, forKey: "ARK_BASE_URL") }
    }

    static var arkModel: String {
        get { UserDefaults.standard.string(forKey: "ARK_MODEL") ?? EditProvider.seedream5Pro.defaultModel }
        set { UserDefaults.standard.set(newValue, forKey: "ARK_MODEL") }
    }

    static var openaiImageModel: String {
        get { UserDefaults.standard.string(forKey: "OPENAI_IMAGE_MODEL") ?? EditProvider.gptImage2.defaultModel }
        set { UserDefaults.standard.set(newValue, forKey: "OPENAI_IMAGE_MODEL") }
    }

    static var geminiImageModel: String {
        get { UserDefaults.standard.string(forKey: "GEMINI_IMAGE_MODEL") ?? EditProvider.nanoBanana.defaultModel }
        set { UserDefaults.standard.set(newValue, forKey: "GEMINI_IMAGE_MODEL") }
    }

    static var qwenImageModel: String {
        get { UserDefaults.standard.string(forKey: "QWEN_IMAGE_MODEL") ?? EditProvider.qwenImage3.defaultModel }
        set { UserDefaults.standard.set(newValue, forKey: "QWEN_IMAGE_MODEL") }
    }
}
