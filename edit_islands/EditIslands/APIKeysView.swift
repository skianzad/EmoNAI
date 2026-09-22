import SwiftUI

struct APIKeysView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var drafts: [String: String] = [:]
    @State private var dashscopeBase = APIKeyStore.dashscopeBaseURL
    @State private var arkBase = APIKeyStore.arkBaseURL
    @State private var arkModel = APIKeyStore.arkModel
    @State private var openaiModel = APIKeyStore.openaiImageModel
    @State private var geminiModel = APIKeyStore.geminiImageModel
    @State private var qwenModel = APIKeyStore.qwenImageModel

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text("Keys stay on-device in the Keychain. Same env names as genai_image_editing/.env.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Button("Reload from Secrets.plist") {
                        APIKeyStore.reseedFromBundledSecrets()
                        load()
                    }
                }

                ForEach(EditProvider.allCases) { provider in
                    Section(provider.displayName) {
                        SecureField(provider.apiKeyEnvName, text: binding(for: provider.apiKeyEnvName))
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                        Link("Get API key", destination: provider.keyHelpURL)
                            .font(.caption)
                    }
                }

                Section("Model overrides") {
                    TextField("Gemini model", text: $geminiModel)
                    TextField("Qwen model", text: $qwenModel)
                    TextField("Ark / Seedream model", text: $arkModel)
                    TextField("OpenAI image model", text: $openaiModel)
                }

                Section("Base URLs") {
                    TextField("DashScope base", text: $dashscopeBase)
                    TextField("Ark base", text: $arkBase)
                }
            }
            .navigationTitle("API Keys")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { save(); dismiss() }
                }
            }
            .onAppear { load() }
        }
    }

    private func binding(for env: String) -> Binding<String> {
        Binding(
            get: { drafts[env] ?? "" },
            set: { drafts[env] = $0 }
        )
    }

    private func load() {
        for provider in EditProvider.allCases {
            drafts[provider.apiKeyEnvName] = APIKeyStore.get(provider.apiKeyEnvName)
        }
    }

    private func save() {
        for provider in EditProvider.allCases {
            APIKeyStore.set(provider.apiKeyEnvName, value: drafts[provider.apiKeyEnvName] ?? "")
        }
        APIKeyStore.dashscopeBaseURL = dashscopeBase
        APIKeyStore.arkBaseURL = arkBase
        APIKeyStore.arkModel = arkModel
        APIKeyStore.openaiImageModel = openaiModel
        APIKeyStore.geminiImageModel = geminiModel
        APIKeyStore.qwenImageModel = qwenModel
    }
}
