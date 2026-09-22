import SwiftUI

@main
struct EditIslandsApp: App {
    init() {
        // Always reseed from bundled Secrets.plist so .env updates apply on rebuild.
        APIKeyStore.reseedFromBundledSecrets()
    }

    var body: some Scene {
        WindowGroup {
            HomeView()
        }
    }
}
