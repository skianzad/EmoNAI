import SwiftUI

struct ResultsView: View {
    let referenceImage: UIImage?
    let results: [ApproachResult]

    @State private var selectedTab: String = ""

    var body: some View {
        VStack(spacing: 0) {
            // Reference photo pinned at top
            if let ref = referenceImage {
                VStack(spacing: 4) {
                    Text("Reference")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Image(uiImage: ref)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 80, height: 80)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                        .shadow(radius: 3)
                }
                .padding(.top, 8)
                .padding(.bottom, 4)
            }

            if results.isEmpty {
                ContentUnavailableView(
                    "No Results",
                    systemImage: "photo.badge.exclamationmark",
                    description: Text("No approaches produced results."))
            } else if results.count == 1 {
                // Single approach — no tabs needed
                approachGrid(results[0])
            } else {
                // Multiple approaches — tab view
                TabView(selection: $selectedTab) {
                    ForEach(results) { result in
                        approachGrid(result)
                            .tabItem {
                                Label(result.name, systemImage: tabIcon(result.id))
                            }
                            .tag(result.id)
                    }
                }
            }
        }
        .navigationTitle("Results")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            if selectedTab.isEmpty, let first = results.first {
                selectedTab = first.id
            }
        }
    }

    @ViewBuilder
    private func approachGrid(_ result: ApproachResult) -> some View {
        VStack(spacing: 4) {
            HStack {
                Text(result.name)
                    .font(.headline)
                Spacer()
                if result.failedCount > 0 {
                    Text("\(result.failedCount) no face")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
            .padding(.horizontal)
            .padding(.top, 4)

            ScrollView {
                LazyVGrid(
                    columns: [
                        GridItem(.adaptive(minimum: 100, maximum: 140), spacing: 8)
                    ],
                    spacing: 8
                ) {
                    ForEach(result.candidates) { candidate in
                        candidateCell(candidate)
                    }
                }
                .padding(.horizontal)
                .padding(.bottom, 16)
            }
        }
    }

    @ViewBuilder
    private func candidateCell(_ candidate: ScoredCandidate) -> some View {
        VStack(spacing: 2) {
            Image(uiImage: candidate.image)
                .resizable()
                .scaledToFill()
                .frame(width: 100, height: 100)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(alignment: .bottom) {
                    Text(scoreLabel(candidate))
                        .font(.caption2.bold().monospacedDigit())
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            Capsule()
                                .fill(scoreColor(candidate).opacity(0.85))
                        )
                        .padding(.bottom, 4)
                }
                .overlay {
                    if !candidate.faceDetected {
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.black.opacity(0.5))
                            .overlay {
                                Image(systemName: "face.dashed")
                                    .foregroundStyle(.white)
                                    .font(.title3)
                            }
                    }
                }
        }
    }

    private func scoreLabel(_ c: ScoredCandidate) -> String {
        if !c.faceDetected { return "no face" }
        return String(format: "%.3f", c.score)
    }

    private func scoreColor(_ c: ScoredCandidate) -> Color {
        if !c.faceDetected { return .gray }
        if c.score > 0.8 { return .green }
        if c.score > 0.5 { return .blue }
        if c.score > 0.2 { return .orange }
        return .red
    }

    private func tabIcon(_ id: String) -> String {
        switch id {
        case "A": return "point.3.connected.trianglepath.dotted"
        case "B": return "brain.head.profile"
        case "C": return "arrow.triangle.branch"
        default: return "questionmark"
        }
    }
}
