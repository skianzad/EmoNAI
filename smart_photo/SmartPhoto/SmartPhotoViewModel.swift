import PhotosUI
import SwiftUI
import UIKit

/// Identifies which approaches to run.
struct ApproachSelection {
    var landmarks: Bool = true
    var posterV2: Bool = false
    var fecNet: Bool = false

    var any: Bool { landmarks || posterV2 || fecNet }
}

/// One candidate's result for a single approach.
struct ScoredCandidate: Identifiable {
    let id: String
    let image: UIImage
    let score: Float
    /// nil if face detection failed on this candidate.
    let faceDetected: Bool
}

/// A picked still or Live Photo frame, with an optional face crop for display.
struct CandidatePhoto: Identifiable {
    let id: String
    let image: UIImage
    let faceCrop: UIImage?

    var preview: UIImage { faceCrop ?? image }
}

/// Results for one approach.
struct ApproachResult: Identifiable {
    let id: String          // "A", "B", "C"
    let name: String        // "Landmarks", "POSTER V2", "FECNet"
    let candidates: [ScoredCandidate]
    let failedCount: Int    // how many candidates had no face
}

@MainActor
final class SmartPhotoViewModel: ObservableObject {

    // MARK: - Input state

    @Published var referenceItem: PhotosPickerItem?
    @Published var candidateItems: [PhotosPickerItem] = []
    @Published var livePhotoItem: PhotosPickerItem?

    @Published var referenceImage: UIImage?
    @Published var referenceFaceCrop: UIImage?
    @Published var candidateImages: [CandidatePhoto] = []

    /// When on, the candidate picker takes one Live Photo and expands it into frames.
    @Published var livePhotoMode = false
    @Published var livePhotoError: String?
    @Published var isExtracting = false

    @Published var approaches = ApproachSelection()

    // MARK: - Output state

    @Published var results: [ApproachResult] = []
    @Published var isRunning = false
    @Published var progress: Double = 0
    @Published var statusMessage: String = ""

    // MARK: - Photo loading

    func loadReference() async {
        guard let item = referenceItem else { return }
        if let data = try? await item.loadTransferable(type: Data.self),
           let img = UIImage(data: data) {
            referenceImage = img
            referenceFaceCrop = FaceDetector.faceImage(from: img)
        }
    }

    func loadCandidates() async {
        var loaded: [CandidatePhoto] = []
        for (i, item) in candidateItems.enumerated() {
            if let data = try? await item.loadTransferable(type: Data.self),
               let img = UIImage(data: data) {
                loaded.append(CandidatePhoto(
                    id: "candidate_\(i)",
                    image: img,
                    faceCrop: FaceDetector.faceImage(from: img)))
            }
        }
        candidateImages = loaded
    }

    func loadLivePhotoFrames() async {
        livePhotoError = nil
        guard let item = livePhotoItem else {
            candidateImages = []
            return
        }
        isExtracting = true
        candidateImages = []
        statusMessage = "Extracting Live Photo frames..."
        do {
            let frames = try await LivePhotoFrameExtractor.frames(from: item)
            statusMessage = "Cropping faces..."
            candidateImages = frames.enumerated().map { i, img in
                CandidatePhoto(
                    id: "frame_\(i)",
                    image: img,
                    faceCrop: FaceDetector.faceImage(from: img))
            }
            statusMessage = "\(frames.count) frames"
        } catch {
            candidateImages = []
            livePhotoError = error.localizedDescription
            statusMessage = ""
            print("[LivePhoto] \(error)")
        }
        isExtracting = false
    }

    // MARK: - Run

    func run() async {
        guard let refImage = referenceImage, !candidateImages.isEmpty, approaches.any else { return }
        isRunning = true
        progress = 0
        results = []
        statusMessage = "Starting..."

        var allResults: [ApproachResult] = []
        var approachList: [(id: String, name: String, encodeFn: (UIImage) -> [Float]?)] = []

        if approaches.landmarks {
            approachList.append(("A", "Landmarks", { ApproachA.encode(image: $0) }))
        }
        if approaches.posterV2 {
            approachList.append(("B", "POSTER V2", { ApproachB.encode(image: $0) }))
        }
        if approaches.fecNet {
            approachList.append(("C", "FECNet", { ApproachC.encode(image: $0) }))
        }

        let totalWork = Double(approachList.count * (1 + candidateImages.count))
        var completed = 0.0

        for approach in approachList {
            statusMessage = "\(approach.name): encoding reference..."
            let refEmbedding = approach.encodeFn(refImage)
            completed += 1
            progress = completed / totalWork

            guard let refEmb = refEmbedding else {
                allResults.append(ApproachResult(
                    id: approach.id, name: approach.name,
                    candidates: [], failedCount: candidateImages.count))
                completed += Double(candidateImages.count)
                progress = completed / totalWork
                continue
            }

            var candidateEmbeddings: [(id: String, embedding: [Float], image: UIImage)] = []
            var noFaceImages: [(id: String, image: UIImage)] = []

            for (i, cand) in candidateImages.enumerated() {
                statusMessage = "\(approach.name): encoding \(i + 1)/\(candidateImages.count)..."
                if let emb = approach.encodeFn(cand.image) {
                    candidateEmbeddings.append((
                        id: cand.id,
                        embedding: emb,
                        image: cand.faceCrop ?? cand.image))
                } else {
                    noFaceImages.append((id: cand.id, image: cand.preview))
                }
                completed += 1
                progress = completed / totalWork
            }

            let ranked: [(id: String, score: Float)]
            if approach.id == "A" {
                ranked = Matcher.rankCandidates(
                    reference: refEmb,
                    candidates: candidateEmbeddings.map { (id: $0.id, embedding: $0.embedding) },
                    metric: .euclidean,
                    scales: ApproachA.featureScales)
            } else {
                ranked = Matcher.rankCandidates(
                    reference: refEmb,
                    candidates: candidateEmbeddings.map { (id: $0.id, embedding: $0.embedding) })
            }

            let imageMap = Dictionary(
                uniqueKeysWithValues: candidateEmbeddings.map { ($0.id, $0.image) })

            var scoredCandidates = ranked.compactMap { r -> ScoredCandidate? in
                guard let img = imageMap[r.id] else { return nil }
                return ScoredCandidate(id: r.id, image: img, score: r.score, faceDetected: true)
            }

            // Append no-face candidates at the end with score 0
            for nf in noFaceImages {
                scoredCandidates.append(ScoredCandidate(
                    id: nf.id, image: nf.image, score: 0, faceDetected: false))
            }

            allResults.append(ApproachResult(
                id: approach.id, name: approach.name,
                candidates: scoredCandidates, failedCount: noFaceImages.count))
        }

        results = allResults
        statusMessage = "Done"
        isRunning = false
    }
}
