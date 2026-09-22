import Foundation

enum Matcher {

    static func l2Normalize(_ v: [Float]) -> [Float] {
        let norm = sqrt(v.reduce(0) { $0 + $1 * $1 }) + 1e-8
        return v.map { $0 / norm }
    }

    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    /// Weighted L2, then mapped to (0, 1] so the results UI stays comparable.
    /// Identical vectors → 1. Does not L2-normalize (that would throw away magnitude).
    static func euclideanSimilarity(_ a: [Float], _ b: [Float], scales: [Float]) -> Float {
        let n = min(a.count, b.count, scales.count)
        guard n > 0 else { return 0 }
        var sum: Float = 0
        for i in 0..<n {
            let d = (a[i] - b[i]) / max(scales[i], 1e-8)
            sum += d * d
        }
        return exp(-sqrt(sum))
    }

    static func rankCandidates(
        reference: [Float],
        candidates: [(id: String, embedding: [Float])],
        metric: Metric = .cosine,
        scales: [Float] = []
    ) -> [(id: String, score: Float)] {
        switch metric {
        case .cosine:
            let refNorm = l2Normalize(reference)
            return candidates
                .map { (id: $0.id, score: cosineSimilarity(refNorm, l2Normalize($0.embedding))) }
                .sorted { $0.score > $1.score }
        case .euclidean:
            return candidates
                .map { (id: $0.id, score: euclideanSimilarity(reference, $0.embedding, scales: scales)) }
                .sorted { $0.score > $1.score }
        }
    }

    enum Metric {
        case cosine
        case euclidean
    }
}
