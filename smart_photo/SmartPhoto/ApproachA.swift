import Foundation
import UIKit
import Vision

/// Approach A: Landmark geometry — head pose + geometric ratios.
/// No Core ML model needed. Pure arithmetic on Vision landmarks.
///
/// Feature vector (8 dimensions):
///   [yaw, pitch, roll, ear_left, ear_right, mouth_open, eye_mouth_ratio, face_aspect]
///
/// Ranking uses weighted Euclidean distance (not cosine, not batch z-score).
/// Cosine on 8 mixed-scale numbers is dominated by face aspect / eye–mouth
/// distance. Batch z-score on a Live Photo is worse: consecutive frames have
/// near-zero std, so sensor jitter becomes the ranking signal.
enum ApproachA {

    /// Per-dimension scales ≈ one std of natural variation. Larger = less weight.
    /// Face aspect is down-weighted; it tracks framing, not expression.
    static let featureScales: [Float] = [
        0.25, // yaw (rad)
        0.20, // pitch
        0.20, // roll
        0.05, // ear left
        0.05, // ear right
        0.08, // mouth open
        0.10, // eye–mouth ratio
        0.50, // face aspect
    ]

    static func encode(observation: VNFaceObservation) -> [Float]? {
        guard let landmarks = observation.landmarks else { return nil }

        let yaw   = Float(observation.yaw?.doubleValue   ?? 0)
        let pitch = Float(observation.pitch?.doubleValue  ?? 0)
        let roll  = Float(observation.roll?.doubleValue   ?? 0)

        let earLeft  = aspectRatio(landmarks.leftEye)
        let earRight = aspectRatio(landmarks.rightEye)
        let mouthOpen = mouthOpenRatio(
            outerLips: landmarks.outerLips,
            innerLips: landmarks.innerLips)
        let eyeMouthRatio = eyeMouthDistanceRatio(
            leftEye: landmarks.leftEye,
            rightEye: landmarks.rightEye,
            outerLips: landmarks.outerLips)

        let box = observation.boundingBox
        let faceAspect = Float(box.width / max(box.height, 1e-6))

        return [yaw, pitch, roll, earLeft, earRight, mouthOpen, eyeMouthRatio, faceAspect]
    }

    static func encode(image: UIImage) -> [Float]? {
        // Do not roll-align: yaw/pitch/roll are the features.
        guard let detection = FaceDetector.detect(in: image, align: false) else { return nil }
        return encode(observation: detection.observation)
    }

    // MARK: - Geometric helpers

    /// Height/width of a landmark contour. Vision eyes have 8 points (not
    /// dlib's 6), so the classic EAR index pairs (0–3, 1–5, 2–4) are wrong.
    private static func aspectRatio(_ region: VNFaceLandmarkRegion2D?) -> Float {
        guard let pts = region?.normalizedPoints, pts.count >= 4 else { return 0 }
        let xs = pts.map(\.x)
        let ys = pts.map(\.y)
        let width = (xs.max() ?? 0) - (xs.min() ?? 0)
        let height = (ys.max() ?? 0) - (ys.min() ?? 0)
        guard width > 1e-6 else { return 0 }
        return Float(height / width)
    }

    /// Inner-lip vertical span over outer-lip width.
    /// Points walk the contour, so index 0 and n/2 are the corners (width),
    /// not top and bottom. Using those made mouth_open nearly constant.
    private static func mouthOpenRatio(
        outerLips: VNFaceLandmarkRegion2D?,
        innerLips: VNFaceLandmarkRegion2D?
    ) -> Float {
        let inner = innerLips?.normalizedPoints ?? []
        let outer = outerLips?.normalizedPoints ?? []
        let source = inner.count >= 4 ? inner : outer
        guard source.count >= 4, outer.count >= 4 else { return 0 }

        let innerH = span(source, axis: \.y)
        let outerW = span(outer, axis: \.x)
        guard outerW > 1e-6 else { return 0 }
        return Float(innerH / outerW)
    }

    private static func eyeMouthDistanceRatio(
        leftEye: VNFaceLandmarkRegion2D?,
        rightEye: VNFaceLandmarkRegion2D?,
        outerLips: VNFaceLandmarkRegion2D?
    ) -> Float {
        guard let le = leftEye, let re = rightEye, let lips = outerLips else { return 0 }
        let lePts = le.normalizedPoints
        let rePts = re.normalizedPoints
        let lipPts = lips.normalizedPoints
        guard !lePts.isEmpty, !rePts.isEmpty, !lipPts.isEmpty else { return 0 }

        let leftCenter = centroid(lePts)
        let rightCenter = centroid(rePts)
        let eyeMid = CGPoint(
            x: (leftCenter.x + rightCenter.x) / 2,
            y: (leftCenter.y + rightCenter.y) / 2)
        let mouthCenter = centroid(lipPts)

        let interOcular = distance(leftCenter, rightCenter)
        guard interOcular > 1e-6 else { return 0 }
        return Float(distance(eyeMid, mouthCenter) / interOcular)
    }

    private static func span(_ pts: [CGPoint], axis: KeyPath<CGPoint, CGFloat>) -> CGFloat {
        let vals = pts.map { $0[keyPath: axis] }
        return (vals.max() ?? 0) - (vals.min() ?? 0)
    }

    private static func distance(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        hypot(a.x - b.x, a.y - b.y)
    }

    private static func centroid(_ pts: [CGPoint]) -> CGPoint {
        let sum = pts.reduce(CGPoint.zero) {
            CGPoint(x: $0.x + $1.x, y: $0.y + $1.y)
        }
        return CGPoint(x: sum.x / CGFloat(pts.count),
                       y: sum.y / CGFloat(pts.count))
    }
}
