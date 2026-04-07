//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import CoreImage
import CoreVideo
import MediaPipeTasksVision

// MARK: - Results

/// Detected face data returned by `FaceLandmarkerService`.
struct FaceLandmarkResult {
    /// Frame cropped and padded to the face bounding box, ready for VLM inference.
    let croppedImage: CIImage
    /// 478-point MediaPipe face landmarks (normalised, top-left origin).
    let landmarks: [[CGPoint]]?
}

/// Lightweight result for driving the real-time landmark overlay.
struct FaceLandmarkDisplayResult {
    /// Per-face arrays of normalised (x, y) points.  Top-left origin, [0, 1].
    let landmarks: [[CGPoint]]
    /// Pixel dimensions of the source frame used for aspect-fill mapping.
    let imageSize: CGSize
}

// MARK: - Service

/// Detects faces using MediaPipe's FaceLandmarker (478 landmarks per face)
/// and optionally returns a padded crop for VLM inference.
///
/// Two separate `FaceLandmarker` instances are kept so that
/// `detectObservation` (display) and `detectAndCrop` (VLM) can be called
/// concurrently from different async streams without contention —
/// MediaPipe's `FaceLandmarker` is **not** thread-safe.
final class FaceLandmarkerService {

    // MARK: - Configuration

    /// Fractional padding around the bounding box before cropping.
    var padding: CGFloat = 0.25

    // MARK: - Private state

    private let displayLandmarker: FaceLandmarker?
    private let cropLandmarker: FaceLandmarker?

    // MARK: - Init

    init() {
        displayLandmarker = Self.makeLandmarker()
        cropLandmarker    = Self.makeLandmarker()
    }

    private static func makeLandmarker() -> FaceLandmarker? {
        guard let modelPath = Bundle.main.path(
            forResource: "face_landmarker", ofType: "task"
        ) else {
            print("[FaceLandmarker] face_landmarker.task not found in bundle")
            return nil
        }

        let opts = FaceLandmarkerOptions()
        opts.baseOptions.modelAssetPath = modelPath
        opts.runningMode = .image
        opts.numFaces = 1
        opts.minFaceDetectionConfidence = 0.5
        opts.minFacePresenceConfidence  = 0.5
        opts.minTrackingConfidence      = 0.5
        do {
            let landmarker = try FaceLandmarker(options: opts)
            print("[FaceLandmarker] Initialised successfully (\(modelPath))")
            return landmarker
        } catch {
            print("[FaceLandmarker] Init failed: \(error)")
            return nil
        }
    }

    // MARK: - Public API

    /// Detect face landmarks for the overlay — no crop produced.
    func detectObservation(in pixelBuffer: CVImageBuffer) -> FaceLandmarkDisplayResult? {
        guard let landmarker = displayLandmarker,
              let mpImage = try? MPImage(pixelBuffer: pixelBuffer),
              let result  = try? landmarker.detect(image: mpImage),
              !result.faceLandmarks.isEmpty else { return nil }

        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        let landmarks: [[CGPoint]] = result.faceLandmarks.map { face in
            face.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
        }
        return FaceLandmarkDisplayResult(
            landmarks: landmarks, imageSize: CGSize(width: w, height: h))
    }

    /// Detect the largest face and return a padded crop for VLM inference.
    func detectAndCrop(in pixelBuffer: CVImageBuffer) -> FaceLandmarkResult? {
        guard let landmarker = cropLandmarker,
              let mpImage = try? MPImage(pixelBuffer: pixelBuffer),
              let result  = try? landmarker.detect(image: mpImage),
              let firstFace = result.faceLandmarks.first,
              !firstFace.isEmpty else { return nil }

        let xs = firstFace.map { CGFloat($0.x) }
        let ys = firstFace.map { CGFloat($0.y) }
        guard let minX = xs.min(), let maxX = xs.max(),
              let minY = ys.min(), let maxY = ys.max() else { return nil }

        let box = CGRect(x: minX, y: minY,
                         width: maxX - minX, height: maxY - minY)
        let cropped = crop(pixelBuffer: pixelBuffer, normalizedBox: box)

        let landmarks: [[CGPoint]] = result.faceLandmarks.map { face in
            face.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
        }
        return FaceLandmarkResult(croppedImage: cropped, landmarks: landmarks)
    }

    // MARK: - Private helpers

    /// Expand the normalised bounding box by `padding`, convert to CIImage
    /// pixel coordinates (bottom-left origin), clamp, crop, and reset the
    /// origin to (0, 0).
    private func crop(pixelBuffer: CVImageBuffer, normalizedBox box: CGRect) -> CIImage {
        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let imageBounds = CGRect(x: 0, y: 0, width: w, height: h)

        let padX = box.width  * padding
        let padY = box.height * padding

        let padded = CGRect(
            x: box.origin.x - padX,
            y: box.origin.y - padY,
            width:  box.width  + 2 * padX,
            height: box.height + 2 * padY)

        // MediaPipe uses top-left origin; CIImage uses bottom-left.
        let pixelRect = CGRect(
            x: padded.minX * w,
            y: (1.0 - padded.maxY) * h,
            width:  padded.width  * w,
            height: padded.height * h
        ).intersection(imageBounds)

        return CIImage(cvPixelBuffer: pixelBuffer)
            .cropped(to: pixelRect)
            .transformed(by: CGAffineTransform(
                translationX: -pixelRect.minX, y: -pixelRect.minY))
    }
}
