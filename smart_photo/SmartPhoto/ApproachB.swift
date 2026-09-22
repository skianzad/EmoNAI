import CoreML
import CoreVideo
import UIKit
import Vision

/// Approach B: POSTER V2 — learned expression embedding (768-d).
///
/// Bundled model: `PosterV2Embedding.mlpackage` (compiled to `.mlmodelc` at build).
/// Input: 224×224 RGB face crop. ImageNet mean/std is baked into the model
/// (ImageType scale=1/255 → internal normalize).
/// Output: 768-d L2-normalised SE-block vector (Float16 in the mlprogram).
enum ApproachB {

    private static let inputName = "input"
    private static let outputName = "embedding"

    private static let mlModel: MLModel? = {
        let config = MLModelConfiguration()
        config.computeUnits = .all

        // Prefer compiled .mlmodelc (Xcode build); fall back to .mlpackage in bundle.
        let candidates: [URL?] = [
            Bundle.main.url(forResource: "PosterV2Embedding", withExtension: "mlmodelc"),
            Bundle.main.url(forResource: "PosterV2Embedding", withExtension: "mlpackage"),
        ]
        for case let url? in candidates {
            do {
                let model = try MLModel(contentsOf: url, configuration: config)
                print("[ApproachB] loaded model from \(url.lastPathComponent)")
                return model
            } catch {
                print("[ApproachB] failed to load \(url.lastPathComponent): \(error)")
            }
        }
        print("[ApproachB] PosterV2Embedding not found in bundle")
        return nil
    }()

    static var isAvailable: Bool { mlModel != nil }

    /// Encode a pre-cropped face CGImage (any size; resized to 224×224) into a 768-d embedding.
    static func encode(faceImage: CGImage) -> [Float]? {
        guard let model = mlModel else { return nil }
        guard let pixelBuffer = makeRGBPixelBuffer(from: faceImage, width: 224, height: 224) else {
            print("[ApproachB] pixel buffer creation failed")
            return nil
        }

        do {
            let input = try MLDictionaryFeatureProvider(
                dictionary: [inputName: MLFeatureValue(pixelBuffer: pixelBuffer)]
            )
            let out = try model.prediction(from: input)
            guard let multiArray = out.featureValue(for: outputName)?.multiArrayValue else {
                let keys = out.featureNames
                print("[ApproachB] missing '\(outputName)' output; available: \(keys)")
                return nil
            }
            let emb = multiArrayToFloats(multiArray)
            if emb.count != 768 {
                print("[ApproachB] unexpected embedding length \(emb.count)")
            }
            let norm = sqrt(emb.reduce(0) { $0 + $1 * $1 })
            if norm < 1e-6 {
                print("[ApproachB] near-zero embedding (norm=\(norm)) — model output may be wrong")
            }
            return emb
        } catch {
            print("[ApproachB] prediction failed: \(error)")
            return nil
        }
    }

    /// Encode a full UIImage: detect face → crop 224×224 → embed.
    static func encode(image: UIImage) -> [Float]? {
        guard let detection = FaceDetector.detect(
            in: image, targetSize: CGSize(width: 224, height: 224), align: true) else {
            print("[ApproachB] no face detected")
            return nil
        }
        return encode(faceImage: detection.croppedFace)
    }

    // MARK: - Helpers

    /// Copy MLMultiArray to [Float], handling Float16 / Float32 / Double.
    private static func multiArrayToFloats(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        switch array.dataType {
        case .float16:
            let ptr = array.dataPointer.bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(ptr[$0]) }
        case .float32:
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        case .double:
            let ptr = array.dataPointer.bindMemory(to: Double.self, capacity: count)
            return (0..<count).map { Float(ptr[$0]) }
        default:
            return (0..<count).map { Float(truncating: array[$0]) }
        }
    }

    /// Render a CGImage into a 224×224 BGRA CVPixelBuffer for Core ML ImageType input.
    private static func makeRGBPixelBuffer(
        from image: CGImage, width: Int, height: Int
    ) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}
