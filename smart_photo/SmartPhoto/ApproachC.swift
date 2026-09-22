import CoreML
import UIKit
import Vision

/// Approach C: FECNet-style embedding — purpose-built for expression similarity (16-d).
///
/// Requires a converted Core ML model in the app bundle. Build it with:
///   `python smart_photo/model_export/export_fecnet.py`
/// then copy `FECNetEmbedding.mlpackage` into this target folder.
/// Note: the DenseNet-BC head ships randomly initialized until trained on FEC.
///
/// Input: 160×160 face crop ((x-127.5)/128 baked into the model).
/// Output: 16-d L2-normalised embedding.
/// If the model is not bundled, `isAvailable` returns false and encode() returns nil.
enum ApproachC {

    static var isAvailable: Bool {
        Bundle.main.url(forResource: "FECNetEmbedding", withExtension: "mlmodelc") != nil
    }

    private static let model: VNCoreMLModel? = {
        guard let url = Bundle.main.url(forResource: "FECNetEmbedding", withExtension: "mlmodelc"),
              let mlModel = try? MLModel(contentsOf: url),
              let vnModel = try? VNCoreMLModel(for: mlModel) else { return nil }
        return vnModel
    }()

    /// Encode a pre-cropped face CGImage (160×160) into a 16-d embedding.
    static func encode(faceImage: CGImage) -> [Float]? {
        guard let vnModel = model else { return nil }

        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cgImage: faceImage, options: [:])
        do { try handler.perform([request]) } catch { return nil }

        guard let result = request.results?.first as? VNCoreMLFeatureValueObservation,
              let multiArray = result.featureValue.multiArrayValue else { return nil }

        let count = multiArray.count
        let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }

    /// Encode a full UIImage: detect face → crop 160×160 → embed.
    static func encode(image: UIImage) -> [Float]? {
        guard let detection = FaceDetector.detect(
            in: image, targetSize: CGSize(width: 160, height: 160), align: true) else { return nil }
        return encode(faceImage: detection.croppedFace)
    }
}
