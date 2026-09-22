import CoreImage
import UIKit
import Vision

/// Detects a face in a UIImage and returns a square crop plus the Vision observation.
enum FaceDetector {

    struct Detection {
        let croppedFace: CGImage
        let observation: VNFaceObservation
        let originalSize: CGSize
    }

    /// Detect the largest face and return a padded square crop resized to `targetSize`.
    ///
    /// - Parameter align: If true, rotate the crop by −roll so the eyes sit on a
    ///   horizontal line (what POSTER V2 / FECNet were trained on). Approach A
    ///   should pass `false` so yaw / pitch / roll stay in the photo's frame.
    static func detect(
        in image: UIImage,
        targetSize: CGSize = CGSize(width: 224, height: 224),
        padding: CGFloat = 0.3,
        align: Bool = true
    ) -> Detection? {
        guard let cgImage = uprightCGImage(from: image) else { return nil }

        let request = VNDetectFaceLandmarksRequest()
        request.revision = VNDetectFaceLandmarksRequestRevision3
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
        do { try handler.perform([request]) } catch { return nil }

        let faces = request.results ?? []
        guard let observation = faces.max(by: {
            ($0.boundingBox.width * $0.boundingBox.height)
                < ($1.boundingBox.width * $1.boundingBox.height)
        }) else { return nil }

        let imgW = CGFloat(cgImage.width)
        let imgH = CGFloat(cgImage.height)
        let pixelRect = squarePixelRect(
            visionBox: observation.boundingBox,
            imageSize: CGSize(width: imgW, height: imgH),
            padding: padding)

        guard pixelRect.width > 1, pixelRect.height > 1,
              let cropped = cgImage.cropping(to: pixelRect) else { return nil }

        let roll = align ? CGFloat(observation.roll?.doubleValue ?? 0) : 0
        guard let finalCG = render(cropped, to: targetSize, roll: roll) else { return nil }
        return Detection(
            croppedFace: finalCG,
            observation: observation,
            originalSize: CGSize(width: imgW, height: imgH))
    }

    /// Square face crop for thumbnails. Unaligned so the preview matches the photo.
    static func faceImage(from image: UIImage, size: CGSize = CGSize(width: 224, height: 224)) -> UIImage? {
        guard let detection = detect(in: image, targetSize: size, align: false) else { return nil }
        return UIImage(cgImage: detection.croppedFace)
    }

    /// Bake `UIImage.imageOrientation` into pixel data so Vision and cropping agree.
    static func uprightCGImage(from image: UIImage) -> CGImage? {
        if image.imageOrientation == .up, let cg = image.cgImage {
            return cg
        }
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let size = image.size
        guard size.width > 1, size.height > 1 else { return image.cgImage }
        return UIGraphicsImageRenderer(size: size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }.cgImage
    }

    /// Vision box is normalised, origin bottom-left. Convert to an in-bounds
    /// square pixel rect with origin top-left. Stretching a non-square box to
    /// 224×224 was distorting faces and poisoning embeddings.
    private static func squarePixelRect(
        visionBox box: CGRect,
        imageSize: CGSize,
        padding: CGFloat
    ) -> CGRect {
        let imgW = imageSize.width
        let imgH = imageSize.height
        var rect = CGRect(
            x: box.minX * imgW,
            y: (1 - box.maxY) * imgH,
            width: box.width * imgW,
            height: box.height * imgH)

        let pad = padding * max(rect.width, rect.height)
        rect = rect.insetBy(dx: -pad, dy: -pad)

        let side = max(rect.width, rect.height)
        var square = CGRect(
            x: rect.midX - side / 2,
            y: rect.midY - side / 2,
            width: side,
            height: side)

        if square.minX < 0 { square.origin.x = 0 }
        if square.minY < 0 { square.origin.y = 0 }
        if square.maxX > imgW { square.origin.x = max(0, imgW - square.width) }
        if square.maxY > imgH { square.origin.y = max(0, imgH - square.height) }

        return square.intersection(CGRect(origin: .zero, size: imageSize)).integral
    }

    private static func render(_ cropped: CGImage, to targetSize: CGSize, roll: CGFloat) -> CGImage? {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        let image = renderer.image { ctx in
            let c = ctx.cgContext
            if roll != 0 {
                c.translateBy(x: targetSize.width / 2, y: targetSize.height / 2)
                // Vision roll is radians, positive = counterclockwise in image space.
                c.rotate(by: -roll)
                c.translateBy(x: -targetSize.width / 2, y: -targetSize.height / 2)
            }
            UIImage(cgImage: cropped).draw(in: CGRect(origin: .zero, size: targetSize))
        }
        return image.cgImage
    }
}
