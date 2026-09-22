import CoreGraphics
import UIKit

enum PixelFormat {
    static let bytesPerPixel = 4
}

struct ImageBuffer {
    let width: Int
    let height: Int
    /// RGBA8888, row-major.
    var rgba: [UInt8]

    var pixelCount: Int { width * height }

    static func from(_ image: UIImage, maxSide: CGFloat = 1024) -> ImageBuffer? {
        guard let cg = image.cgImage else { return nil }
        let srcW = CGFloat(cg.width)
        let srcH = CGFloat(cg.height)
        let scale = min(1, maxSide / max(srcW, srcH))
        let width = max(1, Int((srcW * scale).rounded()))
        let height = max(1, Int((srcH * scale).rounded()))

        var rgba = [UInt8](repeating: 0, count: width * height * PixelFormat.bytesPerPixel)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * PixelFormat.bytesPerPixel,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        return ImageBuffer(width: width, height: height, rgba: rgba)
    }

    /// White = region that may change, black = keep. Same size as the working canvas.
    static func visibleMaskImage(mask: [UInt8], width: Int, height: Int) -> UIImage? {
        guard width > 0, height > 0, mask.count == width * height else { return nil }
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<mask.count {
            let on: UInt8 = mask[i] != 0 ? 255 : 0
            let o = i * 4
            rgba[o] = on
            rgba[o + 1] = on
            rgba[o + 2] = on
            rgba[o + 3] = 255
        }
        return ImageBuffer(width: width, height: height, rgba: rgba).toUIImage()
    }

    func toUIImage() -> UIImage? {
        guard width > 0, height > 0, rgba.count >= width * height * PixelFormat.bytesPerPixel else {
            return nil
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixels = rgba
        guard let ctx = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * PixelFormat.bytesPerPixel,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let cg = ctx.makeImage() else { return nil }
        return UIImage(cgImage: cg)
    }

    func luminance() -> [Float] {
        var out = [Float](repeating: 0, count: pixelCount)
        rgba.withUnsafeBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            for i in 0..<pixelCount {
                let o = i * 4
                let r = Float(base[o])
                let g = Float(base[o + 1])
                let b = Float(base[o + 2])
                out[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
            }
        }
        return out
    }

    static func matchSize(_ a: ImageBuffer, _ b: ImageBuffer) -> (ImageBuffer, ImageBuffer)? {
        if a.width == b.width && a.height == b.height { return (a, b) }
        let w = min(a.width, b.width)
        let h = min(a.height, b.height)
        guard let ai = a.toUIImage(), let bi = b.toUIImage() else { return nil }
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: w, height: h))
        let ar = renderer.image { _ in ai.draw(in: CGRect(x: 0, y: 0, width: w, height: h)) }
        let br = renderer.image { _ in bi.draw(in: CGRect(x: 0, y: 0, width: w, height: h)) }
        guard let ab = ImageBuffer.from(ar, maxSide: CGFloat(max(w, h))),
              let bb = ImageBuffer.from(br, maxSide: CGFloat(max(w, h)))
        else { return nil }
        return (ab, bb)
    }
}
