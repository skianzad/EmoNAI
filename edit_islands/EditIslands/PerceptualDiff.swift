import Foundation

/// Local-SSIM change mask. Avoids raw RGB subtraction (VAE-noise prone).
enum PerceptualDiff {
    struct Options {
        var windowRadius: Int = 5
        /// Pixels with SSIM below this are treated as changed.
        var ssimThreshold: Float = 0.85
        var morphRadius: Int = 1
        var minIslandArea: Int = 64
    }

    struct Result {
        let width: Int
        let height: Int
        /// 1 = changed
        let changeMask: [UInt8]
        let ssimMap: [Float]
        let changeFraction: Float
    }

    static func changeMask(
        original: ImageBuffer,
        edited: ImageBuffer,
        options: Options = Options()
    ) -> Result? {
        guard let (aBuf, bBuf) = ImageBuffer.matchSize(original, edited) else { return nil }
        let w = aBuf.width
        let h = aBuf.height
        let a = aBuf.luminance()
        let b = bBuf.luminance()

        let r = max(1, options.windowRadius)
        let meanA = boxBlur(a, width: w, height: h, radius: r)
        let meanB = boxBlur(b, width: w, height: h, radius: r)
        let meanAA = boxBlur(a.map { $0 * $0 }, width: w, height: h, radius: r)
        let meanBB = boxBlur(b.map { $0 * $0 }, width: w, height: h, radius: r)
        let meanAB = boxBlur(zip(a, b).map(*), width: w, height: h, radius: r)

        // Stabilizers for 8-bit luminance in [0,1]
        let c1: Float = 0.01 * 0.01
        let c2: Float = 0.03 * 0.03

        var ssim = [Float](repeating: 0, count: w * h)
        var mask = [UInt8](repeating: 0, count: w * h)
        var changed = 0

        for i in 0..<(w * h) {
            let muA = meanA[i]
            let muB = meanB[i]
            let sigmaA2 = max(0, meanAA[i] - muA * muA)
            let sigmaB2 = max(0, meanBB[i] - muB * muB)
            let sigmaAB = meanAB[i] - muA * muB

            let num = (2 * muA * muB + c1) * (2 * sigmaAB + c2)
            let den = (muA * muA + muB * muB + c1) * (sigmaA2 + sigmaB2 + c2)
            let value = den > 0 ? num / den : 1
            ssim[i] = value
            if value < options.ssimThreshold {
                mask[i] = 1
                changed += 1
            }
        }

        if options.morphRadius > 0 {
            mask = morphOpenClose(mask, width: w, height: h, radius: options.morphRadius)
            changed = mask.reduce(0) { $0 + Int($1) }
        }

        return Result(
            width: w,
            height: h,
            changeMask: mask,
            ssimMap: ssim,
            changeFraction: Float(changed) / Float(max(1, w * h))
        )
    }

    // MARK: - Filters

    private static func boxBlur(_ src: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        let tmp = horizontalBoxBlur(src, width: width, height: height, radius: radius)
        return verticalBoxBlur(tmp, width: width, height: height, radius: radius)
    }

    private static func horizontalBoxBlur(_ src: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        var out = [Float](repeating: 0, count: width * height)
        let window = Float(radius * 2 + 1)
        for y in 0..<height {
            let row = y * width
            var sum: Float = 0
            for x in -radius...radius {
                let xx = min(max(x, 0), width - 1)
                sum += src[row + xx]
            }
            out[row] = sum / window
            for x in 1..<width {
                let add = min(x + radius, width - 1)
                let rem = max(x - radius - 1, 0)
                sum += src[row + add] - src[row + rem]
                out[row + x] = sum / window
            }
        }
        return out
    }

    private static func verticalBoxBlur(_ src: [Float], width: Int, height: Int, radius: Int) -> [Float] {
        var out = [Float](repeating: 0, count: width * height)
        let window = Float(radius * 2 + 1)
        for x in 0..<width {
            var sum: Float = 0
            for y in -radius...radius {
                let yy = min(max(y, 0), height - 1)
                sum += src[yy * width + x]
            }
            out[x] = sum / window
            for y in 1..<height {
                let add = min(y + radius, height - 1)
                let rem = max(y - radius - 1, 0)
                sum += src[add * width + x] - src[rem * width + x]
                out[y * width + x] = sum / window
            }
        }
        return out
    }

    private static func morphOpenClose(_ mask: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        let eroded = erode(mask, width: width, height: height, radius: radius)
        let opened = dilate(eroded, width: width, height: height, radius: radius)
        let dilated = dilate(opened, width: width, height: height, radius: radius)
        return erode(dilated, width: width, height: height, radius: radius)
    }

    private static func erode(_ mask: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                var keep: UInt8 = 1
                outer: for dy in -radius...radius {
                    for dx in -radius...radius {
                        let xx = x + dx
                        let yy = y + dy
                        if xx < 0 || yy < 0 || xx >= width || yy >= height || mask[yy * width + xx] == 0 {
                            keep = 0
                            break outer
                        }
                    }
                }
                out[y * width + x] = keep
            }
        }
        return out
    }

    private static func dilate(_ mask: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                var hit: UInt8 = 0
                outer: for dy in -radius...radius {
                    for dx in -radius...radius {
                        let xx = x + dx
                        let yy = y + dy
                        if xx >= 0, yy >= 0, xx < width, yy < height, mask[yy * width + xx] != 0 {
                            hit = 1
                            break outer
                        }
                    }
                }
                out[y * width + x] = hit
            }
        }
        return out
    }
}
