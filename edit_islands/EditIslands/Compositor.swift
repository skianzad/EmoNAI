import UIKit

enum Compositor {
    /// Final = original, then for each visible layer paint enabled-island pixels from that layer's edit.
    static func composite(
        original: ImageBuffer,
        layers: [EditLayer]
    ) -> ImageBuffer {
        var out = original.rgba
        let w = original.width
        let h = original.height
        guard w > 0, h > 0 else {
            return ImageBuffer(width: max(w, 1), height: max(h, 1), rgba: [0, 0, 0, 255])
        }
        let n = w * h

        for layer in layers where layer.isVisible {
            precondition(layer.width == w && layer.height == h)
            precondition(layer.editedPixels.count == out.count)
            let edited = layer.editedPixels
            for i in 0..<n where layer.combinedMask[i] != 0 {
                let o = i * 4
                out[o] = edited[o]
                out[o + 1] = edited[o + 1]
                out[o + 2] = edited[o + 2]
                out[o + 3] = 255
            }
        }
        return ImageBuffer(width: w, height: h, rgba: out)
    }

    /// Tint overlay for selection UI. Only draws largest islands + any currently selected.
    static func islandOverlay(
        size: (width: Int, height: Int),
        layers: [EditLayer],
        selectedIslandIDs: Set<UUID>,
        maxIslandsPerLayer: Int = 8,
        overlayAlpha: UInt8 = 90
    ) -> ImageBuffer {
        let w = size.width
        let h = size.height
        guard w > 0, h > 0 else {
            return ImageBuffer(width: 1, height: 1, rgba: [0, 0, 0, 0])
        }
        let n = w * h
        var rgba = [UInt8](repeating: 0, count: n * 4)

        for layer in layers where layer.isVisible {
            let ranked = layer.islands
                .filter(\.isEnabled)
                .sorted { $0.area > $1.area }
            var drawSet = Array(ranked.prefix(maxIslandsPerLayer))
            for island in layer.islands where selectedIslandIDs.contains(island.id) && island.isEnabled {
                if !drawSet.contains(where: { $0.id == island.id }) {
                    drawSet.append(island)
                }
            }

            for island in drawSet {
                let selected = selectedIslandIDs.contains(island.id)
                let a: UInt8 = selected ? 200 : overlayAlpha
                let r = UInt8(min(255, Int(island.tint.x * 255)))
                let g = UInt8(min(255, Int(island.tint.y * 255)))
                let b = UInt8(min(255, Int(island.tint.z * 255)))
                island.mask.withUnsafeBufferPointer { maskBuf in
                    guard let maskBase = maskBuf.baseAddress else { return }
                    rgba.withUnsafeMutableBufferPointer { outBuf in
                        guard let outBase = outBuf.baseAddress else { return }
                        for i in 0..<n where maskBase[i] != 0 {
                            let o = i * 4
                            if selected || outBase[o + 3] == 0 {
                                outBase[o] = r
                                outBase[o + 1] = g
                                outBase[o + 2] = b
                                outBase[o + 3] = a
                            }
                        }
                    }
                }
            }
        }
        return ImageBuffer(width: w, height: h, rgba: rgba)
    }
}
