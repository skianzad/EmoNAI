import CoreGraphics
import Foundation
import UIKit

enum EditorTool: String, CaseIterable, Identifiable {
    case hand
    case select
    case eraser

    var id: String { rawValue }

    var title: String {
        switch self {
        case .hand: return "Hand"
        case .select: return "Cursor"
        case .eraser: return "Eraser"
        }
    }

    var systemImage: String {
        switch self {
        case .hand: return "hand.draw"
        case .select: return "cursorarrow"
        case .eraser: return "eraser"
        }
    }
}

enum BooleanOp: String, CaseIterable, Identifiable {
    case unite
    case intersect
    case subtract
    case exclude

    var id: String { rawValue }

    var title: String {
        switch self {
        case .unite: return "Unite"
        case .intersect: return "Intersect"
        case .subtract: return "Subtract"
        case .exclude: return "Exclude"
        }
    }

    var systemImage: String {
        switch self {
        case .unite: return "plus.square.fill.on.square.fill"
        case .intersect: return "square.on.circle"
        case .subtract: return "minus.square"
        case .exclude: return "circle.lefthalf.filled"
        }
    }
}

struct MaskIsland: Identifiable, Hashable {
    let id: UUID
    var label: String
    /// Flat row-major mask; 1 = island pixel. Same size as working canvas.
    var mask: [UInt8]
    var width: Int
    var height: Int
    var area: Int
    /// Inclusive pixel bounds for marquee hit-testing (minX, minY, maxX, maxY).
    var bbox: (minX: Int, minY: Int, maxX: Int, maxY: Int)
    var isEnabled: Bool
    var tint: SIMD3<Float>

    init(
        id: UUID = UUID(),
        label: String,
        mask: [UInt8],
        width: Int,
        height: Int,
        area: Int,
        bbox: (minX: Int, minY: Int, maxX: Int, maxY: Int)? = nil,
        isEnabled: Bool = true,
        tint: SIMD3<Float>
    ) {
        self.id = id
        self.label = label
        self.mask = mask
        self.width = width
        self.height = height
        self.area = area
        self.bbox = bbox ?? MaskIsland.computeBBox(mask: mask, width: width, height: height)
        self.isEnabled = isEnabled
        self.tint = tint
    }

    func contains(x: Int, y: Int) -> Bool {
        guard x >= 0, y >= 0, x < width, y < height else { return false }
        return mask[y * width + x] != 0
    }

    func intersects(rect: CGRect) -> Bool {
        let r = rect.standardized
        let islandRect = CGRect(
            x: CGFloat(bbox.minX),
            y: CGFloat(bbox.minY),
            width: CGFloat(max(1, bbox.maxX - bbox.minX + 1)),
            height: CGFloat(max(1, bbox.maxY - bbox.minY + 1))
        )
        return islandRect.intersects(r)
    }

    /// True when any mask pixel lies inside the rectangle. Ignores empty bbox padding.
    func containsAnyPixel(in rect: CGRect) -> Bool {
        let r = rect.standardized
        let minX = max(bbox.minX, max(0, Int(floor(r.minX))))
        let maxX = min(bbox.maxX, min(width - 1, Int(ceil(r.maxX))))
        let minY = max(bbox.minY, max(0, Int(floor(r.minY))))
        let maxY = min(bbox.maxY, min(height - 1, Int(ceil(r.maxY))))
        guard minX <= maxX, minY <= maxY, mask.count >= width * height else { return false }
        for y in minY...maxY {
            let row = y * width
            for x in minX...maxX where mask[row + x] != 0 {
                return true
            }
        }
        return false
    }

    static func computeBBox(mask: [UInt8], width: Int, height: Int) -> (minX: Int, minY: Int, maxX: Int, maxY: Int) {
        var minX = width, minY = height, maxX = 0, maxY = 0
        var any = false
        for y in 0..<height {
            let row = y * width
            for x in 0..<width where mask[row + x] != 0 {
                any = true
                if x < minX { minX = x }
                if x > maxX { maxX = x }
                if y < minY { minY = y }
                if y > maxY { maxY = y }
            }
        }
        return any ? (minX, minY, maxX, maxY) : (0, 0, 0, 0)
    }

    // Hashable/Equatable ignore mask contents for Set membership by id.
    static func == (lhs: MaskIsland, rhs: MaskIsland) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

struct EditLayer: Identifiable {
    let id: UUID
    var prompt: String
    var provider: EditProvider?
    /// Edited RGB pixels at working resolution (RGBA8888).
    var editedPixels: [UInt8]
    var width: Int
    var height: Int
    var islands: [MaskIsland]
    var isVisible: Bool
    /// Set when this version is a follow-up edit of another version.
    var parentID: UUID?
    /// Combined enabled-island mask for this layer (updated when islands change).
    var combinedMask: [UInt8]

    var displayTitle: String {
        if let provider {
            return "[\(provider.shortName)] \(prompt)"
        }
        return prompt
    }

    init(
        id: UUID = UUID(),
        prompt: String,
        provider: EditProvider? = nil,
        editedPixels: [UInt8],
        width: Int,
        height: Int,
        islands: [MaskIsland],
        isVisible: Bool = true,
        parentID: UUID? = nil
    ) {
        self.id = id
        self.prompt = prompt
        self.provider = provider
        self.editedPixels = editedPixels
        self.width = width
        self.height = height
        self.islands = islands
        self.isVisible = isVisible
        self.parentID = parentID
        self.combinedMask = EditLayer.rebuildCombinedMask(islands: islands, width: width, height: height)
    }

    mutating func refreshCombinedMask() {
        combinedMask = EditLayer.rebuildCombinedMask(islands: islands, width: width, height: height)
    }

    static func rebuildCombinedMask(islands: [MaskIsland], width: Int, height: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height)
        for island in islands where island.isEnabled {
            for i in 0..<out.count where island.mask[i] != 0 {
                out[i] = 1
            }
        }
        return out
    }
}

enum IslandPalette {
    static func tint(for index: Int) -> SIMD3<Float> {
        let hue = Float((index * 67) % 360) / 360
        return hsvToRGB(h: hue, s: 0.85, v: 1)
    }

    private static func hsvToRGB(h: Float, s: Float, v: Float) -> SIMD3<Float> {
        let i = Int(h * 6)
        let f = h * 6 - Float(i)
        let p = v * (1 - s)
        let q = v * (1 - f * s)
        let t = v * (1 - (1 - f) * s)
        switch i % 6 {
        case 0: return SIMD3(v, t, p)
        case 1: return SIMD3(q, v, p)
        case 2: return SIMD3(p, v, t)
        case 3: return SIMD3(p, q, v)
        case 4: return SIMD3(t, p, v)
        default: return SIMD3(v, p, q)
        }
    }

    static func uiColor(_ tint: SIMD3<Float>, alpha: CGFloat = 0.45) -> UIColor {
        UIColor(red: CGFloat(tint.x), green: CGFloat(tint.y), blue: CGFloat(tint.z), alpha: alpha)
    }
}
