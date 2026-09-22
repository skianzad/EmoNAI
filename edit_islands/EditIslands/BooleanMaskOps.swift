import Foundation

enum BooleanMaskOps {
    static func apply(
        _ op: BooleanOp,
        a: [UInt8],
        b: [UInt8]
    ) -> [UInt8] {
        precondition(a.count == b.count)
        var out = [UInt8](repeating: 0, count: a.count)
        switch op {
        case .unite:
            for i in 0..<a.count { out[i] = (a[i] | b[i]) != 0 ? 1 : 0 }
        case .intersect:
            for i in 0..<a.count { out[i] = (a[i] & b[i]) != 0 ? 1 : 0 }
        case .subtract:
            for i in 0..<a.count { out[i] = (a[i] != 0 && b[i] == 0) ? 1 : 0 }
        case .exclude:
            for i in 0..<a.count { out[i] = (a[i] ^ b[i]) != 0 ? 1 : 0 }
        }
        return out
    }

    /// Fast circular stamp — no per-pixel area recount.
    static func erase(
        mask: inout [UInt8],
        width: Int,
        height: Int,
        atX x: Int,
        atY y: Int,
        radius: Int
    ) {
        let r = max(1, radius)
        let r2 = r * r
        let y0 = max(0, y - r)
        let y1 = min(height - 1, y + r)
        let x0 = max(0, x - r)
        let x1 = min(width - 1, x + r)
        mask.withUnsafeMutableBufferPointer { buf in
            guard let base = buf.baseAddress else { return }
            for yy in y0...y1 {
                let dy = yy - y
                let dy2 = dy * dy
                let row = base.advanced(by: yy * width)
                for xx in x0...x1 {
                    let dx = xx - x
                    if dx * dx + dy2 <= r2 {
                        row[xx] = 0
                    }
                }
            }
        }
    }

    static func area(of mask: [UInt8]) -> Int {
        var total = 0
        for v in mask where v != 0 { total += 1 }
        return total
    }
}
