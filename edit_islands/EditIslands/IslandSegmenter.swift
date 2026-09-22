import Foundation

enum IslandSegmenter {
    static func segment(
        changeMask: [UInt8],
        width: Int,
        height: Int,
        minArea: Int = 64
    ) -> [MaskIsland] {
        let n = width * height
        precondition(changeMask.count == n)

        var labels = [Int](repeating: 0, count: n)
        var nextLabel = 1
        var areas: [Int: Int] = [:]

        // 4-connected flood fill (stable islands; less bleed than 8-connect on noisy diffs)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * width + x
                if changeMask[i] == 0 || labels[i] != 0 { continue }
                let label = nextLabel
                nextLabel += 1
                var area = 0
                var stack = [(x, y)]
                labels[i] = label
                while let (cx, cy) = stack.popLast() {
                    area += 1
                    let neighbors = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
                    for (nx, ny) in neighbors {
                        guard nx >= 0, ny >= 0, nx < width, ny < height else { continue }
                        let ni = ny * width + nx
                        if changeMask[ni] != 0 && labels[ni] == 0 {
                            labels[ni] = label
                            stack.append((nx, ny))
                        }
                    }
                }
                areas[label] = area
            }
        }

        let kept = areas
            .filter { $0.value >= minArea }
            .sorted { $0.value > $1.value }

        var islands: [MaskIsland] = []
        for (idx, entry) in kept.enumerated() {
            var mask = [UInt8](repeating: 0, count: n)
            var minX = width, minY = height, maxX = 0, maxY = 0
            for i in 0..<n where labels[i] == entry.key {
                mask[i] = 1
                let x = i % width
                let y = i / width
                if x < minX { minX = x }
                if x > maxX { maxX = x }
                if y < minY { minY = y }
                if y > maxY { maxY = y }
            }
            islands.append(
                MaskIsland(
                    label: "Island \(idx + 1)",
                    mask: mask,
                    width: width,
                    height: height,
                    area: entry.value,
                    bbox: (minX, minY, maxX, maxY),
                    isEnabled: true,
                    tint: IslandPalette.tint(for: idx)
                )
            )
        }
        return islands
    }

    /// Hit-test: return island index containing pixel, preferring smallest area (most specific).
    static func hitTest(islands: [MaskIsland], x: Int, y: Int) -> UUID? {
        var best: (UUID, Int)?
        for island in islands where island.isEnabled && island.contains(x: x, y: y) {
            if best == nil || island.area < best!.1 {
                best = (island.id, island.area)
            }
        }
        return best?.0
    }
}
