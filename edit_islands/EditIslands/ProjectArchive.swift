import Foundation
import SwiftUI
import UniformTypeIdentifiers

enum ProjectArchiveError: LocalizedError {
    case notAProject
    case corrupt(String)

    var errorDescription: String? {
        switch self {
        case .notAProject:
            return "That file is not an EditIslands project."
        case .corrupt(let detail):
            return "Project file is incomplete (\(detail))."
        }
    }
}

/// One file containing the original, every version, prompts, masks, and subprompts.
enum ProjectArchive {
    static let magic = Data("EISL".utf8)
    static let formatVersion: UInt32 = 1

    struct Snapshot {
        var original: ImageBuffer
        var layers: [EditLayer]
        var selectedLayerID: UUID?
        var selectedIslandIDs: Set<UUID>
        var ssimThreshold: Double
        var minIslandArea: Int
    }

    static func encode(_ snapshot: Snapshot) throws -> Data {
        var blobs: [String: Data] = [:]
        let originalName = "original"
        blobs[originalName] = Data(snapshot.original.rgba)

        let layers = snapshot.layers.map { layer -> LayerRecord in
            let pixelsName = "pixels-\(layer.id.uuidString)"
            blobs[pixelsName] = Data(layer.editedPixels)
            let islands = layer.islands.map { island -> IslandRecord in
                let maskName = "mask-\(island.id.uuidString)"
                blobs[maskName] = Data(island.mask)
                return IslandRecord(
                    id: island.id,
                    label: island.label,
                    area: island.area,
                    isEnabled: island.isEnabled,
                    tint: [island.tint.x, island.tint.y, island.tint.z],
                    width: island.width,
                    height: island.height,
                    maskBlob: maskName
                )
            }
            return LayerRecord(
                id: layer.id,
                prompt: layer.prompt,
                provider: layer.provider?.rawValue,
                parentID: layer.parentID,
                isVisible: layer.isVisible,
                width: layer.width,
                height: layer.height,
                pixelsBlob: pixelsName,
                islands: islands
            )
        }

        let manifest = Manifest(
            original: ImageRecord(
                width: snapshot.original.width,
                height: snapshot.original.height,
                blob: originalName
            ),
            layers: layers,
            selectedLayerID: snapshot.selectedLayerID,
            selectedIslandIDs: Array(snapshot.selectedIslandIDs),
            ssimThreshold: snapshot.ssimThreshold,
            minIslandArea: snapshot.minIslandArea
        )
        let json = try JSONEncoder().encode(manifest)

        var writer = Writer()
        writer.data.append(magic)
        writer.u32(formatVersion)
        writer.bytes(json)
        let names = blobs.keys.sorted()
        writer.u32(UInt32(names.count))
        for name in names {
            writer.str(name)
            writer.bytes(blobs[name] ?? Data())
        }
        return writer.data
    }

    static func decode(_ data: Data) throws -> Snapshot {
        var reader = Reader(data: data)
        let magic = try reader.take(4)
        guard magic == self.magic else { throw ProjectArchiveError.notAProject }
        let version = try reader.u32()
        guard version == formatVersion else {
            throw ProjectArchiveError.corrupt("unknown version \(version)")
        }
        let json = try reader.bytes()
        let manifest = try JSONDecoder().decode(Manifest.self, from: json)
        let blobCount = try reader.u32()
        var blobs: [String: Data] = [:]
        for _ in 0..<blobCount {
            let name = try reader.str()
            blobs[name] = try reader.bytes()
        }

        let originalPixels = try pixels(
            named: manifest.original.blob,
            width: manifest.original.width,
            height: manifest.original.height,
            blobs: blobs
        )
        let original = ImageBuffer(
            width: manifest.original.width,
            height: manifest.original.height,
            rgba: originalPixels
        )

        let layers: [EditLayer] = try manifest.layers.map { record in
            let pixels = try pixels(
                named: record.pixelsBlob,
                width: record.width,
                height: record.height,
                blobs: blobs
            )
            let islands: [MaskIsland] = try record.islands.map { island in
                guard let maskData = blobs[island.maskBlob] else {
                    throw ProjectArchiveError.corrupt("missing \(island.maskBlob)")
                }
                let mask = [UInt8](maskData)
                guard mask.count == island.width * island.height else {
                    throw ProjectArchiveError.corrupt("mask size for \(island.label)")
                }
                let tint = island.tint.count == 3
                    ? SIMD3<Float>(island.tint[0], island.tint[1], island.tint[2])
                    : SIMD3<Float>(1, 0, 0)
                return MaskIsland(
                    id: island.id,
                    label: island.label,
                    mask: mask,
                    width: island.width,
                    height: island.height,
                    area: island.area,
                    isEnabled: island.isEnabled,
                    tint: tint
                )
            }
            let provider = record.provider.flatMap(EditProvider.init(rawValue:))
            return EditLayer(
                id: record.id,
                prompt: record.prompt,
                provider: provider,
                editedPixels: pixels,
                width: record.width,
                height: record.height,
                islands: islands,
                isVisible: record.isVisible,
                parentID: record.parentID
            )
        }

        return Snapshot(
            original: original,
            layers: layers,
            selectedLayerID: manifest.selectedLayerID,
            selectedIslandIDs: Set(manifest.selectedIslandIDs),
            ssimThreshold: manifest.ssimThreshold,
            minIslandArea: manifest.minIslandArea
        )
    }

    private static func pixels(
        named name: String,
        width: Int,
        height: Int,
        blobs: [String: Data]
    ) throws -> [UInt8] {
        guard let data = blobs[name] else {
            throw ProjectArchiveError.corrupt("missing \(name)")
        }
        let pixels = [UInt8](data)
        guard width > 0, height > 0, pixels.count == width * height * 4 else {
            throw ProjectArchiveError.corrupt("image size for \(name)")
        }
        return pixels
    }

    private struct Manifest: Codable {
        var original: ImageRecord
        var layers: [LayerRecord]
        var selectedLayerID: UUID?
        var selectedIslandIDs: [UUID]
        var ssimThreshold: Double
        var minIslandArea: Int
    }

    private struct ImageRecord: Codable {
        var width: Int
        var height: Int
        var blob: String
    }

    private struct LayerRecord: Codable {
        var id: UUID
        var prompt: String
        var provider: String?
        var parentID: UUID?
        var isVisible: Bool
        var width: Int
        var height: Int
        var pixelsBlob: String
        var islands: [IslandRecord]
    }

    private struct IslandRecord: Codable {
        var id: UUID
        var label: String
        var area: Int
        var isEnabled: Bool
        var tint: [Float]
        var width: Int
        var height: Int
        var maskBlob: String
    }

    private struct Writer {
        var data = Data()

        mutating func u32(_ value: UInt32) {
            var le = value.littleEndian
            withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
        }

        mutating func bytes(_ payload: Data) {
            u32(UInt32(payload.count))
            data.append(payload)
        }

        mutating func str(_ value: String) {
            bytes(Data(value.utf8))
        }
    }

    private struct Reader {
        let data: Data
        var offset = 0

        mutating func take(_ count: Int) throws -> Data {
            guard offset + count <= data.count else {
                throw ProjectArchiveError.corrupt("truncated")
            }
            let slice = data.subdata(in: offset..<(offset + count))
            offset += count
            return slice
        }

        mutating func u32() throws -> UInt32 {
            let raw = [UInt8](try take(4))
            return UInt32(raw[0])
                | (UInt32(raw[1]) << 8)
                | (UInt32(raw[2]) << 16)
                | (UInt32(raw[3]) << 24)
        }

        mutating func bytes() throws -> Data {
            let count = Int(try u32())
            return try take(count)
        }

        mutating func str() throws -> String {
            let raw = try bytes()
            guard let text = String(data: raw, encoding: .utf8) else {
                throw ProjectArchiveError.corrupt("bad name")
            }
            return text
        }
    }
}

struct EditIslandsProjectDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.data] }

    var data: Data

    init(data: Data) {
        self.data = data
    }

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw ProjectArchiveError.notAProject
        }
        self.data = data
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}
