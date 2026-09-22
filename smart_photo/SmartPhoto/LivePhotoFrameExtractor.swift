import AVFoundation
import CoreImage
import Photos
import PhotosUI
import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// Pulls every video frame from a Live Photo so they can be treated as candidates.
enum LivePhotoFrameExtractor {

    private static let maxDimension: CGFloat = 720

    struct MovieFile: Transferable {
        let url: URL

        static var transferRepresentation: some TransferRepresentation {
            FileRepresentation(contentType: .movie) { movie in
                SentTransferredFile(movie.url)
            } importing: { received in
                let dest = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString)
                    .appendingPathExtension(received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension)
                try FileManager.default.copyItem(at: received.file, to: dest)
                return MovieFile(url: dest)
            }
        }
    }

    static func frames(from item: PhotosPickerItem) async throws -> [UIImage] {
        let url: URL
        if let movie = try? await item.loadTransferable(type: MovieFile.self) {
            url = movie.url
        } else if let paired = try await writePairedVideo(from: item) {
            url = paired
        } else {
            throw ExtractError.noVideo
        }

        defer { try? FileManager.default.removeItem(at: url) }
        return try await Task.detached(priority: .userInitiated) {
            try await decodeAllFrames(fromVideoURL: url)
        }.value
    }

    private static func writePairedVideo(from item: PhotosPickerItem) async throws -> URL? {
        guard let identifier = item.itemIdentifier else { return nil }

        let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        guard status == .authorized || status == .limited else { return nil }

        let assets = PHAsset.fetchAssets(withLocalIdentifiers: [identifier], options: nil)
        guard let asset = assets.firstObject else { return nil }

        let resources = PHAssetResource.assetResources(for: asset)
        guard let video = resources.first(where: { $0.type == .pairedVideo }) else { return nil }

        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mov")

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            PHAssetResourceManager.default().writeData(for: video, toFile: dest, options: nil) { error in
                if let error {
                    cont.resume(throwing: error)
                } else {
                    cont.resume()
                }
            }
        }
        return dest
    }

    nonisolated private static func decodeAllFrames(fromVideoURL url: URL) async throws -> [UIImage] {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ExtractError.noVideo
        }
        // preferredTransform is for a top-left / y-down space. Applying it
        // directly to CIImage (bottom-left / y-up) flips frames. Map it to
        // UIImage.Orientation instead and bake it in when drawing.
        let orientation = imageOrientation(from: try await track.load(.preferredTransform))

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
        )
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw ExtractError.noVideo }
        reader.add(output)
        guard reader.startReading() else { throw ExtractError.decodeFailed }

        let context = CIContext(options: [.useSoftwareRenderer: false])
        var images: [UIImage] = []

        while reader.status == .reading {
            guard let sample = output.copyNextSampleBuffer(),
                  let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { break }

            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let extent = ciImage.extent
            guard extent.width > 1, extent.height > 1,
                  let cgImage = context.createCGImage(ciImage, from: extent) else { continue }
            let oriented = UIImage(cgImage: cgImage, scale: 1, orientation: orientation)
            images.append(downsampled(oriented))
        }

        if images.isEmpty { throw ExtractError.decodeFailed }
        print("[LivePhoto] extracted \(images.count) frames")
        return images
    }

    nonisolated private static func downsampled(_ image: UIImage) -> UIImage {
        let size = image.size
        let longest = max(size.width, size.height)
        let scale = longest > maxDimension ? maxDimension / longest : 1
        let target = CGSize(width: (size.width * scale).rounded(), height: (size.height * scale).rounded())
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        // Drawing bakes UIImage.Orientation into upright pixels.
        return UIGraphicsImageRenderer(size: target, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: target))
        }
    }

    nonisolated private static func imageOrientation(from transform: CGAffineTransform) -> UIImage.Orientation {
        switch (transform.a, transform.b, transform.c, transform.d) {
        case (0, 1, -1, 0): return .right
        case (0, -1, 1, 0): return .left
        case (1, 0, 0, 1): return .up
        case (-1, 0, 0, -1): return .down
        case (1, 0, 0, -1): return .downMirrored
        case (-1, 0, 0, 1): return .upMirrored
        case (0, 1, 1, 0): return .rightMirrored
        case (0, -1, -1, 0): return .leftMirrored
        default: return .up
        }
    }

    enum ExtractError: LocalizedError {
        case noVideo
        case decodeFailed

        var errorDescription: String? {
            switch self {
            case .noVideo:
                return "Could not read the Live Photo video. Pick a Live Photo, not a still."
            case .decodeFailed:
                return "Could not decode Live Photo frames."
            }
        }
    }
}
