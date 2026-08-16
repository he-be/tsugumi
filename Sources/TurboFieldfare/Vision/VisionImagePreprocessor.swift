import CoreGraphics
import Foundation
import ImageIO

/// One image, resized and flattened into the tower's input layout.
public struct VisionPreprocessedImage: Sendable, Equatable {
    public let geometry: VisionImageGeometry
    /// `[patchCount, 3 * patchSize^2]`, row-major over patches. Within a patch
    /// the element order is `(py, px, channel)` — upstream's
    /// `transpose(1, 3, 2, 4, 0)` in `convert_image_to_patches`.
    public let patches: [Float16]

    public init(geometry: VisionImageGeometry, patches: [Float16]) {
        self.geometry = geometry
        self.patches = patches
    }
}

/// Image decode, resize, and patchify — the whole CPU side of the vision path.
///
/// Deliberately dependency-free: ImageIO decodes, CoreGraphics resizes. There
/// is no attempt to match torchvision's `antialias=True` bicubic bit for bit,
/// and none is possible — upstream itself ships two implementations (PIL and
/// torchvision) that do not agree with each other. PLAN_VISION §6-1 handles
/// this by measuring the tower with upstream's own `pixel_values` (layer B) and
/// reporting the resize difference separately (layer C), so a resize mismatch
/// can never be confused with a kernel bug.
public enum VisionImagePreprocessor {

    public static func preprocess(contentsOf url: URL,
                                  config: VisionPreprocessorConfig) throws -> VisionPreprocessedImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw VisionError.imageDecodeFailed(path: url.path, detail: "unreadable or not an image")
        }
        guard CGImageSourceGetCount(source) > 0,
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw VisionError.imageDecodeFailed(path: url.path, detail: "no decodable frame")
        }
        return try preprocess(image: image, config: config)
    }

    public static func preprocess(data: Data,
                                  config: VisionPreprocessorConfig) throws -> VisionPreprocessedImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            throw VisionError.imageDecodeFailed(path: "<data>", detail: "unrecognised container")
        }
        guard CGImageSourceGetCount(source) > 0,
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw VisionError.imageDecodeFailed(path: "<data>", detail: "no decodable frame")
        }
        return try preprocess(image: image, config: config)
    }

    public static func preprocess(image: CGImage,
                                  config: VisionPreprocessorConfig) throws -> VisionPreprocessedImage {
        let geometry = try config.geometry(imageWidth: image.width, imageHeight: image.height)
        let rgb = try renderRGB8(image: image,
                                 width: geometry.targetWidth,
                                 height: geometry.targetHeight)
        let patches = patchify(rgb8: rgb, geometry: geometry, patchSize: config.patchSize)
        return VisionPreprocessedImage(geometry: geometry, patches: patches)
    }

    /// Decode into tightly packed 8-bit sRGB RGB at the target size.
    ///
    /// `noneSkipLast` rather than an alpha-carrying layout: the tower has no
    /// notion of transparency, and a premultiplied layout would silently scale
    /// the colour channels of any PNG with an alpha channel. Transparent pixels
    /// composite against the black the context is cleared to, which is what
    /// `do_convert_rgb` does upstream.
    static func renderRGB8(image: CGImage, width: Int, height: Int) throws -> [UInt8] {
        let bytesPerRow = width * 4
        var buffer = [UInt8](repeating: 0, count: bytesPerRow * height)
        let space = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

        let ok: Bool = buffer.withUnsafeMutableBytes { raw -> Bool in
            guard let context = CGContext(
                data: raw.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            ) else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard ok else {
            throw VisionError.imageDecodeFailed(path: "<cgimage>", detail: "could not create \(width)x\(height) context")
        }

        // Drop the skipped alpha byte so the patchifier walks a 3-byte stride.
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            rgb[i * 3 + 0] = buffer[i * 4 + 0]
            rgb[i * 3 + 1] = buffer[i * 4 + 1]
            rgb[i * 3 + 2] = buffer[i * 4 + 2]
        }
        return rgb
    }

    /// `[H, W, 3]` bytes to `[patchCount, patchSize^2 * 3]` in `[0, 1]`.
    ///
    /// The `1/255` rescale lives here rather than in a separate pass because
    /// `do_normalize` is false for this checkpoint (mean 0, std 1): dividing by
    /// 255 is the *entire* normalisation the preprocessor does. The `[0,1]` to
    /// `[-1,1]` mapping happens in the patch embedder, in the model
    /// (PLAN_VISION §1-3, §2-2).
    static func patchify(rgb8: [UInt8],
                         geometry: VisionImageGeometry,
                         patchSize: Int) -> [Float16] {
        let width = geometry.targetWidth
        let elementsPerPatch = patchSize * patchSize * 3
        var out = [Float16](repeating: 0, count: geometry.patchCount * elementsPerPatch)
        let inverse255 = Float(1.0) / Float(255.0)

        out.withUnsafeMutableBufferPointer { dst in
            rgb8.withUnsafeBufferPointer { src in
                for patch in 0..<geometry.patchCount {
                    let (px, py) = geometry.position(ofPatch: patch)
                    let originX = px * patchSize
                    let originY = py * patchSize
                    var cursor = patch * elementsPerPatch
                    for row in 0..<patchSize {
                        var source = ((originY + row) * width + originX) * 3
                        for _ in 0..<patchSize {
                            dst[cursor + 0] = Float16(Float(src[source + 0]) * inverse255)
                            dst[cursor + 1] = Float16(Float(src[source + 1]) * inverse255)
                            dst[cursor + 2] = Float16(Float(src[source + 2]) * inverse255)
                            cursor += 3
                            source += 3
                        }
                    }
                }
            }
        }
        return out
    }
}
