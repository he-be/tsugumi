import Foundation
import Tsugumi

/// What this server will accept as an image, and how much of it.
///
/// The bounds are policy, not physics: the tower resizes every image down to at
/// most `maxSoftTokens * 9` patches, so a 100-megapixel PNG produces exactly as
/// many soft tokens as a 1-megapixel one. What it does produce is a decode that
/// allocates for the original resolution first. Everything here exists to bound
/// that decode before it starts (PLAN_VISION §4-6).
public struct ServerImagePolicy: Equatable, Sendable {
    /// Soft-token budget per image (upstream `max_soft_tokens`). An upper bound
    /// on the count, never the count itself.
    public let maxSoftTokens: Int
    /// Images accepted in one request, across all turns.
    public let maxImagesPerRequest: Int
    /// Decoded (post-base64) bytes accepted per image.
    public let maxImageBytes: Int
    /// Pixels accepted per image, read from the container header.
    public let maxImagePixels: Int

    public init(maxSoftTokens: Int = 280,
                maxImagesPerRequest: Int = 4,
                maxImageBytes: Int = 8 * 1_048_576,
                maxImagePixels: Int = 50_000_000) {
        self.maxSoftTokens = maxSoftTokens
        self.maxImagesPerRequest = maxImagesPerRequest
        self.maxImageBytes = maxImageBytes
        self.maxImagePixels = maxImagePixels
    }

    public static let `default` = ServerImagePolicy()

    /// Request-body ceiling that lets the configured images through.
    ///
    /// base64 costs 4 bytes per 3, and the JSON around each part costs a little
    /// more; `textBodyBytes` covers the prompt itself. Without this the 1 MiB
    /// text-only ceiling would reject every image request with
    /// `request_too_large` before validation could say anything useful.
    public var maximumBodyBytes: Int {
        let encoded = (maxImageBytes + 2) / 3 * 4 + 1_024
        return ServerImagePolicy.textBodyBytes + maxImagesPerRequest * encoded
    }

    /// The historical text-only body ceiling, kept as the floor.
    public static let textBodyBytes = 1_048_576
}

/// One image as it arrived: the bytes, plus what the container header says.
///
/// Held as bytes rather than as a decoded bitmap so the size checks can run
/// before any decode, and so the preprocessing (resize, patchify) can happen
/// off the coordinator's single-generation lock.
public struct ServerImageAttachment: Equatable, Sendable {
    public let data: Data
    public let mediaType: String
    public let pixelWidth: Int
    public let pixelHeight: Int

    public init(data: Data, mediaType: String, pixelWidth: Int, pixelHeight: Int) {
        self.data = data
        self.mediaType = mediaType
        self.pixelWidth = pixelWidth
        self.pixelHeight = pixelHeight
    }
}

/// `image_url` handling: `data:` URIs only.
///
/// Fetching a URL the caller supplied is server-side request forgery with extra
/// steps, and a loopback inference server has no business making outbound
/// requests on behalf of whoever can reach it. `http(s)` is refused by name so
/// the answer is a 400 that says why, not a timeout (PLAN_VISION §4-6, §9).
public enum ServerImageDecoder {

    public static func attachment(fromImageURL url: String,
                                  policy: ServerImagePolicy,
                                  index: Int) throws -> ServerImageAttachment {
        let trimmed = url.trimmingCharacters(in: .whitespacesAndNewlines)
        let scheme = trimmed.prefix(while: { $0 != ":" }).lowercased()
        guard scheme == "data" else {
            let named = scheme.isEmpty ? "the given" : "\(scheme):"
            throw ServerRequestError.invalid(
                message: "image \(index) uses \(named) URL; this server accepts only "
                    + "data: URIs and never fetches image URLs",
                param: "messages",
                code: "unsupported_image_url")
        }
        let (mediaType, encoded) = try splitDataURI(trimmed, index: index)
        guard mediaType.hasPrefix("image/") else {
            throw ServerRequestError.invalid(
                message: "image \(index) declares media type \(mediaType); expected image/*",
                param: "messages",
                code: "unsupported_image_media_type")
        }
        // Reject on the encoded length first: decoding 4/3 of the ceiling to
        // find out it was over the ceiling is the one allocation this check is
        // supposed to prevent.
        let encodedCeiling = (policy.maxImageBytes + 2) / 3 * 4 + 4
        guard encoded.utf8.count <= encodedCeiling else {
            throw tooLarge(index: index,
                           detail: "its base64 payload exceeds the "
                               + "\(policy.maxImageBytes)-byte per-image limit")
        }
        guard let data = Data(base64Encoded: encoded, options: [.ignoreUnknownCharacters]),
              !data.isEmpty else {
            throw ServerRequestError.invalid(
                message: "image \(index) has an unreadable base64 payload",
                param: "messages",
                code: "invalid_image_data")
        }
        guard data.count <= policy.maxImageBytes else {
            throw tooLarge(index: index,
                           detail: "it is \(data.count) bytes, over the "
                               + "\(policy.maxImageBytes)-byte per-image limit")
        }

        let size: (width: Int, height: Int)
        do {
            size = try VisionImagePreprocessor.pixelSize(data: data)
        } catch {
            throw ServerRequestError.invalid(
                message: "image \(index) could not be decoded: \(error)",
                param: "messages",
                code: "invalid_image_data")
        }
        guard size.width * size.height <= policy.maxImagePixels else {
            throw tooLarge(index: index,
                           detail: "it is \(size.width)x\(size.height) pixels, over the "
                               + "\(policy.maxImagePixels)-pixel per-image limit")
        }
        return ServerImageAttachment(data: data,
                                     mediaType: mediaType,
                                     pixelWidth: size.width,
                                     pixelHeight: size.height)
    }

    /// `data:[<media type>][;base64],<payload>`.
    ///
    /// Only base64 payloads are accepted. A percent-encoded `data:` URI is legal
    /// but nothing produces one for a JPEG, and accepting it would mean carrying
    /// a second decoder for no caller.
    private static func splitDataURI(_ uri: String,
                                     index: Int) throws -> (mediaType: String, payload: String) {
        let body = uri.dropFirst("data:".count)
        guard let comma = body.firstIndex(of: ",") else {
            throw ServerRequestError.invalid(
                message: "image \(index) is not a valid data: URI",
                param: "messages",
                code: "invalid_image_data")
        }
        let header = body[body.startIndex..<comma].lowercased()
        let payload = String(body[body.index(after: comma)...])
        let fields = header.split(separator: ";", omittingEmptySubsequences: false)
            .map(String.init)
        guard fields.contains("base64") else {
            throw ServerRequestError.invalid(
                message: "image \(index) is not base64-encoded; only "
                    + "data:image/*;base64 URIs are accepted",
                param: "messages",
                code: "invalid_image_data")
        }
        let mediaType = fields.first.map { $0.isEmpty ? "text/plain" : $0 } ?? "text/plain"
        return (mediaType, payload)
    }

    private static func tooLarge(index: Int, detail: String) -> ServerRequestError {
        .invalid(message: "image \(index) was rejected because \(detail)",
                         param: "messages",
                         code: "image_too_large")
    }
}
