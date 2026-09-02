import Foundation

/// The one seam the web tools reach the network through, so a test can
/// answer a request with a fixture instead of a socket.
public protocol HTTPTransport: Sendable {
    func perform(_ request: URLRequest) async throws -> (Data, HTTPURLResponse)
}

public struct URLSessionTransport: HTTPTransport {
    private let session: URLSession

    public init(timeout: TimeInterval = 20) {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout * 2
        configuration.httpAdditionalHeaders = [
            "Accept-Language": "ja,en;q=0.8",
        ]
        session = URLSession(configuration: configuration)
    }

    public func perform(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw WebToolError.transport("not an HTTP response")
        }
        return (data, http)
    }
}

public enum WebToolError: Error, Equatable, Sendable, CustomStringConvertible {
    case missingKey(String)
    case transport(String)
    case status(Int, String)
    case badResponse(String)
    case unsafeURL(String)
    case unsupportedContent(String)

    public var description: String {
        switch self {
        case .missingKey(let provider): "\(provider) API key is not configured"
        case .transport(let detail): "network error: \(detail)"
        case .status(let code, let provider): "\(provider) answered HTTP \(code)"
        case .badResponse(let detail): "unexpected response: \(detail)"
        case .unsafeURL(let url): "refusing to fetch \(url)"
        case .unsupportedContent(let type): "cannot read \(type) content"
        }
    }
}

extension URL {
    /// Only public web addresses are fetched. The model chooses the URL, so a
    /// page could steer it at the local server or the LAN; this keeps a
    /// fetch to what a search result can legitimately point at.
    public var isPublicWebAddress: Bool {
        guard let scheme = scheme?.lowercased(), scheme == "http" || scheme == "https",
              let host = host?.lowercased(), !host.isEmpty else { return false }
        if host == "localhost" || host.hasSuffix(".local") || host.hasSuffix(".localhost") {
            return false
        }
        if host == "0.0.0.0" || host == "::1" || host == "[::1]" { return false }
        let parts = host.split(separator: ".").compactMap { Int($0) }
        if parts.count == 4 {
            if parts[0] == 10 || parts[0] == 127 || parts[0] == 0 { return false }
            if parts[0] == 169 && parts[1] == 254 { return false }
            if parts[0] == 172 && (16...31).contains(parts[1]) { return false }
            if parts[0] == 192 && parts[1] == 168 { return false }
            if parts[0] == 100 && (64...127).contains(parts[1]) { return false }
        }
        if host.hasPrefix("fe80:") || host.hasPrefix("fc") || host.hasPrefix("fd") { return false }
        return true
    }
}
