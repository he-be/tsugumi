import Foundation
import Tsugumi

/// The error taxonomy of SPEC §10 (ERR-2).
///
/// One `type` per HTTP status, and nothing else: an error this server can
/// answer with is always one of these six. The mapping is the reference
/// implementation's (`server-common.cpp:25-54`); the envelope around it is
/// OpenAI's, which outranks the reference for `/v1/*` wire shape (DEV-1).
public enum ServerErrorType: String, Codable, Equatable, Sendable, CaseIterable {
    case invalidRequest = "invalid_request_error"
    case notFound = "not_found_error"
    case notSupported = "not_supported_error"
    case unavailable = "unavailable_error"
    case exceedContextSize = "exceed_context_size_error"
    case server = "server_error"
}

extension ServerErrorType {
    /// ERR-2. The status is a function of the type and of nothing else, so an
    /// error cannot be raised with one and answered with the other.
    public var httpStatusCode: Int {
        switch self {
        case .invalidRequest: 400
        case .notFound: 404
        case .notSupported: 501
        case .unavailable: 503
        case .exceedContextSize: 400
        case .server: 500
        }
    }
}

/// ERR-1. OpenAI's error body: four keys, always present, `code` a string or
/// null. The reference implementation puts the HTTP number in `code`; `/v1/*`
/// follows OpenAI instead (DEV-1).
public struct OpenAIErrorEnvelope: Codable, Equatable, Sendable {
    public struct Detail: Codable, Equatable, Sendable {
        public let message: String
        public let type: String
        public let param: String?
        public let code: String?

        enum CodingKeys: String, CodingKey {
            case message, type, param, code
        }

        // Written by hand because the synthesized encoder drops a nil optional
        // rather than writing null, and ERR-1 wants all four keys on the wire.
        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(message, forKey: .message)
            try container.encode(type, forKey: .type)
            try container.encode(param, forKey: .param)
            try container.encode(code, forKey: .code)
        }
    }

    public let error: Detail

    public init(message: String,
                type: ServerErrorType = .invalidRequest,
                param: String? = nil,
                code: String? = nil) {
        error = Detail(message: message,
                       type: type.rawValue,
                       param: param,
                       code: code)
    }
}

/// A refusal, carrying everything ERR-1 needs and nothing the transport has to
/// guess at.
public struct ServerRequestError: Error, Equatable, Sendable {
    public let type: ServerErrorType
    public let message: String
    public let param: String?
    public let code: String?

    public init(type: ServerErrorType,
                message: String,
                param: String? = nil,
                code: String? = nil) {
        self.type = type
        self.message = message
        self.param = param
        self.code = code
    }

    public var envelope: OpenAIErrorEnvelope {
        OpenAIErrorEnvelope(message: message, type: type, param: param, code: code)
    }

    public static func invalid(message: String,
                               param: String? = nil,
                               code: String? = nil) -> Self {
        Self(type: .invalidRequest, message: message, param: param, code: code)
    }

    public static func notSupported(message: String,
                                    param: String? = nil,
                                    code: String? = nil) -> Self {
        Self(type: .notSupported, message: message, param: param, code: code)
    }

    public static func notFound(message: String,
                                param: String? = nil,
                                code: String? = nil) -> Self {
        Self(type: .notFound, message: message, param: param, code: code)
    }

    public static func unavailable(message: String,
                                   param: String? = nil,
                                   code: String? = nil) -> Self {
        Self(type: .unavailable, message: message, param: param, code: code)
    }

    public static func exceedContextSize(message: String,
                                         param: String? = nil,
                                         code: String? = nil) -> Self {
        Self(type: .exceedContextSize, message: message, param: param, code: code)
    }

    /// S-2 (`docs/qwen38/26` §5): a structured output the decoder could not
    /// finish is a context overflow, not a malformed call, when generation was
    /// stopped by the context itself. The prompt fit, the model began a call,
    /// and the last position of the context cut it in half — the same
    /// condition as a prompt that does not fit, reported the same way.
    ///
    /// `reserved` is what the loop keeps back past the prompt (the verify
    /// pass's drafted row). Nil when the stop was the request's own limit or
    /// anything but the token budget: that failure is still the decoder's.
    static func generationReachedContext(stop: StopReason,
                                         promptTokens: Int,
                                         generatedTokens: Int,
                                         reserved: Int = 0,
                                         maxContext: Int) -> Self? {
        guard stop == .maxTokens, promptTokens + generatedTokens + reserved >= maxContext else { return nil }
        return .exceedContextSize(
            message: "prompt of \(promptTokens) tokens and \(generatedTokens) generated tokens reached the "
                + "configured context of \(maxContext) inside a tool call",
            param: "messages",
            code: "context_length_exceeded")
    }

    /// LIF-4: a full slot and a full queue is a 503, not a 429 — the client
    /// should come back, and nothing about the request was wrong.
    public static let queueFull = Self(type: .unavailable,
                                       message: "generation queue is full",
                                       code: "queue_full")
}
