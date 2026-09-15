import Foundation

/// A constraint that forbids a fixed set of token ids and allows everything
/// else, including the end of generation (SPEC §6 GEN-4, `docs/qwen38/26` §2).
///
/// This is what `tool_choice: none` is: the tool declarations stay in the
/// prompt (the cache keeps its prefix), so the model still sees the tools, and
/// the only guarantee that no call is produced is that the call's start token
/// cannot be drawn. A line in the prompt asking the model not to call lowers
/// the probability; it does not make it zero (`docs/qwen38/25` §3: 0.30 at the
/// real last round).
///
/// Only the ids are forbidden, not their spelling as a run of ordinary tokens:
/// the decoders cut a call out by token id (DEV-22), so a spelled marker is
/// text and never a call.
///
/// Stateless: `accept` has nothing to advance, and the same ids are forbidden
/// at every position — inside the thought channel as well.
public final class ForbiddenTokensConstraint: GenerationConstraint {
    public let forbiddenTokenIDs: Set<Int32>

    public init(forbiddenTokenIDs: Set<Int32>) {
        self.forbiddenTokenIDs = forbiddenTokenIDs
    }

    public var mayEndHere: Bool { true }

    public func allows(tokenID: Int32) -> Bool {
        !forbiddenTokenIDs.contains(tokenID)
    }

    public func fillAllowedMask(_ allowed: UnsafeMutableBufferPointer<Bool>) throws {
        allowed.update(repeating: true)
        for id in forbiddenTokenIDs where id >= 0 && Int(id) < allowed.count {
            allowed[Int(id)] = false
        }
    }

    public func accept(tokenID: Int32) throws {}
}
