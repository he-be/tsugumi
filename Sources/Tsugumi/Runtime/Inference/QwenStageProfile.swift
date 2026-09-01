import Foundation

/// Where one token's wall clock goes, stage by stage, on both of the runner's
/// paths — the per-token decode loop and the T-row chunk path the speculative
/// verify pass borrows.
///
/// This exists because `36-MTP-DECODE.md` §4-4 named two suspects for the
/// remaining 1.21x between them ("host +11 ms = the per-layer route grouping,
/// GPU +2.7 ms = prefill attention and the block router") and marked both
/// **unconfirmed**. `37-MTP-POSTMORTEM-PLAN.md` §1-4 rule 3 says an estimate
/// whose base has not been measured does not get to lead the next design, so
/// the stages are cut the same way on both paths and the two are subtracted.
///
/// Off unless `TF_QWEN_STAGE_PROFILE=1`: the accumulators are three stores per
/// region, but the regions bracket `commit`/`waitUntilCompleted` pairs and the
/// point of the number is to be trusted, not to be free.
public enum QwenStage: Int, CaseIterable, Sendable {
    /// The embedding lookup.
    case embed
    /// Input norm, attention or the delta rule, the residual, the post-attention
    /// norm and the router — the layer's first command buffer, which is joined
    /// because the host has to read the routes.
    case preRouter
    /// Reading the router's picks back and turning them into what the MoE
    /// kernels take: on decode a list of eight experts, on the chunk path the
    /// sorted pairs, groups and tiles of `PrefillMoEGrouping`.
    case routeReadback
    /// Choosing cache slots for the experts and issuing their reads.
    case expertPlan
    /// Blocking on those reads.
    case expertIO
    /// The shared expert branch.
    case sharedExpert
    /// The routed experts themselves — argument buffers, encode, and the tiles'
    /// GPU time.
    case routedExperts
    /// The token-major reduce and the residual adds.
    case reduceTail
    /// Joining whatever the layers left committed.
    case drain
    /// The 508 MB head.
    case head
    /// The MTP drafter's own pass (`QwenMTPDrafter`).
    case draft
    /// Anything not bracketed.
    case other

    public var name: String {
        switch self {
        case .embed: return "embed"
        case .preRouter: return "preRouter"
        case .routeReadback: return "routes"
        case .expertPlan: return "plan"
        case .expertIO: return "io"
        case .sharedExpert: return "shared"
        case .routedExperts: return "routed"
        case .reduceTail: return "tail"
        case .drain: return "drain"
        case .head: return "head"
        case .draft: return "draft"
        case .other: return "other"
        }
    }
}

/// Wall and GPU seconds per stage, plus how many regions were entered.
///
/// `wall` is the host's own clock around the region, so it contains that
/// region's GPU time and its expert reads; `wall - gpu` is what the host spent
/// encoding, reading back and waiting. Regions do not nest — a stage that
/// brackets a `wait` does not also bracket the plan inside it — so the sum of
/// `wall` is the path's wall clock minus whatever ran outside any region
/// (which is what `.other` would hold if anything did).
public struct QwenStageProfile: Sendable {
    public private(set) var wall: [Double]
    public private(set) var gpu: [Double]
    public private(set) var regions: [Int]
    public private(set) var buffers: [Int]
    /// Units the totals should be divided by — decode tokens, or verify passes.
    public var units: Int = 0

    public init() {
        let n = QwenStage.allCases.count
        wall = [Double](repeating: 0, count: n)
        gpu = [Double](repeating: 0, count: n)
        regions = [Int](repeating: 0, count: n)
        buffers = [Int](repeating: 0, count: n)
    }

    mutating func addWall(_ stage: QwenStage, _ seconds: Double) {
        wall[stage.rawValue] += seconds
        regions[stage.rawValue] += 1
    }

    mutating func addGPU(_ stage: QwenStage, _ seconds: Double) {
        gpu[stage.rawValue] += seconds
        buffers[stage.rawValue] += 1
    }

    /// This profile minus an earlier snapshot of itself — how a phase split
    /// when the accumulators were not reset at its boundary.
    public func subtracting(_ other: QwenStageProfile) -> QwenStageProfile {
        var out = self
        for i in wall.indices {
            out.wall[i] -= other.wall[i]
            out.gpu[i] -= other.gpu[i]
            out.regions[i] -= other.regions[i]
            out.buffers[i] -= other.buffers[i]
        }
        return out
    }

    public var totalWall: Double { wall.reduce(0, +) }
    public var totalGPU: Double { gpu.reduce(0, +) }

    /// One line per stage that was entered, in milliseconds per unit.
    public func report(units: Int) -> String {
        guard units > 0 else { return "" }
        var out = ""
        for stage in QwenStage.allCases where regions[stage.rawValue] > 0 {
            let w = wall[stage.rawValue] * 1e3 / Double(units)
            let g = gpu[stage.rawValue] * 1e3 / Double(units)
            out += "[stage \(stage.name) wall=\(String(format: "%.3f", w))ms"
            out += " gpu=\(String(format: "%.3f", g))ms"
            out += " host=\(String(format: "%.3f", w - g))ms"
            out += " n=\(String(format: "%.1f", Double(regions[stage.rawValue]) / Double(units)))]\n"
        }
        out += "[stage total wall=\(String(format: "%.3f", totalWall * 1e3 / Double(units)))ms"
        out += " gpu=\(String(format: "%.3f", totalGPU * 1e3 / Double(units)))ms]\n"
        return out
    }
}
