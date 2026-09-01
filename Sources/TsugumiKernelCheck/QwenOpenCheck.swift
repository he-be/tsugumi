import Foundation
import Metal
import Tsugumi

// Open a repacked Qwen3.5-MoE install and report how its weights are
// quantized, tensor by tensor.
//
//   swift run -c release TsugumiKernelCheck \
//     --qwen-open scratch/ornith-oq4e-g64.moepack
//
// This is the gate `docs/qwen35moe/04-PHASES.md` 次の一手 #11 asks for: until
// the runtime accepted mixed bit widths, `Model.load` refused the production
// checkpoint, and no amount of finished kernel work could be wired to it.
// Loading is the whole check -- `Model.load` runs the manifest gates, the
// SHA-256 of `model_weights.bin`, and `validateRuntimeSchema` over all 613
// resident tensors. The census printed afterwards is what the schema derived,
// so a silent fall-back to one uniform width would be visible here as a table
// with one row.
//
// No kernels run and nothing is timed: this is a load, not a measurement.

private enum QwenOpenError: Error, CustomStringConvertible {
    case noMetalDevice

    var description: String {
        switch self {
        case .noMetalDevice: return "no Metal device"
        }
    }
}

/// Roles a resident tensor can fill, in the spelling this family uses.
private func qwenRole(of name: String) -> String? {
    let layerAgnostic = name.replacingOccurrences(
        of: #"\.layers\.[0-9]+\."#, with: ".layers.N.", options: .regularExpression)
    guard layerAgnostic.hasSuffix(".weight") else { return nil }
    return String(layerAgnostic.dropFirst("language_model.".count))
}

func runQwenOpenCheck(modelPath: String) throws -> Bool {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw QwenOpenError.noMetalDevice
    }
    let directoryURL = URL(fileURLWithPath: modelPath)
    print("=== open a qwen3_5_moe install (docs/qwen35moe/04-PHASES 次の一手 #11) ===")
    print("  model  \(directoryURL.path)")

    var stats = ModelLoadStats()
    let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
    let model = try Model.load(directoryURL: directoryURL,
                               device: device,
                               expecting: .ornith1_5_35B_A3B,
                               streamingMode: .pread(slotCount: 32),
                               loadStats: &stats)
    let elapsed = Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started) / 1e6

    print("  loaded in \(String(format: "%.0f", elapsed)) ms"
          + " (sha256 \(String(format: "%.0f", Double(stats.eagerSha256Nanos) / 1e6)) ms)")
    print("  arch   \(model.config.numLayers) layers, \(model.config.numExperts) experts,"
          + " group \(model.affineGroupSize), scheme \(model.affineScheme)")

    // Census by role: how many tensors of each width the index actually holds.
    var census: [String: [Int: Int]] = [:]
    for name in model.residentTensorNames {
        guard let role = qwenRole(of: name), let bits = model.residentWeightBits(name) else {
            continue
        }
        census[role, default: [:]][bits, default: 0] += 1
    }
    let mixedRoles = census.filter { $0.value.count > 1 }
    print("")
    print("  weight widths derived from the resident index:")
    for role in census.keys.sorted() {
        let widths = census[role]!.keys.sorted()
            .map { "\($0)-bit x\(census[role]![$0]!)" }
            .joined(separator: ", ")
        let padding = String(repeating: " ", count: max(1, 56 - role.count))
        print("    \(role)\(padding)\(widths)")
    }
    print("")
    if mixedRoles.isEmpty {
        print("FAIL  no role mixes widths — this install would not have needed #11")
        return false
    }
    print("PASS  opened; \(mixedRoles.count) of \(census.count) roles mix widths in one slot")
    return true
}
