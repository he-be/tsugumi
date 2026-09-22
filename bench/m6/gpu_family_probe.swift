// M6 prefill G0 probe: what this GPU *actually* reports, and whether the
// tensor-ops kernel really compiles and builds a pipeline here.
// docs/m6-prefill/04-GATES.md G0-1. Run with `swift bench/m6/gpu_family_probe.swift`
// from the repo root (needs Sources/Tsugumi/Metal/TensorCore/tensorops.metal).
//
// Everything is printed as observed. Nothing is inferred from the chip name.
import Foundation
import Metal

guard let device = MTLCreateSystemDefaultDevice() else {
    print("FAIL: no Metal device"); exit(1)
}
let os = ProcessInfo.processInfo.operatingSystemVersion
print("device: \(device.name)")
print("os: \(os.majorVersion).\(os.minorVersion).\(os.patchVersion)")
print("registryID: \(device.registryID) unifiedMemory: \(device.hasUnifiedMemory)")
print("recommendedMaxWorkingSetSize: \(device.recommendedMaxWorkingSetSize / 1_048_576) MiB")
print("maxThreadsPerThreadgroup: \(device.maxThreadsPerThreadgroup)")

// Raw values so this compiles on SDK 26.5 (MBP) and SDK 27 (M6) alike.
// apple1 = 1001 ... apple12 = 1012, mac2 = 2002, metal3 = 5001, metal4 = 5002.
print("\n[gpu families]")
for (name, raw) in [("apple7", 1007), ("apple8", 1008), ("apple9", 1009), ("apple10", 1010),
                    ("apple11", 1011), ("apple12", 1012), ("mac2", 2002),
                    ("metal3", 5001), ("metal4", 5002)] {
    if let family = MTLGPUFamily(rawValue: raw) {
        print("  \(name) (\(raw)): \(device.supportsFamily(family))")
    } else {
        print("  \(name) (\(raw)): enum value not constructible on this SDK")
    }
}

// Runtime availability the runtime itself keys on (MetalContext.shaderLanguageVersion).
if #available(macOS 26.0, *) {
    print("\n#available(macOS 26.0): true  -> runtime picks MSL 4.0")
} else {
    print("\n#available(macOS 26.0): false -> runtime picks MSL 3.2")
}

// Compile the real tensor-ops shader with MSL 3.2 and 4.0 and report what survives.
let root = FileManager.default.currentDirectoryPath
let shaderPath = root + "/Sources/Tsugumi/Metal/TensorCore/tensorops.metal"
guard let source = try? String(contentsOfFile: shaderPath, encoding: .utf8) else {
    print("FAIL: cannot read \(shaderPath) (run from repo root)"); exit(1)
}
let kernelName = "mpp_prefill_affine_threadgroup_f16"
// MTLLanguageVersion raw = (major << 16) | minor
for (label, raw) in [("3.2", (3 << 16) | 2), ("4.0", (4 << 16) | 0)] {
    print("\n[compile tensorops.metal as MSL \(label)]")
    guard let version = MTLLanguageVersion(rawValue: UInt(raw)) else {
        print("  MTLLanguageVersion \(label) not constructible on this SDK"); continue
    }
    let opts = MTLCompileOptions()
    opts.languageVersion = version
    opts.preprocessorMacros = ["MOEPACK_AFFINE_GROUP_SIZE": 64 as NSNumber,
                               "MOEPACK_AFFINE_SYMMETRIC": 1 as NSNumber]
    do {
        let lib = try device.makeLibrary(source: source, options: opts)
        print("  makeLibrary: ok, functions = \(lib.functionNames)")
        if let fn = lib.makeFunction(name: kernelName) {
            do {
                let pso = try device.makeComputePipelineState(function: fn)
                print("  PSO \(kernelName): ok, maxTotalThreadsPerThreadgroup = \(pso.maxTotalThreadsPerThreadgroup), simdWidth = \(pso.threadExecutionWidth)")
            } catch {
                print("  PSO \(kernelName): FAILED \(error)")
            }
        } else {
            print("  function \(kernelName): absent (expected when __HAVE_TENSOR__ is undefined)")
        }
    } catch {
        let msg = "\(error)".split(separator: "\n").prefix(3).joined(separator: " | ")
        print("  makeLibrary: FAILED \(msg)")
    }
}
