import Foundation
import Metal

enum MetalError: Error, CustomStringConvertible {
    case noDevice
    case noQueue
    case missingShaderResource(String)
    case missingFunction(String)
    case libraryCompileFailed(String)
    case unsupportedGroupSize(Int)
    case groupSizeLockedByCompiledLibrary(current: Int, requested: Int)
    case affineSchemeLockedByCompiledLibrary(current: String, requested: String)

    public var description: String {
        switch self {
        case .noDevice:                   return "No Metal device"
        case .noQueue:                    return "Failed to create Metal command queue"
        case .missingShaderResource(let n): return "Shader resource missing: \(n)"
        case .missingFunction(let n):     return "Metal function missing in library: \(n)"
        case .libraryCompileFailed(let s):return "Metal library compile failed: \(s)"
        case .unsupportedGroupSize(let n):
            return "Unsupported affine group size \(n); supported: " +
                   "\(Quantization.supportedGroupSizes.sorted())"
        case .groupSizeLockedByCompiledLibrary(let current, let requested):
            return """
                affine group size is already baked into the compiled shader \
                library as \(current); cannot switch to \(requested). Set it \
                before the first pipeline() or library access.
                """
        case .affineSchemeLockedByCompiledLibrary(let current, let requested):
            return """
                affine scheme is already baked into the compiled shader \
                library as \(current); cannot switch to \(requested). Set it \
                before the first pipeline() or library access.
                """
        }
    }
}

func checkCommandBufferError(_ error: (any Error)?) throws {
    if let error {
        throw error
    }
}

public struct MetalFunctionConstant: Hashable, Sendable {
    public enum Value: Hashable, Sendable {
        case bool(Bool)
        case uint32(UInt32)
        case float(Float)
    }

    public let index: Int
    public let value: Value

    public init(index: Int, value: Value) {
        self.index = index
        self.value = value
    }
}

/// Single owner of the `MTLDevice`, queue, and the runtime-compiled shader library.
/// On Mac and iOS we ship `.metal` source files as bundle resources and compile
/// them into one combined `MTLLibrary` at startup. This keeps the dev loop
/// fast — edit a shader, rebuild the Swift target, no Xcode metallib step.
///
/// The affine quantization group size is a **process-wide compile-time
/// constant** injected into that source as `TURBO_AFFINE_GROUP_SIZE`. One
/// process serves one model, and a model's group size is uniform (the source
/// `config.json` carries a single base with no per-tensor overrides), so
/// specializing the whole library is simpler than carrying the group size as a
/// runtime value through every kernel. This is why the library is compiled
/// lazily: `MetalContext` is constructed before `Model.load` at all three entry
/// points, so the group size is not known yet at `init`.
///
/// `@unchecked Sendable`: device and queue are immutable and Metal objects are
/// thread-safe for encoding; the group size, the lazily compiled library, and
/// the pipeline cache are the mutable state and are lock-guarded.
public final class MetalContext: @unchecked Sendable {
    public let device:  MTLDevice
    public let queue:   MTLCommandQueue

    private struct PipelineCacheKey: Hashable {
        var name: String
        var constants: [MetalFunctionConstant]
        var maxTotalThreadsPerThreadgroup: Int?
    }

    private var pipelineCache: [PipelineCacheKey: MTLComputePipelineState] = [:]
    private let pipelineCacheLock = NSLock()

    /// Guards `compiledLibrary` and `groupSize`. Held across the compile so two
    /// threads racing on first use produce one library, not two.
    private let libraryLock = NSLock()
    private var compiledLibrary: MTLLibrary?
    private var groupSize: Int = Quantization.groupSize
    private var scheme: Quantization.AffineScheme = .affine

    public init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw MetalError.noDevice }
        guard let q   = dev.makeCommandQueue()           else { throw MetalError.noQueue }
        self.device  = dev
        self.queue   = q
    }

    /// A second context on the *same device* — its own queue, its own shader
    /// library, its own affine group size.
    ///
    /// The "one process, one group size" rule above holds per model, and the
    /// MTP drafter breaks it from inside one file: the 26B target is packed at
    /// group 32 and the pinned drafter at group 64 (`manifest.draft.quant`),
    /// so their kernels cannot come from one specialized library. Buffers are
    /// device-scoped, so the two contexts pass buffers freely; ordering across
    /// the two queues is the caller's business.
    public init(sharingDeviceWith other: MetalContext) throws {
        guard let q = other.device.makeCommandQueue() else { throw MetalError.noQueue }
        self.device = other.device
        self.queue  = q
    }

    /// The affine group size the shader library is (or will be) compiled for.
    public var affineGroupSize: Int {
        libraryLock.lock()
        defer { libraryLock.unlock() }
        return groupSize
    }

    /// Select the affine group size for this process. Must be called before the
    /// first `pipeline(...)` or `library` access — after that the value is baked
    /// into the compiled library and switching would silently mismatch the
    /// pipelines already handed out.
    ///
    /// Setting the value it already has is always allowed (and is a no-op), so
    /// loading a group-64 model on a fresh context never has to care about order.
    public func setAffineGroupSize(_ value: Int) throws {
        guard Quantization.supportedGroupSizes.contains(value) else {
            throw MetalError.unsupportedGroupSize(value)
        }
        libraryLock.lock()
        defer { libraryLock.unlock() }
        if groupSize == value { return }
        if compiledLibrary != nil {
            throw MetalError.groupSizeLockedByCompiledLibrary(current: groupSize,
                                                              requested: value)
        }
        groupSize = value
    }

    /// The affine scheme the shader library is (or will be) compiled for.
    public var affineScheme: Quantization.AffineScheme {
        libraryLock.lock()
        defer { libraryLock.unlock() }
        return scheme
    }

    /// Select the affine scheme for this context, with the same lifecycle rule
    /// as `setAffineGroupSize`: before the first pipeline or library access.
    ///
    /// `sym` compiles the library with `TURBO_AFFINE_SYMMETRIC = 1`, which
    /// turns every group's zero point into `-8 * scale` and stops the kernels
    /// reading the bias arrays. The bias *bindings* stay -- callers alias them
    /// onto the scale buffer -- so no kernel signature depends on the scheme.
    public func setAffineScheme(_ value: Quantization.AffineScheme) throws {
        libraryLock.lock()
        defer { libraryLock.unlock() }
        if scheme == value { return }
        if compiledLibrary != nil {
            throw MetalError.affineSchemeLockedByCompiledLibrary(current: scheme.rawValue,
                                                                 requested: value.rawValue)
        }
        scheme = value
    }

    /// Whether `setAffineScheme(value)` would succeed.
    public func canUseAffineScheme(_ value: Quantization.AffineScheme) -> Bool {
        libraryLock.lock()
        defer { libraryLock.unlock() }
        return scheme == value || compiledLibrary == nil
    }

    /// Whether `setAffineGroupSize(value)` would succeed. False once the
    /// library has been compiled for a different size — a long-lived context
    /// (the Mac app reuses one across model loads) has to be replaced rather
    /// than re-specialized.
    public func canUseAffineGroupSize(_ value: Int) -> Bool {
        guard Quantization.supportedGroupSizes.contains(value) else { return false }
        libraryLock.lock()
        defer { libraryLock.unlock() }
        return groupSize == value || compiledLibrary == nil
    }

    /// The combined shader library, compiled on first access.
    public var library: MTLLibrary {
        get throws {
            libraryLock.lock()
            defer { libraryLock.unlock() }
            if let compiledLibrary { return compiledLibrary }
            let built = try Self.compileShaderLibrary(device: device,
                                                      affineGroupSize: groupSize,
                                                      affineScheme: scheme)
            compiledLibrary = built
            return built
        }
    }

    /// Production shader modules compiled into the shared runtime library.
    private static let shaderModules: [String] = [
        "dequant_int4",
        "dequant_int8",
        "rmsnorm",
        "rope",
        "attention",
        "moe",
        "logit",
        "utility",
        "fused",
        "prefill",
        "vision",
    ]

    /// Bundle locations for runtime shader modules. `tensorops` and `gdn` are
    /// listed here but not in `shaderModules`: they are compiled on their own
    /// through `moduleLibrary(device:module:)` so they cost the Gemma 4 path
    /// nothing at startup.
    private static let shaderSubdirectories: [String: String] = [
        "attention": "Metal/Attention",
        "dequant_int4": "Metal/Quant",
        "dequant_int8": "Metal/Quant",
        "fused": "Metal/Fusions",
        "logit": "Metal/Sampling",
        "moe": "Metal/MoE",
        "prefill": "Metal/Prefill",
        "rmsnorm": "Metal/Primitives",
        "rope": "Metal/Primitives",
        "gdn": "Metal/Qwen",
        "tensorops": "Metal/TensorCore",
        "utility": "Metal/Primitives",
        "vision": "Metal/Vision",
    ]

    private static func shaderURL(module: String) -> URL? {
        guard let subdirectory = shaderSubdirectories[module] else { return nil }
        return Bundle.module.url(forResource: module, withExtension: "metal",
                                 subdirectory: subdirectory)
    }

    /// Highest Metal Shading Language version the running OS accepts.
    /// MSL 4.0 exists only on macOS 26 and carries the MPP tensor operations
    /// used by the prefill fast paths. On macOS 15 the shaders still compile:
    /// every tensor kernel is guarded by `__HAVE_TENSOR__`, which the MSL 3.2
    /// compiler leaves undefined, so those kernels drop out of the library and
    /// the callers fall back to their non-tensor pipelines.
    static var shaderLanguageVersion: MTLLanguageVersion {
        if #available(macOS 26.0, iOS 26.0, *) {
            return .version4_0
        }
        return .version3_2
    }

    /// Compile options shared by the combined library and single-module builds.
    /// `TURBO_AFFINE_GROUP_SIZE` is what every `k*GroupSize` constant in the
    /// shaders resolves to; each `.metal` file still carries an `#ifndef`
    /// fallback of 64 so the sources compile standalone.
    private static func compileOptions(affineGroupSize: Int,
                                       affineScheme: Quantization.AffineScheme)
        -> MTLCompileOptions {
        let opts = MTLCompileOptions()
        // The MPP prefill path requires MSL 4.0 tensor operations.
        opts.languageVersion = shaderLanguageVersion
        opts.preprocessorMacros = [
            "TURBO_AFFINE_GROUP_SIZE": NSNumber(value: affineGroupSize),
            "TURBO_AFFINE_SYMMETRIC": NSNumber(value: affineScheme == .sym ? 1 : 0),
        ]
        return opts
    }

    private static func compileShaderLibrary(
        device: MTLDevice,
        affineGroupSize: Int,
        affineScheme: Quantization.AffineScheme) throws -> MTLLibrary {
        var combined = ""
        for name in shaderModules {
            guard let url = shaderURL(module: name) else {
                throw MetalError.missingShaderResource(name)
            }
            let src = try String(contentsOf: url, encoding: .utf8)
            combined += "\n// ==== \(name).metal ====\n" + src + "\n"
        }
        do {
            return try device.makeLibrary(
                source: combined,
                options: compileOptions(affineGroupSize: affineGroupSize,
                                        affineScheme: affineScheme))
        } catch {
            throw MetalError.libraryCompileFailed("\(error)")
        }
    }

    /// Compile a shader module separately from the shared runtime library.
    public static func moduleLibrary(device: MTLDevice,
                                     module: String,
                                     affineGroupSize: Int = Quantization.groupSize,
                                     affineScheme: Quantization.AffineScheme = .affine)
        throws -> MTLLibrary {
        guard let url = shaderURL(module: module) else {
            throw MetalError.missingShaderResource(module)
        }
        let src = try String(contentsOf: url, encoding: .utf8)
        do {
            return try device.makeLibrary(
                source: src,
                options: compileOptions(affineGroupSize: affineGroupSize,
                                        affineScheme: affineScheme))
        } catch {
            throw MetalError.libraryCompileFailed("\(error)")
        }
    }

    public func pipeline(_ name: String) throws -> MTLComputePipelineState {
        try pipeline(name, constants: [])
    }

    public func pipeline(_ name: String,
                         constants: [MetalFunctionConstant]) throws -> MTLComputePipelineState {
        try pipeline(name, constants: constants, maxTotalThreadsPerThreadgroup: nil)
    }

    public func pipeline(_ name: String,
                         constants: [MetalFunctionConstant],
                         maxTotalThreadsPerThreadgroup hint: Int?) throws -> MTLComputePipelineState {
        if let hint {
            precondition(hint > 0, "maxTotalThreadsPerThreadgroup must be positive")
        }
        let sortedConstants = constants.sorted {
            if $0.index != $1.index { return $0.index < $1.index }
            return Self.constantSortKey($0.value) < Self.constantSortKey($1.value)
        }
        let key = PipelineCacheKey(name: name,
                                   constants: sortedConstants,
                                   maxTotalThreadsPerThreadgroup: hint)
        pipelineCacheLock.lock()
        let cached = pipelineCache[key]
        pipelineCacheLock.unlock()
        if let cached { return cached }

        let library = try self.library
        guard library.functionNames.contains(name) else {
            throw MetalError.missingFunction(name)
        }

        let values = MTLFunctionConstantValues()
        for constant in sortedConstants {
            switch constant.value {
            case .bool(let value):
                var v = value
                values.setConstantValue(&v, type: .bool, index: constant.index)
            case .uint32(let value):
                var v = value
                values.setConstantValue(&v, type: .uint, index: constant.index)
            case .float(let value):
                var v = value
                values.setConstantValue(&v, type: .float, index: constant.index)
            }
        }

        let fn = try library.makeFunction(name: name, constantValues: values)
        let p: MTLComputePipelineState
        if let hint {
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = fn
            descriptor.maxTotalThreadsPerThreadgroup = hint
            var reflection: MTLAutoreleasedComputePipelineReflection?
            p = try device.makeComputePipelineState(descriptor: descriptor,
                                                    options: [],
                                                    reflection: &reflection)
        } else {
            p = try device.makeComputePipelineState(function: fn)
        }
        pipelineCacheLock.lock()
        pipelineCache[key] = p
        pipelineCacheLock.unlock()
        return p
    }

    private static func constantSortKey(_ value: MetalFunctionConstant.Value) -> String {
        switch value {
        case .bool(let v):   return "b:\(v ? 1 : 0)"
        case .uint32(let v): return "u:\(v)"
        case .float(let v):  return "f:\(v.bitPattern)"
        }
    }
}
