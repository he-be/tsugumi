// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Tsugumi",
    // English is the key language of every `Localizable.strings`; Japanese
    // is the one translation shipped (Sources/TsugumiApp/*/Resources/*.lproj).
    defaultLocalization: "en",
    platforms: [
        .macOS(.v15),
        .iOS(.v26),
    ],
    products: [
        .library(name: "Tsugumi", targets: ["Tsugumi"]),
        .executable(name: "TsugumiRepack", targets: ["TsugumiRepack"]),
        .executable(name: "TsugumiCLI", targets: ["TsugumiCLI"]),
        .executable(name: "TsugumiMac", targets: ["TsugumiMac"]),
        .executable(name: "TsugumiDecodeService", targets: ["TsugumiDecodeService"]),
        .executable(name: "TsugumiServer", targets: ["TsugumiServer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.101.3"),
        // GFM parser for the Mac app's response renderer: tables, HTML nodes
        // and nesting come back as a typed AST rather than flattened runs.
        .package(url: "https://github.com/swiftlang/swift-markdown.git", from: "0.5.0"),
    ],
    targets: [
        .target(
            name: "MoEPackFormat",
            path: "Sources/MoEPackFormat"
        ),
        // Finds SwiftPM resource bundles in the layout the shipped `.app`
        // stores them in; shared by every target that carries resources.
        .target(
            name: "TsugumiBundleLocation",
            path: "Sources/TsugumiBundleLocation"
        ),
        .target(
            name: "Tsugumi",
            dependencies: [
                "MoEPackFormat",
                "TsugumiBundleLocation",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/Tsugumi",
            resources: [
                .copy("Metal"),
                .copy("Templates"),
            ]
        ),
        .target(
            name: "TsugumiRepackCore",
            dependencies: ["MoEPackFormat"],
            path: "Sources/TsugumiRepack/Core"
        ),
        .executableTarget(
            name: "TsugumiRepack",
            dependencies: ["TsugumiRepackCore"],
            path: "Sources/TsugumiRepack/Command"
        ),
        .target(
            name: "TsugumiCLICore",
            dependencies: ["Tsugumi"],
            path: "Sources/TsugumiCLI",
            exclude: ["Command"]
        ),
        .executableTarget(
            name: "TsugumiCLI",
            dependencies: ["TsugumiCLICore"],
            path: "Sources/TsugumiCLI/Command"
        ),
        .target(
            name: "TsugumiAppCore",
            dependencies: [
                "Tsugumi",
                "TsugumiBundleLocation",
                "TsugumiRepackCore",
                "TsugumiDecodeProtocol",
                "TsugumiServerCore",
            ],
            path: "Sources/TsugumiApp/Core",
            resources: [
                .copy("Resources/web-search-system-prompt.txt"),
                .copy("Resources/search-tool-prompts.json"),
                .process("Resources/en.lproj"),
                .process("Resources/ja.lproj"),
            ]
        ),
        .target(
            name: "TsugumiMacPresentation",
            dependencies: [
                "TsugumiAppCore",
                .product(name: "Markdown", package: "swift-markdown"),
            ],
            path: "Sources/TsugumiApp/MacPresentation"
        ),
        .target(
            name: "TsugumiDecodeProtocol",
            path: "Sources/TsugumiDecodeProtocol"
        ),
        .executableTarget(
            name: "TsugumiDecodeService",
            dependencies: ["TsugumiAppCore", "TsugumiDecodeProtocol"],
            path: "Sources/TsugumiDecodeService"
        ),
        .target(
            name: "TsugumiServerCore",
            dependencies: [
                "Tsugumi",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "Sources/TsugumiServer/Core"
        ),
        .executableTarget(
            name: "TsugumiServer",
            dependencies: ["TsugumiServerCore"],
            path: "Sources/TsugumiServer/Command"
        ),
        .executableTarget(
            name: "TsugumiMac",
            dependencies: ["TsugumiAppCore", "TsugumiMacPresentation",
                           "TsugumiBundleLocation"],
            path: "Sources/TsugumiApp/Mac",
            resources: [
                .copy("Resources/tsugumi-app-icon.png"),
                .process("Resources/en.lproj"),
                .process("Resources/ja.lproj"),
            ]
        ),
        .target(
            name: "TsugumiValidationSupport",
            dependencies: ["Tsugumi"],
            path: "Sources/TsugumiValidation/Support"
        ),
        // Numeric self-check for the affine-INT4 kernels at each supported
        // group size. An executable rather than a test target because
        // `swift test` cannot run in the development environment.
        .executableTarget(
            name: "TsugumiKernelCheck",
            dependencies: ["Tsugumi", "TsugumiValidationSupport"],
            path: "Sources/TsugumiKernelCheck"
        ),
        .testTarget(
            name: "MoEPackFormatTests",
            dependencies: ["MoEPackFormat"],
            path: "Tests/MoEPackFormat"
        ),
        .testTarget(
            name: "MoEPackFormatCompatibilityTests",
            dependencies: ["MoEPackFormat", "Tsugumi", "TsugumiRepackCore"],
            path: "Tests/MoEPackFormatCompatibility",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "TsugumiTestsCore",
            dependencies: [
                "Tsugumi",
                "TsugumiValidationSupport",
                "TsugumiRepackCore",
                "TsugumiCLICore",
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Tests/Tsugumi/Core"
        ),
        .testTarget(
            name: "TsugumiRepackTests",
            dependencies: ["MoEPackFormat", "TsugumiRepackCore"],
            path: "Tests/TsugumiRepack/Core"
        ),
        .testTarget(
            name: "TsugumiAppCoreTests",
            dependencies: ["TsugumiAppCore", "Tsugumi", "TsugumiRepackCore", "TsugumiDecodeProtocol"],
            path: "Tests/TsugumiApp/Core",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "TsugumiDecodeServiceTests",
            dependencies: ["TsugumiDecodeService", "TsugumiAppCore", "TsugumiDecodeProtocol"],
            path: "Tests/TsugumiDecodeService"
        ),
        .testTarget(
            name: "TsugumiMacPresentationTests",
            dependencies: ["TsugumiAppCore", "TsugumiMacPresentation"],
            path: "Tests/TsugumiApp/MacPresentation",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "TsugumiServerTests",
            dependencies: [
                "TsugumiServerCore",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "Tests/TsugumiServer",
            resources: [.copy("Fixtures")]
        ),
    ]
)
