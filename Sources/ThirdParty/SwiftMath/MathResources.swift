import Foundation
import TsugumiBundleLocation

/// The resource bundle with the math fonts, in the layout the app ships in
/// first (`Contents/Resources/Tsugumi_SwiftMath.bundle`), then the one
/// SwiftPM built beside the executable (`swift build`, `swift test`).
enum MathResources {
    static let bundle: Bundle = PackagedResourceBundle.named("Tsugumi_SwiftMath") ?? Bundle.module
}
