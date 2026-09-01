import Foundation
import TsugumiBundleLocation

/// This target's resource bundle (`app-prompts.json`), found the way the
/// shipped `.app` stores it. See `PackagedResourceBundle`.
enum AppCoreResources {
    static let bundle: Bundle = PackagedResourceBundle.named("Tsugumi_TsugumiAppCore")
        ?? .module
}
