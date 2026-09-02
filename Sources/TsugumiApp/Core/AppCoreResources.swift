import Foundation
import TsugumiBundleLocation

/// This target's resource bundle (`app-prompts.json`,
/// `web-search-system-prompt.txt`), found the way the shipped `.app` stores
/// it. See `PackagedResourceBundle`.
enum AppCoreResources {
    static let bundle: Bundle = PackagedResourceBundle.named("Tsugumi_TsugumiAppCore")
        ?? .module
}
