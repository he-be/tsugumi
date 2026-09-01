import Foundation
import TsugumiBundleLocation

/// This target's resource bundle (the app icon), found the way the shipped
/// `.app` stores it. See `PackagedResourceBundle`.
enum MacResources {
    static let bundle: Bundle = PackagedResourceBundle.named("Tsugumi_TsugumiMac")
        ?? .module
}
