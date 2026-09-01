import Foundation
import TsugumiBundleLocation

/// This target's resource bundle (`Metal/`, `Templates/`), found the way the
/// shipped `.app` stores it and falling back to the SwiftPM accessor. Use this
/// rather than `Bundle.module` for anything that has to work inside the app
/// bundle — the Metal sources are compiled at runtime, so losing them is a
/// hard failure at model load rather than a missing nicety.
enum TsugumiResources {
    static let bundle: Bundle = PackagedResourceBundle.named("Tsugumi_Tsugumi")
        ?? .module
}
