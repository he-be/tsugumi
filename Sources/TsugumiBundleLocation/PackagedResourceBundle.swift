import Foundation

/// Finds a SwiftPM resource bundle in the layout the app is *shipped* in.
///
/// The accessor SwiftPM generates for `Bundle.module` looks for
/// `<package>_<target>.bundle` beside `Bundle.main.bundleURL`. That is right
/// for a bare executable, where `bundleURL` is the directory the executable
/// sits in — exactly where `swift build` leaves the resource bundles. Inside a
/// `.app` it is wrong twice over: `bundleURL` is the bundle root, and an app
/// bundle may not carry unsealed files there, so `codesign` rejects the
/// layout `Bundle.module` would need.
///
/// The packaged layout puts the resource bundles in `Contents/Resources`
/// instead. That directory is `Bundle.main.resourceURL` for the app process
/// and for the decode service alike: a helper executable in `Contents/MacOS`
/// gets the enclosing `.app` as its main bundle, not its own directory. For a
/// bare executable `resourceURL` is that same executable directory, so this
/// one lookup covers the development layout too and `Bundle.module` is only
/// the fallback — it still matters under `swift test`, where the main bundle
/// is the test harness rather than anything this package built.
public enum PackagedResourceBundle {
    /// The bundle named `name` (without the `.bundle` extension) as shipped,
    /// or `nil` when the running layout is not a packaged one.
    public static func named(_ name: String) -> Bundle? {
        guard let resources = Bundle.main.resourceURL else { return nil }
        return Bundle(url: resources.appendingPathComponent("\(name).bundle",
                                                            isDirectory: true))
    }
}
