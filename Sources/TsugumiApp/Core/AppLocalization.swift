import Foundation

/// User-facing strings of this target, looked up in its own resource bundle
/// (`Resources/<lang>.lproj/Localizable.strings`). The English text is the
/// key, so a missing entry shows English rather than a placeholder.
///
/// Which language is used follows the process, not this type: macOS picks
/// the main bundle's language first and matches sub-bundles to it, which is
/// why `make_app.sh` gives the `.app` its own `en.lproj` / `ja.lproj`.
public enum AppLocalization {
    public static func string(_ key: String.LocalizationValue) -> String {
        String(localized: key, bundle: AppCoreResources.bundle)
    }
}
