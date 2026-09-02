import Foundation

/// The app's user-facing strings, looked up in this target's resource
/// bundle (`Resources/<lang>.lproj/Localizable.strings`). The English text
/// is the key; a missing entry shows the English.
///
/// Strings from `TsugumiAppCore` (state labels, error messages) arrive
/// already localized through `AppLocalization` and are shown as they are.
func L(_ key: String.LocalizationValue) -> String {
    String(localized: key, bundle: MacResources.bundle)
}
