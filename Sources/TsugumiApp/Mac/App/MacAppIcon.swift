import AppKit

// The icon is a resource of this target, so it cannot be loaded from the
// presentation library — a target's resource bundle resolves only inside it.
enum MacAppIcon {
    static func load() -> NSImage? {
        guard let url = MacResources.bundle.url(
            forResource: "tsugumi-app-icon",
            withExtension: "png"
        ) else { return nil }
        return NSImage(contentsOf: url)
    }
}
