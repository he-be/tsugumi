import Foundation

enum AppModelLocation {
    static func defaultURL(for kind: AppModelKind = .defaultKind) -> URL {
        let fileManager = FileManager.default
        let applicationSupport = (try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: false)) ?? fileManager.homeDirectoryForCurrentUser
        return resolve(
            kind: kind,
            explicitURL: nil,
            executableURL: Bundle.main.executableURL,
            currentDirectoryURL: URL(fileURLWithPath: fileManager.currentDirectoryPath,
                                     isDirectory: true),
            applicationSupportURL: applicationSupport,
            fileExists: fileManager.fileExists(atPath:))
    }

    static func resolve(kind: AppModelKind = .defaultKind,
                        explicitURL: URL?,
                        executableURL: URL?,
                        currentDirectoryURL: URL,
                        applicationSupportURL: URL,
                        fileExists: (String) -> Bool) -> URL {
        if let explicitURL {
            return absoluteURL(explicitURL, relativeTo: currentDirectoryURL)
        }
        if let executableURL,
           let root = packageRoot(startingAt: executableURL.deletingLastPathComponent(),
                                  fileExists: fileExists) {
            return installed(in: root.appendingPathComponent("scratch", isDirectory: true),
                             kind: kind, fileExists: fileExists)
        }
        if let root = packageRoot(startingAt: currentDirectoryURL, fileExists: fileExists) {
            return installed(in: root.appendingPathComponent("scratch", isDirectory: true),
                             kind: kind, fileExists: fileExists)
        }
        return installed(in: applicationSupportURL
            .appendingPathComponent("Tsugumi", isDirectory: true),
                         kind: kind, fileExists: fileExists)
    }

    /// The install directory under `container`, preferring the current
    /// `.moepack` name and falling back to the `.gturbo` one the project wrote
    /// before the rename. An install is recognised by its `manifest.json`: the
    /// directory alone can be an empty leftover, and answering with a leftover
    /// would hide a working install sitting next to it. When neither is
    /// present the current name is returned, so a fresh install lands on it.
    private static func installed(in container: URL,
                                  kind: AppModelKind,
                                  fileExists: (String) -> Bool) -> URL {
        let current = container.appendingPathComponent(kind.directoryName, isDirectory: true)
            .standardizedFileURL
        if fileExists(current.appendingPathComponent("manifest.json").path) {
            return current
        }
        let legacy = container.appendingPathComponent(kind.legacyDirectoryName, isDirectory: true)
            .standardizedFileURL
        if fileExists(legacy.appendingPathComponent("manifest.json").path) {
            return legacy
        }
        return current
    }

    private static func absoluteURL(_ url: URL, relativeTo base: URL) -> URL {
        if url.path.hasPrefix("/") {
            return url.standardizedFileURL
        }
        return base.appendingPathComponent(url.path, isDirectory: true).standardizedFileURL
    }

    private static func packageRoot(startingAt start: URL,
                                    fileExists: (String) -> Bool) -> URL? {
        var candidatePath = start.standardizedFileURL.path
        while true {
            let candidate = URL(fileURLWithPath: candidatePath, isDirectory: true)
            let package = candidate.appendingPathComponent("Package.swift").path
            let appSources = candidate.appendingPathComponent(
                "Sources/TsugumiApp/Mac", isDirectory: true).path
            if fileExists(package), fileExists(appSources) {
                return candidate
            }
            let parentPath = (candidatePath as NSString).deletingLastPathComponent
            if parentPath.isEmpty || parentPath == candidatePath { return nil }
            candidatePath = parentPath
        }
    }
}
