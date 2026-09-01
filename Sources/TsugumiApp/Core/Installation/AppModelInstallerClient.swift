import Foundation

public protocol AppModelInstallerClient: Sendable {
    var descriptor: AppModelInstallDescriptor { get }
    func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement
    func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error>
    /// Whether an interrupted download left state a new install could resume
    /// from. Drives the "saved download" banner and the discard button.
    func hasPartialInstall(outputDirectory: URL) -> Bool
    func discardPartialInstall(outputDirectory: URL) async throws
    func cancel()
}

public extension AppModelInstallerClient {
    func hasPartialInstall(outputDirectory: URL) -> Bool { false }
}
