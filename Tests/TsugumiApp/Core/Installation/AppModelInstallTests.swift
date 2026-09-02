import Foundation
import Testing
import TsugumiRepackCore

@testable import TsugumiAppCore

@Suite struct AppModelInstallTests {

  @MainActor
  @Test func missingModelCanInstall() {
    let installer = MockModelInstallerClient()
    let directory = temporaryInstallPath("missing")
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: installer)

    #expect(!model.isModelInstalled)
    #expect(model.requiresModelInstallation)
    #expect(model.canInstallModel)
  }

  @MainActor
  @Test func installedModelShowsLoadNotInstall() throws {
    let directory = try makeCompleteModelInstall("installed")
    defer { try? FileManager.default.removeItem(at: directory) }
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())

    #expect(model.isModelInstalled)
    #expect(!model.requiresModelInstallation)
    #expect(!model.canInstallModel)
    #expect(model.canLoadModel)
  }

  @MainActor
  @Test func checkAgainDetectsModelInstalledAfterLaunch() throws {
    let directory = try makeCompleteModelInstall("external-install")
    let stagedDirectory = directory.deletingLastPathComponent()
      .appendingPathComponent("staged-\(UUID().uuidString).moepack")
    try FileManager.default.moveItem(at: directory, to: stagedDirectory)
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())

    #expect(model.requiresModelInstallation)
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.moveItem(at: stagedDirectory, to: directory)

    model.refreshInstallReadiness()

    #expect(model.isModelInstalled)
    #expect(!model.requiresModelInstallation)
    #expect(model.canLoadModel)
  }

  @MainActor
  @Test func checkAgainUsesCurrentModelLocation() throws {
    let initialDirectory = temporaryInstallPath("initial-location")
    let currentDirectory = try makeCompleteModelInstall("current-location")
    defer { try? FileManager.default.removeItem(at: currentDirectory) }
    let model = AppModel(
      modelDirectory: initialDirectory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())

    model.modelPathText = currentDirectory.path
    model.recheckModelAtCurrentLocation()

    #expect(model.modelPathText == currentDirectory.standardizedFileURL.path)
    #expect(model.isModelInstalled)
    #expect(model.canLoadModel)
  }

  @MainActor
  @Test func defaultInstallDescriptorMatchesPinnedAudit() {
    // Repinned to the prebuilt sym install published on Hugging Face; the
    // sizes are the sum of the pinned per-file table.
    let descriptor = AppModelInstallDescriptor.default
    #expect(descriptor.kind == .gemmaQATSym)
    #expect(descriptor.displayName == "Gemma 4 26B-A4B QAT (Vision + MTP)")
    #expect(descriptor.repoID == "mh73772/turbofieldfare-gemma4-qat-sym")
    #expect(descriptor.sourceIndexSHA256 == "7dbbeef0345505798abcf0ac54434116a48c2f1e7aad828071c17a7a871adfe7")
    #expect(descriptor.approximateDownloadBytes == 15_681_261_432)
    #expect(descriptor.installedBytes == 15_681_261_432)
    #expect(descriptor.requiredFreeBytes == 15_681_261_432 + 1_073_741_824)
  }

  @MainActor
  @Test func ornithInstallDescriptorMatchesPinnedAudit() {
    let descriptor = AppModelInstallDescriptor.descriptor(for: .ornith)
    #expect(descriptor.kind == .ornith)
    #expect(descriptor.repoID == "mh73772/turbofieldfare-ornith-oq4e-g64")
    #expect(descriptor.sourceIndexSHA256 == "4280eb9999b17eeb94f45f8ac6ba60510afbf4e1ea5adf32aa83754e68d33bf3")
    // 19.6 GB pack + 503 MB MTP-head sidecar, from the pinned table.
    #expect(descriptor.installedBytes == 20_998_071_775)
    #expect(PrebuiltModelSource.ornith.files.contains {
      $0.path.hasPrefix("mtp-head/")
    })
  }

  @MainActor
  @Test func insufficientSpaceDisablesInstallAndExposesShortfall() {
    let requirement = AppModelInstallRequirement(
      probePath: "/volume",
      requiredBytes: 100,
      availableBytes: 40)
    let installer = MockModelInstallerClient(requirement: requirement)
    let model = AppModel(
      modelDirectory: temporaryInstallPath("space"),
      client: MockLifecycleInferenceClient(),
      installer: installer)

    #expect(model.installReadiness == .insufficientSpace(requirement))
    #expect(model.installRequirement?.shortfallBytes == 60)
    #expect(!model.canInstallModel)
  }

  @MainActor
  @Test func installProgressUpdatesStatusAndByteCounts() async throws {
    let directory = temporaryInstallPath("progress")
    let installer = MockModelInstallerClient(
      events: [
        .checking,
        .copyingPayload(
          reusedBytes: 1,
          downloadedThisRunBytes: 3,
          totalBytes: 10),
      ], holdOpen: true)
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: installer)
    model.installModel()

    try await waitUntil {
      model.installState == .copyingPayload(
        reusedBytes: 1,
        downloadedThisRunBytes: 3,
        totalBytes: 10)
    }
    #expect(model.installDownloadedBytes == 4)
    #expect(model.installTotalBytes == 10)
    #expect(model.installProgressFraction == 0.4)
    #expect(model.presentation.label == AppLocalization.string("Downloading model"))
    model.cancelInstall()
    try await waitUntil { model.installState == .cancelled }
  }

  @MainActor
  @Test func installCompletionStopsUnloaded() async throws {
    let requestedDirectory = temporaryInstallPath("requested")
    let completedDirectory = try makeCompleteModelInstall("complete")
    defer { try? FileManager.default.removeItem(at: completedDirectory) }
    let client = MockLifecycleInferenceClient()
    let installer = MockModelInstallerClient(events: [.installed(completedDirectory)])
    let model = AppModel(
      modelDirectory: requestedDirectory,
      client: client,
      installer: installer)

    model.installModel()
    try await waitUntil {
      model.installState == .installed(modelDirectory: completedDirectory.standardizedFileURL)
    }
    #expect(
      model.installState == .installed(modelDirectory: completedDirectory.standardizedFileURL))
    #expect(model.modelPathText == completedDirectory.standardizedFileURL.path)
    #expect(model.loadState == .notLoaded)
    #expect(model.canLoadModel)
    #expect(client.ensureLoadedCallCount() == 0)
  }

  @MainActor
  @Test func installFailureDoesNotAttemptLoad() async throws {
    struct SyntheticError: Error {}
    let client = MockLifecycleInferenceClient()
    let installer = MockModelInstallerClient(failure: SyntheticError())
    let model = AppModel(
      modelDirectory: temporaryInstallPath("failure"),
      client: client,
      installer: installer)
    model.installModel()

    try await waitUntil {
      if case .failed = model.installState { return true }
      return false
    }
    #expect(model.loadState == .notLoaded)
    #expect(client.ensureLoadedCallCount() == 0)
  }

  @MainActor
  @Test func networkFailureWithSavedProgressLeavesResumeEnabled() async throws {
    struct NetworkFailure: Error {}
    let directory = temporaryInstallPath("network-resume")
    let paths = try makeSavedDownload(at: directory)
    defer { cleanUpSavedDownload(paths) }
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(failure: NetworkFailure()))

    model.installModel()
    try await waitUntil {
      if case .recoverable = model.installState { return true }
      return false
    }

    #expect(model.canInstallModel)
    #expect(model.canDiscardModelDownload)
  }

  @MainActor
  @Test func diskFailureCanResumeAfterSpaceRecheck() async throws {
    let directory = temporaryInstallPath("disk-resume")
    let paths = try makeSavedDownload(at: directory)
    defer { cleanUpSavedDownload(paths) }
    let error = RepackError.diskSpaceInsufficient(
      path: "/volume",
      required: 120,
      available: 45)
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(failure: error))

    model.installModel()
    try await waitUntil {
      if case .recoverable = model.installState { return true }
      return false
    }
    #expect(!model.canInstallModel)

    model.recheckModelAtCurrentLocation()

    #expect(model.canInstallModel)
    #expect(model.canDiscardModelDownload)
  }

  @MainActor
  @Test func invalidSavedDownloadsRemainDiscardOnly() throws {
    let descriptor = AppModelInstallDescriptor.default
    for incompatible in [false, true] {
      let directory = temporaryInstallPath(
        incompatible ? "incompatible-checkpoint" : "corrupt-checkpoint")
      let paths = try makeSavedDownload(at: directory)
      defer { cleanUpSavedDownload(paths) }
      if incompatible {
        try RemoteInstallCheckpoint(
          repoID: "other/model",
          requestedRevision: descriptor.revision,
          resolvedCommit: String(repeating: "a", count: 40),
          sourceIndexSHA256: String(repeating: "b", count: 64),
          planFingerprint: String(repeating: "c", count: 64),
          totalSourceBytes: 1
        ).write(
          to: paths.checkpointFile,
          parentDirectory: paths.parentDirectory)
      } else {
        try Data("{}".utf8).write(
          to: URL(fileURLWithPath: paths.checkpointFile))
      }
      let model = AppModel(
        modelDirectory: directory,
        client: MockLifecycleInferenceClient(),
        installer: RepackModelInstallerClient(descriptor: descriptor))

      #expect(!model.canInstallModel)
      #expect(model.canDiscardModelDownload)
      guard case .failed = model.installReadiness else {
        Issue.record("invalid checkpoint did not fail readiness")
        continue
      }
    }
  }

  @MainActor
  @Test func diskFailureKeepsExactRequirementAndShortfall() async throws {
    let error = RepackError.diskSpaceInsufficient(
      path: "/volume",
      required: 120,
      available: 45)
    let installer = MockModelInstallerClient(failure: error)
    let model = AppModel(
      modelDirectory: temporaryInstallPath("disk-failure"),
      client: MockLifecycleInferenceClient(),
      installer: installer)
    model.installModel()

    try await waitUntil {
      if case .failed = model.installState { return true }
      return false
    }

    let expected = AppModelInstallRequirement(
      probePath: "/volume",
      requiredBytes: 120,
      availableBytes: 45)
    #expect(model.installReadiness == .insufficientSpace(expected))
    #expect(model.installRequirement?.shortfallBytes == 75)
  }

  @MainActor
  @Test func cancelInstallWaitsForAcknowledgementAndAllowsRetry() async throws {
    let installer = MockModelInstallerClient(
      events: [.downloadingMetadata],
      holdOpen: true,
      delayCancellationAcknowledgement: true)
    let model = AppModel(
      modelDirectory: temporaryInstallPath("cancel"),
      client: MockLifecycleInferenceClient(),
      installer: installer)
    model.installModel()
    try await waitUntil { model.installState == .downloadingMetadata }

    model.cancelInstall()
    #expect(installer.cancelCalled)
    try await waitUntil { installer.cancellationAcknowledgementPending }
    #expect(model.installState == .cancelling)
    #expect(!model.canInstallModel)

    await installer.releaseCancellationAcknowledgement()
    try await waitUntil { model.installState == .cancelled }

    #expect(model.loadState == .notLoaded)
    #expect(model.canInstallModel)
  }

  private func temporaryInstallPath(_ tag: String) -> URL {
    FileManager.default.temporaryDirectory
      .appendingPathComponent("tsugumi-app-install-\(tag)-\(UUID().uuidString).moepack")
  }

  private func makeSavedDownload(at directory: URL) throws -> RemoteInstallPaths {
    let paths = try RemoteInstallPaths(outputDirectory: directory.path)
    try FileManager.default.createDirectory(
      atPath: paths.partialDirectory,
      withIntermediateDirectories: true)
    return paths
  }

  private func cleanUpSavedDownload(_ paths: RemoteInstallPaths) {
    for path in [
      paths.finalDirectory,
      paths.partialDirectory,
      paths.checkpointFile,
      paths.lockFile,
    ] {
      try? FileManager.default.removeItem(atPath: path)
    }
  }

  @MainActor
  private func waitUntil(_ predicate: @escaping @MainActor () -> Bool) async throws {
    for _ in 0..<200 {
      if predicate() { return }
      try await Task.sleep(for: .milliseconds(5))
    }
    Issue.record("timed out waiting for condition")
  }

}
