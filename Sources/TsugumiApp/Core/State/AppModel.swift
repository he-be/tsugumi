import Foundation
import TsugumiRepackCore
import Observation

@MainActor
@Observable
public final class AppModel {
    public enum RunState: Equatable {
        case idle
        case running
    }

    public var modelPathText: String
    public private(set) var selectedModelKind: AppModelKind = .defaultKind
    /// All conversations, oldest first. Exactly one is selected; at most
    /// one is generating, and it need not be the selected one.
    public private(set) var chats: [AppChatSession]
    public private(set) var selectedChatID: UUID
    /// The chat the running generation streams into; nil while idle.
    private var generatingChat: AppChatSession?
    /// The chat whose text the transcript mailbox currently holds. The
    /// mailbox is a single client-owned channel, so reads must be gated on
    /// this or another chat would display the generating chat's text.
    private var mailboxOwnerChatID: UUID?
    public var runState: RunState = .idle
    public var runtimeOptions = AppRuntimeOptions()
    public var maxNewTokensOverride: Int?
    public var maxContextTokens: Int = 4096
    public var temperature: Double = 1.0
    public var topKEnabled: Bool = true
    public var topK: Int = 64
    public var topPEnabled: Bool = true
    public var topP: Double = 0.95
    public var thinkingEnabled: Bool = false
    /// How the chat uses the web tools. Only Gemma declares them
    /// (`webSearchAvailable`); for another model the mode is kept but
    /// ignored.
    public var webSearchMode: AppWebSearchMode = .off
    /// Keys and limits for the web tools, one file for the app. Edited in
    /// the Inspector; `saveWebSearchConfiguration` writes it back.
    public var webSearchConfiguration: WebSearchConfiguration
    public private(set) var newlineShortcut: AppNewlineShortcut = .return
    public private(set) var sentPromptBehavior: AppSentPromptBehavior = .keep
    public var diagnostics: AppDiagnostics?
    public var error: AppInferenceError?
    public var installState: AppModelInstallState = .idle
    public private(set) var installETAPresentation: DownloadETAPresentation = .hidden
    public private(set) var installETAText: String?
    public private(set) var installReadiness: AppModelInstallReadiness = .checking
    public private(set) var installationStatus: AppModelInstallationStatus

    public var loadState: AppModelLoadState = .notLoaded
    public private(set) var loadedRuntimeKey: AppLoadedRuntimeKey?
    public private(set) var phase: AppGenerationPhase = .idle
    public private(set) var liveTokenCount: Int = 0
    public private(set) var liveElapsedDecodeSeconds: Double = 0
    public private(set) var livePrefillDone: Int = 0
    public private(set) var livePrefillTotal: Int = 0
    public private(set) var liveMemoryBytes: UInt64?
    public private(set) var isCancellationPending: Bool = false

    private let client: any AppInferenceClient
    /// Builds the installer for one model kind. The default downloads the
    /// prebuilt install; tests inject a fixed mock through the `installer`
    /// init parameter, which pins this to a constant.
    private let installerProvider: (AppModelKind) -> any AppModelInstallerClient
    private var installer: any AppModelInstallerClient
    private var runTask: Task<Void, Never>?
    private var loadTask: Task<Void, Never>?
    private var installTask: Task<Void, Never>?
    private var unloadTask: Task<Void, Never>?
    private var loadGeneration: UInt64 = 0
    private var unloadGeneration: UInt64 = 0
    private var installGeneration: UInt64 = 0
    private var pendingExplicitLoadRuntimeKey: AppLoadedRuntimeKey?
    private var activeRunRuntimeKey: AppLoadedRuntimeKey?
    private var hasHandledTerminalEvent = false
    /// The tool loop of the running turn: which executor answers the calls,
    /// the calls the current round has asked for so far, how many rounds
    /// have run, and the task executing the calls between rounds.
    private let toolExecutorProvider: (WebSearchConfiguration) throws -> any AppToolExecutor
    private let webSearchConfigurationURL: URL?
    private var activeToolExecutor: (any AppToolExecutor)?
    private var activeWebSearchMode: AppWebSearchMode = .off
    private var activeSystemPrompt: String?
    private var pendingToolCalls: [AppToolCall] = []
    private var toolRoundsUsed = 0
    private var toolTask: Task<Void, Never>?
    private let memorySampler: AppMemorySampler
    private let settingsPersistenceEnabled: Bool
    private let chatStore: AppChatStore?
    private var chatSaveTask: Task<Void, Never>?
    /// Draft edits save behind this debounce; structural changes and
    /// finished turns save immediately. Tests shorten it.
    nonisolated(unsafe) static var chatSaveDebounceNanos: UInt64 = 1_000_000_000
    private let installETAClock: SuspendingClock
    private let installETAOrigin: SuspendingClock.Instant
    private var installETAEstimator = DownloadETAEstimator()

    public init(modelDirectory: URL? = nil,
                client: any AppInferenceClient = RealInferenceClient(),
                installer: (any AppModelInstallerClient)? = nil,
                memorySampler: AppMemorySampler = AppMemorySampler(),
                settingsPersistenceEnabled: Bool = false,
                chatStore: AppChatStore? = nil,
                webSearchConfigurationURL: URL? = nil,
                toolExecutorProvider: ((WebSearchConfiguration) throws -> any AppToolExecutor)? = nil) {
        self.chatStore = chatStore
        let configurationURL = webSearchConfigurationURL
            ?? (settingsPersistenceEnabled ? WebSearchConfigurationStore.defaultFileURL : nil)
        self.webSearchConfigurationURL = configurationURL
        self.webSearchConfiguration = configurationURL.map { WebSearchConfigurationStore.load(from: $0) }
            ?? WebSearchConfiguration()
        self.toolExecutorProvider = toolExecutorProvider
            ?? { try Self.makeToolExecutor(configuration: $0) }
        if let persisted = chatStore?.load(), !persisted.chats.isEmpty {
            let sessions = persisted.chats.map { $0.makeSession() }
            self.chats = sessions
            let index = min(max(persisted.selectedChatIndex, 0), sessions.count - 1)
            self.selectedChatID = sessions[index].id
            // No generation has run yet, so the transcript mailbox is empty;
            // owning it would mask the restored text with that emptiness.
            self.mailboxOwnerChatID = nil
        } else {
            let firstChat = AppChatSession()
            self.chats = [firstChat]
            self.selectedChatID = firstChat.id
            self.mailboxOwnerChatID = firstChat.id
        }
        let kind = modelDirectory.flatMap(AppModelKind.probe(modelDirectory:))
            ?? .defaultKind
        let directory = (modelDirectory ?? AppModelLocation.defaultURL(for: kind))
            .standardizedFileURL
        let installETAClock = SuspendingClock()
        let settings = settingsPersistenceEnabled
            ? MacAppSettingsFileStore.loadOrCreate(
                forModelDirectory: directory,
                defaults: MacAppSettings.defaults(for: kind))
            : MacAppSettings.defaults(for: kind)
        self.modelPathText = directory.path
        self.selectedModelKind = kind
        self.runtimeOptions = AppRuntimeOptions(
            expertCacheSlots: settings.expertCacheSlots,
            prefillEnabled: settings.prefillEnabled,
            mtpEnabled: settings.mtpEnabled)
        self.maxContextTokens = settings.contextTokens
        self.temperature = settings.temperature
        self.topKEnabled = settings.topKEnabled
        self.topK = settings.topK
        self.topPEnabled = settings.topPEnabled
        self.topP = settings.topP
        self.thinkingEnabled = settings.thinkingEnabled
        self.webSearchMode = settings.webSearchMode
        self.newlineShortcut = settings.newlineShortcut
        self.sentPromptBehavior = settings.sentPromptBehavior
        let provider: (AppModelKind) -> any AppModelInstallerClient
        if let installer {
            provider = { _ in installer }
        } else {
            provider = { PrebuiltModelInstallerClient(kind: $0) }
        }
        self.installerProvider = provider
        self.installer = provider(kind)
        self.installationStatus = AppModelInstallationProbe.status(
            at: directory, descriptor: provider(kind).descriptor)
        self.client = client
        self.memorySampler = memorySampler
        self.settingsPersistenceEnabled = settingsPersistenceEnabled
        self.installETAClock = installETAClock
        self.installETAOrigin = installETAClock.now
        refreshInstallReadiness()
    }

    public var isRunning: Bool { runState == .running }

    public var selectedChat: AppChatSession {
        chats.first(where: { $0.id == selectedChatID }) ?? chats[0]
    }

    public var generatingChatID: UUID? { generatingChat?.id }

    public var isSelectedChatGenerating: Bool {
        isRunning && generatingChat === selectedChat
    }

    /// While one chat generates, every other chat is view-only: no sending,
    /// no editing, no clearing — switching back is the way to interact.
    public var isSelectedChatReadOnly: Bool {
        isRunning && generatingChat !== selectedChat
    }

    // The single-conversation surface the views and tests were built on;
    // each member now reads or writes the selected chat.
    public var promptText: String {
        get { selectedChat.promptText }
        set {
            selectedChat.promptText = newValue
            scheduleChatPersist()
        }
    }

    public var conversationTurns: [AppChatTurn] { selectedChat.conversationTurns }

    public var outputPromptText: String { selectedChat.outputPromptText }

    public var outputImagePaths: [String] { selectedChat.outputImagePaths }

    public var outputText: String {
        get { selectedChat.outputText }
        set { selectedChat.outputText = newValue }
    }

    public var outputReasoningText: String { selectedChat.outputReasoningText }

    public var outputToolTrace: [AppToolTraceEntry] { selectedChat.outputToolTrace }

    public var outputContinuationTurns: [AppChatTurn] { selectedChat.outputContinuationTurns }

    public var attachedImagePaths: [String] {
        get { selectedChat.attachedImagePaths }
        set {
            selectedChat.attachedImagePaths = newValue
            scheduleChatPersist()
        }
    }

    private func persistChatsNow() {
        guard let chatStore else { return }
        chatSaveTask?.cancel()
        chatSaveTask = nil
        let selectedIndex = chats.firstIndex(where: { $0.id == selectedChatID }) ?? 0
        try? chatStore.save(PersistedChats(
            selectedChatIndex: selectedIndex,
            chats: chats.map(PersistedChat.init)))
    }

    private func scheduleChatPersist() {
        guard chatStore != nil else { return }
        chatSaveTask?.cancel()
        chatSaveTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: Self.chatSaveDebounceNanos)
            guard !Task.isCancelled else { return }
            self?.persistChatsNow()
        }
    }

    /// Opens a fresh conversation, reselecting an existing empty one
    /// instead of piling up blanks.
    public func newChat() {
        if let existing = chats.first(where: {
            $0.isEmpty && $0 !== generatingChat && responsePlainText(of: $0).isEmpty
        }) {
            selectedChatID = existing.id
            persistChatsNow()
            return
        }
        let chat = AppChatSession()
        chats.append(chat)
        selectedChatID = chat.id
        persistChatsNow()
    }

    /// Switching is allowed while another chat generates; the newly
    /// selected chat is then read-only (`isSelectedChatReadOnly`).
    public func selectChat(_ id: UUID) {
        guard chats.contains(where: { $0.id == id }) else { return }
        selectedChatID = id
        persistChatsNow()
    }

    public func canDeleteChat(_ id: UUID) -> Bool {
        generatingChat?.id != id
    }

    public func deleteChat(_ id: UUID) {
        guard canDeleteChat(id),
              let index = chats.firstIndex(where: { $0.id == id }) else { return }
        chats.remove(at: index)
        if mailboxOwnerChatID == id { mailboxOwnerChatID = nil }
        if chats.isEmpty { chats = [AppChatSession()] }
        if !chats.contains(where: { $0.id == selectedChatID }) {
            selectedChatID = chats[min(index, chats.count - 1)].id
        }
        persistChatsNow()
    }

    public var isModelAvailable: Bool { loadState.isReady }

    public var hasStaleLoadedRuntime: Bool {
        guard loadState.isReady, let loadedRuntimeKey else { return false }
        return loadedRuntimeKey != currentRuntimeKey
    }

    public var canLoadModel: Bool {
        isModelInstalled && !isRunning && (loadState == .notLoaded || loadState.isFailed)
    }

    public var canCancelLoad: Bool {
        if case .loading = loadState { return loadTask != nil }
        return false
    }

    public var canReloadModel: Bool {
        isModelInstalled && !isRunning && loadState.isReady && hasStaleLoadedRuntime
    }

    public var canUnloadModel: Bool {
        isModelInstalled && !isRunning && loadState.isReady
    }

    public var isModelInstalled: Bool { installationStatus == .complete }

    public var requiresModelInstallation: Bool { !isModelInstalled }

    public var installDescriptor: AppModelInstallDescriptor { installer.descriptor }

    public var installRequirement: AppModelInstallRequirement? {
        installReadiness.requirement
    }

    public var isInstallingModel: Bool { installState.isInstalling }

    public var canInstallModel: Bool {
        guard case .ready = installReadiness else { return false }
        return !isRunning && !loadState.isLoading && !isInstallingModel
            && requiresModelInstallation
    }

    public var canCancelInstall: Bool { installState.canCancel }

    public var installDownloadedBytes: UInt64? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState else {
            return nil
        }
        let addition = reused.addingReportingOverflow(downloaded)
        return min(addition.overflow ? UInt64.max : addition.partialValue, total)
    }

    public var installTotalBytes: UInt64? {
        guard case .copyingPayload(_, _, let total) = installState else {
            return nil
        }
        return total
    }

    public var installReusedBytes: UInt64? {
        guard case .copyingPayload(let reused, _, _) = installState else {
            return nil
        }
        return reused
    }

    public var installDownloadedThisRunBytes: UInt64? {
        guard case .copyingPayload(_, let downloaded, _) = installState else {
            return nil
        }
        return downloaded
    }

    public var installProgressFraction: Double? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState,
              total > 0 else {
            return nil
        }
        let addition = reused.addingReportingOverflow(downloaded)
        let done = addition.overflow ? UInt64.max : addition.partialValue
        return min(max(Double(done) / Double(total), 0), 1)
    }

    public var installPhaseLabel: String {
        switch installState {
        case .idle: return AppLocalization.string("Model required")
        case .checking: return AppLocalization.string("Checking installation")
        case .downloadingMetadata: return AppLocalization.string("Downloading metadata")
        case .planning: return AppLocalization.string("Planning installation")
        case .reservingOutput: return AppLocalization.string("Reserving storage")
        case .copyingPayload: return AppLocalization.string("Downloading model")
        case .hashingOutput(let file): return AppLocalization.string("Verifying \(file)")
        case .finalizing: return AppLocalization.string("Finalizing installation")
        case .cancelling: return AppLocalization.string("Cancelling")
        case .discarding: return AppLocalization.string("Discarding download")
        case .cancelled: return AppLocalization.string("Download paused")
        case .recoverable: return AppLocalization.string("Saved download needs attention")
        case .installed: return AppLocalization.string("Model installed")
        case .failed: return AppLocalization.string("Installation failed")
        }
    }

    public var canRun: Bool {
        !isRunning && isModelAvailable && !loadState.isLoading
            && !hasStaleLoadedRuntime
            && !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public var canCancel: Bool { isRunning && !isCancellationPending }

    public var hasOutputTranscript: Bool {
        !conversationTurns.isEmpty || !outputPromptText.isEmpty || !outputText.isEmpty
    }

    public var outputResponsePlainText: String {
        responsePlainText(of: selectedChat)
    }

    private func responsePlainText(of chat: AppChatSession) -> String {
        guard chat.id == mailboxOwnerChatID,
              let mailbox = generationTranscriptMailbox else {
            return chat.outputText
        }
        return mailbox.completeText
    }

    public var outputConversationPlainText: String {
        // Tool rounds are the model's working, not the conversation.
        var sections = conversationTurns.compactMap { turn -> String? in
            switch turn.role {
            case .user: "You:\n\(turn.text)"
            case .assistant: turn.toolCalls.isEmpty ? "Answer:\n\(turn.text)" : nil
            case .tool: nil
            }
        }
        if !outputPromptText.isEmpty {
            sections.append("You:\n\(outputPromptText)")
        }
        let response = outputResponsePlainText
        if !response.isEmpty {
            sections.append("Answer:\n\(response)")
        }
        return sections.joined(separator: "\n\n")
    }

    public var liveTokensPerSecond: Double {
        liveElapsedDecodeSeconds > 0 ? Double(liveTokenCount) / liveElapsedDecodeSeconds : 0
    }

    public var presentation: AppPresentationState {
        AppPresentationState.resolve(AppPresentationSnapshot(
            requiresInstallation: requiresModelInstallation,
            installState: installState,
            installReadiness: installReadiness,
            loadState: loadState,
            hasStaleRuntime: hasStaleLoadedRuntime,
            isRunning: isRunning,
            isGenerationCancellationPending: isCancellationPending,
            generationPhase: phase,
            livePrefillDone: livePrefillDone,
            livePrefillTotal: livePrefillTotal,
            lastStopReason: diagnostics?.stopReason))
    }

    public var currentProcessMemoryBytes: UInt64? {
        guard loadState.isReady || isRunning else { return nil }
        if let reporter = client as? any AppInferenceMemoryReporting,
           let bytes = reporter.currentInferenceMemoryBytes {
            return bytes
        }
        return memorySampler.sample()
    }

    public var generationTranscriptMailbox: GenerationTranscriptMailbox? {
        (client as? any AppInferenceTranscriptReporting)?.generationTranscriptMailbox
    }

    /// The live mailbox, but only when the selected chat is the one whose
    /// text it holds; the transcript view of any other chat must not drain
    /// another conversation's stream.
    public var selectedChatTranscriptMailbox: GenerationTranscriptMailbox? {
        selectedChat.id == mailboxOwnerChatID ? generationTranscriptMailbox : nil
    }

    private var currentRuntimeKey: AppLoadedRuntimeKey {
        AppLoadedRuntimeKey(modelDirectory: URL(fileURLWithPath: modelPathText),
                            maxContextTokens: maxContextTokens,
                            options: runtimeOptions,
                            forceLogitsHead: currentForceLogitsHead)
    }

    private var currentForceLogitsHead: Bool {
        temperature != 0
    }

    /// Switches to the other shipped checkpoint: its default install
    /// location, its persisted settings, its installer.
    public func selectModel(_ kind: AppModelKind) {
        guard !isRunning else { return }
        guard kind != selectedModelKind else { return }
        setModelURL(AppModelLocation.defaultURL(for: kind), kind: kind)
    }

    public func setModelURL(_ url: URL) {
        setModelURL(url, kind: nil)
    }

    private func setModelURL(_ url: URL, kind: AppModelKind?) {
        guard !isRunning else { return }
        let path = url.standardizedFileURL.path
        guard path != modelPathText || (kind != nil && kind != selectedModelKind) else { return }

        modelPathText = path
        selectedModelKind = kind
            ?? AppModelKind.probe(modelDirectory: url.standardizedFileURL)
            ?? selectedModelKind
        installer.cancel()
        installer = installerProvider(selectedModelKind)
        for chat in chats { chat.attachedImagePaths = [] }
        persistChatsNow()
        applyPersistedSettings(
            forModelDirectory: URL(fileURLWithPath: path, isDirectory: true))
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        installGeneration &+= 1
        installTask?.cancel()
        installer.cancel()
        installTask = nil
        resetInstallETA()
        installState = .idle
        pendingExplicitLoadRuntimeKey = nil
        activeRunRuntimeKey = nil
        loadedRuntimeKey = nil
        loadState = .notLoaded
        diagnostics = nil
        error = nil
        phase = .idle
        installationStatus = AppModelInstallationProbe.status(
            at: URL(fileURLWithPath: path), descriptor: installer.descriptor)
        refreshInstallReadiness()

        if let lifecycle = client as? AppModelLifecycleClient {
            unloadGeneration &+= 1
            let generation = unloadGeneration
            let task = Task { [weak self, lifecycle] in
                await lifecycle.unload()
                self?.clearUnloadTask(generation: generation)
            }
            unloadTask = task
        }
    }

    public func loadModel() {
        guard canLoadModel else { return }
        beginLoad()
    }

    public func perform(_ action: AppModelAction) {
        switch action {
        case .install: installModel()
        case .cancelInstall: cancelInstall()
        case .load, .retryLoad: loadModel()
        case .cancelLoad: cancelLoad()
        case .reload: reloadModel()
        case .unload: unloadModel()
        }
    }

    public func setNewlineShortcut(_ shortcut: AppNewlineShortcut) {
        guard newlineShortcut != shortcut else { return }
        newlineShortcut = shortcut
        persistSettings()
    }

    public func setSentPromptBehavior(_ behavior: AppSentPromptBehavior) {
        guard sentPromptBehavior != behavior else { return }
        sentPromptBehavior = behavior
        persistSettings()
    }

    public func reloadModel() {
        guard canReloadModel else { return }
        beginLoad()
    }

    private func beginLoad() {
        guard let lifecycle = client as? AppModelLifecycleClient else {
            loadState = .failed(.modelLoadFailed("This client has no model load lifecycle."))
            return
        }
        let directory = URL(fileURLWithPath: modelPathText)
        let maxContext = maxContextTokens
        let options = runtimeOptions
        let forceLogitsHead = currentForceLogitsHead
        let runtimeKey = AppLoadedRuntimeKey(modelDirectory: directory,
                                             maxContextTokens: maxContext,
                                             options: options,
                                             forceLogitsHead: forceLogitsHead)
        let pendingUnload = unloadTask
        loadGeneration &+= 1
        let generation = loadGeneration
        pendingExplicitLoadRuntimeKey = runtimeKey
        error = nil
        loadState = .loading(.validatingDirectory)
        loadTask = Task.detached { [weak self, lifecycle, pendingUnload] in
            do {
                await pendingUnload?.value
                try Task.checkCancellation()
                try await lifecycle.ensureLoaded(modelDirectory: directory,
                                                 maxContextTokens: maxContext,
                                                 options: options,
                                                 forceLogitsHead: forceLogitsHead) { [weak self] state in
                    Task { @MainActor in
                        self?.applyLoadState(state, generation: generation)
                    }
                }
            } catch is CancellationError {
            } catch let appError as AppInferenceError {
                await self?.applyLoadState(.failed(appError), generation: generation)
            } catch {
                await self?.applyLoadState(
                    .failed(.modelLoadFailed("\(error)")),
                    generation: generation)
            }
            await self?.clearLoadTask(generation: generation)
        }
    }

    public func cancelLoad() {
        guard canCancelLoad, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .cancelling
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func unloadModel() {
        guard canUnloadModel, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .unloading
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.liveMemoryBytes = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func installModel() {
        guard !isRunning, !loadState.isLoading, !isInstallingModel,
              requiresModelInstallation else {
            return
        }
        refreshInstallReadiness()
        guard canInstallModel else { return }
        installTask?.cancel()
        installer.cancel()
        resetInstallETA()
        let outputDirectory = URL(fileURLWithPath: modelPathText)
        installGeneration &+= 1
        let generation = installGeneration
        installState = .checking
        installTask = Task { [weak self, installer] in
            do {
                for try await event in installer.installDefaultModel(outputDirectory: outputDirectory) {
                    guard let self else { return }
                    self.applyInstallEvent(event, generation: generation)
                }
                self?.finishInstallStream(generation: generation)
            } catch is CancellationError {
                self?.finishInstallCancellation(generation: generation)
            } catch {
                self?.finishInstallFailure(error, generation: generation)
            }
        }
    }

    public func cancelInstall() {
        guard canCancelInstall else { return }
        installState = .cancelling
        installer.cancel()
    }

    public var hasPartialModelDownload: Bool {
        installer.hasPartialInstall(outputDirectory: URL(fileURLWithPath: modelPathText))
    }

    public var canDiscardModelDownload: Bool {
        hasPartialModelDownload && !isInstallingModel && !isRunning
    }

    public func discardModelDownload() {
        guard canDiscardModelDownload else { return }
        let outputDirectory = URL(fileURLWithPath: modelPathText)
        installGeneration &+= 1
        let generation = installGeneration
        installState = .discarding
        installTask = Task { [weak self, installer] in
            do {
                try await installer.discardPartialInstall(
                    outputDirectory: outputDirectory)
                guard let self, generation == self.installGeneration else { return }
                self.installTask = nil
                self.installState = .idle
                self.refreshInstallReadiness()
            } catch {
                self?.finishInstallFailure(error, generation: generation)
            }
        }
    }

    public func refreshInstallReadiness() {
        refreshInstallReadiness(
            at: URL(fileURLWithPath: modelPathText, isDirectory: true).standardizedFileURL)
    }

    public func recheckModelAtCurrentLocation() {
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        modelPathText = directory.path
        refreshInstallReadiness(at: directory)
    }

    private func refreshInstallReadiness(at outputDirectory: URL) {
        installationStatus = AppModelInstallationProbe.status(
            at: outputDirectory,
            descriptor: installer.descriptor)
        guard !isModelInstalled else { return }
        installReadiness = .checking
        do {
            let requirement = try installer.checkInstallRequirement(
                outputDirectory: outputDirectory)
            installReadiness = requirement.canInstall
                ? .ready(requirement)
                : .insufficientSpace(requirement)
        } catch {
            installReadiness = .failed("\(error)")
        }
    }

    private func applyInstallEvent(_ event: AppModelInstallEvent, generation: UInt64) {
        guard generation == installGeneration else { return }
        switch event {
        case .checking:
            resetInstallETA()
            installState = .checking
        case .downloadingMetadata:
            resetInstallETA()
            installState = .downloadingMetadata
        case .planning:
            resetInstallETA()
            installState = .planning
        case .reservingOutput:
            resetInstallETA()
            installState = .reservingOutput
        case .copyingPayload(let reused, let downloadedThisRun, let total):
            installState = .copyingPayload(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
            updateInstallETA(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
        case .hashingOutput(let file):
            resetInstallETA()
            installState = .hashingOutput(file)
        case .finalizing:
            resetInstallETA()
            installState = .finalizing
        case .installed(let directory):
            resetInstallETA()
            let directory = directory.standardizedFileURL
            installationStatus = AppModelInstallationProbe.status(
                at: directory,
                descriptor: installer.descriptor)
            guard installationStatus == .complete else {
                finishInstallFailure(
                    RepackError.configurationInvalid(detail: "completed install did not pass metadata validation"),
                    generation: generation)
                return
            }
            installState = .installed(modelDirectory: directory)
            installTask = nil
            modelPathText = directory.path
            loadState = .notLoaded
        }
    }

    private func finishInstallStream(generation: UInt64) {
        guard generation == installGeneration, installTask != nil else { return }
        if installState == .cancelling {
            finishInstallCancellation(generation: generation)
        } else if !isModelInstalled {
            finishInstallFailure(
                RepackError.configurationInvalid(detail: "installer ended before completion"),
                generation: generation)
        }
    }

    private func finishInstallCancellation(generation: UInt64) {
        guard generation == installGeneration else { return }
        installTask = nil
        installState = .cancelled
        resetInstallETA()
        refreshInstallReadiness()
    }

    private func updateInstallETA(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64
    ) {
        let observation = DownloadETAObservation(
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: downloadedThisRunBytes,
            totalBytes: totalBytes)
        let timestamp = installETATimestamp
        setInstallETAPresentation(
            installETAEstimator.update(observation, timestamp: timestamp))
    }

    private var installETATimestamp: Double {
        let components = installETAOrigin.duration(to: installETAClock.now).components
        return Double(components.seconds)
            + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }

    private func resetInstallETA() {
        installETAEstimator.reset()
        installETAPresentation = .hidden
        installETAText = nil
    }

    private func setInstallETAPresentation(
        _ presentation: DownloadETAPresentation
    ) {
        installETAPresentation = presentation
        installETAText = DownloadETAFormatter.string(for: presentation)
    }

    private func applyPersistedSettings(forModelDirectory modelDirectory: URL) {
        let defaults = MacAppSettings.defaults(for: selectedModelKind)
        let settings = settingsPersistenceEnabled
            ? MacAppSettingsFileStore.loadOrCreate(
                forModelDirectory: modelDirectory, defaults: defaults)
            : defaults
        runtimeOptions = AppRuntimeOptions(
            expertCacheSlots: settings.expertCacheSlots,
            prefillEnabled: settings.prefillEnabled,
            mtpEnabled: settings.mtpEnabled)
        maxContextTokens = settings.contextTokens
        temperature = settings.temperature
        topKEnabled = settings.topKEnabled
        topK = settings.topK
        topPEnabled = settings.topPEnabled
        topP = settings.topP
        thinkingEnabled = settings.thinkingEnabled
        webSearchMode = settings.webSearchMode
        newlineShortcut = settings.newlineShortcut
        sentPromptBehavior = settings.sentPromptBehavior
    }

    private func persistSettings() {
        guard settingsPersistenceEnabled else { return }
        let settings = MacAppSettings(
            contextTokens: maxContextTokens,
            expertCacheSlots: runtimeOptions.expertCacheSlots,
            temperature: temperature,
            topKEnabled: topKEnabled,
            topK: topK,
            topPEnabled: topPEnabled,
            topP: topP,
            prefillEnabled: runtimeOptions.prefillEnabled,
            mtpEnabled: runtimeOptions.mtpEnabled,
            thinkingEnabled: thinkingEnabled,
            newlineShortcut: newlineShortcut,
            sentPromptBehavior: sentPromptBehavior,
            webSearchMode: webSearchMode)
        let modelDirectory = URL(fileURLWithPath: modelPathText, isDirectory: true)
        try? MacAppSettingsFileStore.save(
            settings,
            forModelDirectory: modelDirectory)
    }

    private func finishInstallFailure(_ error: Error, generation: UInt64) {
        guard generation == installGeneration else { return }
        installTask = nil
        resetInstallETA()
        let hasSavedDownload = hasPartialModelDownload
        installState = hasSavedDownload ? .recoverable("\(error)") : .failed("\(error)")
        if let repackError = error as? RepackError,
           case .diskSpaceInsufficient(let path, let required, let available) = repackError {
            let requirement = AppModelInstallRequirement(probePath: path,
                                                          requiredBytes: required,
                                                          availableBytes: available)
            installReadiness = .insufficientSpace(requirement)
        } else {
            refreshInstallReadiness()
            if hasSavedDownload {
                installState = .recoverable("\(error)")
            }
        }
    }

    func applyLoadState(_ state: AppModelLoadState) {
        applyLoadState(state, generation: loadGeneration)
    }

    private func applyLoadState(_ state: AppModelLoadState, generation: UInt64) {
        guard generation == loadGeneration else { return }
        if case .ready(let directory, _) = state,
           directory.standardizedFileURL.path
            != URL(fileURLWithPath: modelPathText).standardizedFileURL.path {
            return
        }
        loadState = state
        switch state {
        case .notLoaded:
            loadedRuntimeKey = nil
        case .loading, .cancelling, .unloading:
            break
        case .ready(_, let seconds):
            loadedRuntimeKey = pendingExplicitLoadRuntimeKey
                ?? activeRunRuntimeKey
                ?? currentRuntimeKey
            pendingExplicitLoadRuntimeKey = nil
            _ = seconds
        case .failed(let loadError):
            pendingExplicitLoadRuntimeKey = nil
            error = loadError
        }
    }

    public func clearOutput() {
        guard !isRunning else { return }
        let chat = selectedChat
        chat.conversationTurns = []
        chat.outputPromptText = ""
        chat.outputImagePaths = []
        chat.outputText = ""
        chat.outputReasoningText = ""
        chat.outputContinuationTurns = []
        chat.outputToolTrace = []
        if chat.id == mailboxOwnerChatID {
            generationTranscriptMailbox?.reset()
            mailboxOwnerChatID = nil
        }
        diagnostics = nil
        error = nil
        persistChatsNow()
    }

    /// Moves the chat's finished live turn into its `conversationTurns` so
    /// the next request resends it as history. A turn whose answer never
    /// produced text or reasoning is dropped instead — resending a user
    /// turn with no assistant turn after it would redraw a conversation the
    /// model never saw.
    private func foldCompletedTurnIntoHistory(of chat: AppChatSession) {
        let response = responsePlainText(of: chat)
        let reasoning = chat.id == mailboxOwnerChatID
            ? (generationTranscriptMailbox?.completeReasoningText
               ?? chat.outputReasoningText)
            : chat.outputReasoningText
        guard !chat.outputPromptText.isEmpty else { return }
        if !response.isEmpty || !reasoning.isEmpty {
            chat.conversationTurns.append(AppChatTurn(role: .user,
                                                      text: chat.outputPromptText,
                                                      imagePaths: chat.outputImagePaths))
            // The tool rounds sit between the question and the answer, as
            // the model saw them; the next request redraws them the same way.
            chat.conversationTurns.append(contentsOf: chat.outputContinuationTurns)
            chat.conversationTurns.append(AppChatTurn(role: .assistant,
                                                      text: response,
                                                      reasoningText: reasoning))
        }
        chat.outputPromptText = ""
        chat.outputImagePaths = []
        chat.outputText = ""
        chat.outputReasoningText = ""
        chat.outputContinuationTurns = []
        chat.outputToolTrace = []
    }

    public func attachImages(_ paths: [String]) {
        guard !isRunning, selectedModelKind.supportsVision else { return }
        var merged = attachedImagePaths
        for path in paths where !merged.contains(path) {
            merged.append(path)
        }
        attachedImagePaths = Array(merged.prefix(4))
    }

    public func removeAttachedImage(_ path: String) {
        attachedImagePaths.removeAll { $0 == path }
    }

    public func run() {
        guard canRun else { return }
        let chat = selectedChat
        foldCompletedTurnIntoHistory(of: chat)
        let mode = effectiveWebSearchMode
        var executor: (any AppToolExecutor)?
        let systemPrompt: String?
        let request: AppGenerationRequest
        if mode != .off {
            guard webSearchConfiguration.resolved().canUseTools else {
                error = .invalidRequest(
                    AppLocalization.string("Web search needs a Serper or Brave API key, or a local Wikipedia index. Add one in the Inspector, or turn Web search off."))
                return
            }
        }
        do {
            if mode != .off {
                executor = try toolExecutorProvider(webSearchConfiguration)
            }
            systemPrompt = executor.map { makeSystemPrompt(for: $0, mode: mode) }
            request = try makeFirstRoundRequest(mode: mode, executor: executor,
                                                systemPrompt: systemPrompt)
        } catch let appError as AppInferenceError {
            error = appError
            return
        } catch {
            let appError = AppInferenceError.unknown("\(error)")
            self.error = appError
            return
        }
        persistSettings()

        generationTranscriptMailbox?.reset()
        mailboxOwnerChatID = chat.id
        generatingChat = chat
        chat.outputPromptText = request.prompt
        chat.outputImagePaths = request.imagePaths
        chat.outputText = ""
        chat.outputReasoningText = ""
        chat.outputContinuationTurns = []
        chat.outputToolTrace = []
        diagnostics = nil
        error = nil
        activeToolExecutor = executor
        activeWebSearchMode = mode
        activeSystemPrompt = systemPrompt
        pendingToolCalls = []
        toolRoundsUsed = 0
        activeRunRuntimeKey = AppLoadedRuntimeKey(
            modelDirectory: request.modelDirectory,
            maxContextTokens: request.maxContextTokens,
            options: request.runtimeOptions,
            forceLogitsHead: !request.isPureGreedy)
        isCancellationPending = false
        liveMemoryBytes = nil
        runState = .running
        if sentPromptBehavior == .clear {
            chat.promptText = ""
        }
        // The images are baked into the request that just left; keeping them
        // attached would silently resend them with the next prompt.
        chat.attachedImagePaths = []
        // The fold and the prompt snapshot are worth surviving a crash even
        // if the answer never lands.
        persistChatsNow()
        startRound(request)
    }

    /// One generation of the running turn. A plain turn has exactly one; a
    /// tool loop has one per round.
    private func startRound(_ request: AppGenerationRequest) {
        hasHandledTerminalEvent = false
        liveTokenCount = 0
        liveElapsedDecodeSeconds = 0
        livePrefillDone = 0
        livePrefillTotal = 0
        phase = .prefill
        runTask = Task.detached { [weak self, client, request] in
            guard let self else { return }
            do {
                for try await event in client.generate(request) {
                    await self.apply(event)
                }
            } catch let appError as AppInferenceError {
                await self.finishStreamFailure(appError)
            } catch {
                await self.finishStreamFailure(.unknown("\(error)"))
            }
        }
    }

    /// The round ended on tool calls: run them, append the assistant turn
    /// and its results to the continuation, and prefill again. Every wait
    /// on the network is on `toolTask`, which `cancel` can stop.
    private func continueToolLoop(after diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        let chat = generatingChat ?? selectedChat
        let calls = pendingToolCalls
        pendingToolCalls = []
        let reasoning = chat.id == mailboxOwnerChatID
            ? (generationTranscriptMailbox?.completeReasoningText ?? chat.outputReasoningText)
            : chat.outputReasoningText
        chat.outputContinuationTurns.append(AppChatTurn(
            role: .assistant,
            text: responsePlainText(of: chat),
            reasoningText: reasoning,
            toolCalls: calls))
        toolRoundsUsed += 1
        phase = .tools
        persistChatsNow()
        guard let executor = activeToolExecutor else {
            finishTerminalRun()
            return
        }
        toolTask = Task { [weak self] in
            for call in calls {
                if Task.isCancelled { return }
                let result = await executor.execute(call)
                if Task.isCancelled { return }
                guard let self else { return }
                self.recordToolResult(result, for: call, in: chat)
            }
            guard let self, !Task.isCancelled else { return }
            self.startNextRound(in: chat)
        }
    }

    private func recordToolResult(_ result: AppToolResult,
                                  for call: AppToolCall,
                                  in chat: AppChatSession) {
        chat.outputContinuationTurns.append(
            .toolResult(callID: call.id, name: call.name, content: result.content))
        if let index = chat.outputToolTrace.firstIndex(where: { $0.id == call.id }) {
            chat.outputToolTrace[index].status = result.isError ? .failed : .done
            chat.outputToolTrace[index].summary = result.summary
        }
    }

    private func startNextRound(in chat: AppChatSession) {
        toolTask = nil
        guard runState == .running, !isCancellationPending else { return }
        let maxRounds = webSearchConfiguration.resolved().maxToolRounds
        let exhausted = toolRoundsUsed >= maxRounds
        let request: AppGenerationRequest
        do {
            request = try makeRequest(
                continuation: chat.outputContinuationTurns,
                systemPrompt: activeSystemPrompt,
                // Past the round budget the tools are withdrawn: the model
                // has to answer with what it has.
                tools: exhausted ? [] : (activeToolExecutor?.definitions ?? []),
                toolChoice: exhausted ? .none : .auto,
                prompt: chat.outputPromptText,
                imagePaths: chat.outputImagePaths,
                history: chat.conversationTurns)
        } catch {
            finishWithError(error as? AppInferenceError ?? .unknown("\(error)"))
            return
        }
        chat.outputText = ""
        chat.outputReasoningText = ""
        startRound(request)
    }

    public func cancel() {
        guard canCancel else { return }
        isCancellationPending = true
        if phase == .tools {
            // Between rounds nothing is generating; the wait is on a tool.
            toolTask?.cancel()
            toolTask = nil
            let chat = generatingChat ?? selectedChat
            for index in chat.outputToolTrace.indices
            where chat.outputToolTrace[index].status == .running {
                chat.outputToolTrace[index].status = .failed
                chat.outputToolTrace[index].summary = "cancelled"
            }
            error = .cancelled
            finishTerminalRun()
            return
        }
        client.cancel()
    }

    /// S1: Ornith runs the official sampler whatever the controls held; the
    /// session would pin the values anyway, so the request carries them
    /// directly and the UI shows them locked.
    public var isSamplingLocked: Bool { selectedModelKind.samplingIsLocked }

    public var supportsVision: Bool { selectedModelKind.supportsVision }

    /// Only Gemma declares the web tools for now; its tool template, grammar
    /// and call parser are the ones the server path has exercised.
    public var webSearchAvailable: Bool { selectedModelKind == .gemmaQATSym }

    public var effectiveWebSearchMode: AppWebSearchMode {
        webSearchAvailable ? webSearchMode : .off
    }

    public func saveWebSearchConfiguration() {
        guard let webSearchConfigurationURL else { return }
        try? WebSearchConfigurationStore.save(webSearchConfiguration, to: webSearchConfigurationURL)
    }

    /// The executor a turn declares: the web tools when a key is set, the
    /// local Wikipedia tools when an index is set, both together when both
    /// are. Nothing to declare is an error the user can act on — the
    /// Inspector names both ways to fix it.
    nonisolated static func makeToolExecutor(configuration: WebSearchConfiguration,
                                             transport: any HTTPTransport = URLSessionTransport()) throws
        -> any AppToolExecutor {
        let resolved = configuration.resolved()
        var executors: [any AppToolExecutor] = []
        if let url = resolved.wikipediaIndexURL {
            do {
                let index = try LocalWikipediaIndex(path: url.path)
                executors.append(WikipediaToolExecutor(
                    index: index, maxResults: resolved.maxSearchResults,
                    pageCharacterLimit: resolved.pageCharacterLimit))
            } catch {
                throw AppInferenceError.invalidRequest(
                    AppLocalization.string("The local Wikipedia index at \(url.path) cannot be used: \(String(describing: error)). Fix the path in the Inspector, or clear it."))
            }
        }
        if resolved.canSearch {
            executors.append(WebSearchToolExecutor(configuration: resolved, transport: transport))
        }
        guard !executors.isEmpty else {
            throw AppInferenceError.invalidRequest(
                AppLocalization.string("Web search needs a Serper or Brave API key, or a local Wikipedia index. Add one in the Inspector, or turn Web search off."))
        }
        return executors.count == 1 ? executors[0] : CompositeToolExecutor(executors)
    }

    private func makeSystemPrompt(for executor: any AppToolExecutor, mode: AppWebSearchMode) -> String {
        WebSearchPrompt.system(maxRounds: webSearchConfiguration.resolved().maxToolRounds,
                               mode: mode, tools: executor.promptFacts)
    }

    /// The request the next `run` would send for the current mode: the tools
    /// and system prompt when web search is on, nothing extra when it is off.
    public func makeRequest() throws -> AppGenerationRequest {
        let mode = effectiveWebSearchMode
        let executor = mode == .off ? nil : try toolExecutorProvider(webSearchConfiguration)
        return try makeFirstRoundRequest(
            mode: mode, executor: executor,
            systemPrompt: executor.map { makeSystemPrompt(for: $0, mode: mode) })
    }

    /// The round that has no results yet. The thought channel is where
    /// Gemma 4 argues with the date instead of searching (`WebSearchPrompt`),
    /// and before the first result there is little else to think about, so
    /// this round is bounded: closed in Always mode (the grammar has already
    /// decided the call), and held to `preSearchThinkingBudget` in Auto.
    /// Rounds that see results think without a bound.
    private func makeFirstRoundRequest(mode: AppWebSearchMode,
                                       executor: (any AppToolExecutor)?,
                                       systemPrompt: String?) throws -> AppGenerationRequest {
        let plan = Self.firstRoundThinking(
            mode: mode, thinking: thinkingEnabled,
            budget: webSearchConfiguration.resolved().preSearchThinkingBudget)
        return try makeRequest(
            continuation: [],
            systemPrompt: systemPrompt,
            tools: executor?.definitions ?? [],
            toolChoice: mode == .always ? .required : .auto,
            enableThinking: plan.enabled,
            reasoningBudgetTokens: plan.budget)
    }

    nonisolated static func firstRoundThinking(mode: AppWebSearchMode, thinking: Bool, budget: Int)
        -> (enabled: Bool, budget: Int) {
        guard thinking, mode != .off else { return (thinking, -1) }
        if mode == .always || budget == 0 { return (false, -1) }
        return (true, budget)
    }

    private func makeRequest(continuation: [AppChatTurn],
                             systemPrompt: String?,
                             tools: [AppToolDefinition],
                             toolChoice: AppToolChoice,
                             enableThinking: Bool? = nil,
                             reasoningBudgetTokens: Int = -1,
                             prompt: String? = nil,
                             imagePaths: [String]? = nil,
                             history: [AppChatTurn]? = nil) throws -> AppGenerationRequest {
        let kind = selectedModelKind
        let temperature = kind.samplingIsLocked ? kind.officialTemperature : temperature
        let topK = kind.samplingIsLocked ? kind.officialTopK : (topKEnabled ? topK : nil)
        let topP = kind.samplingIsLocked
            ? kind.officialTopP
            : (topKEnabled && topPEnabled ? topP : nil)
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: modelPathText),
            history: history ?? conversationTurns,
            prompt: prompt ?? promptText,
            systemPrompt: systemPrompt,
            continuation: continuation,
            tools: tools,
            toolChoice: toolChoice,
            reasoningBudgetTokens: reasoningBudgetTokens,
            maxNewTokens: maxNewTokensOverride ?? maxContextTokens,
            maxContextTokens: maxContextTokens,
            temperature: Float(temperature),
            topK: topK,
            topP: topP.map(Float.init),
            repetitionPenalty: 1.0,
            runtimeOptions: runtimeOptions,
            enableThinking: enableThinking ?? thinkingEnabled,
            imagePaths: imagePaths ?? (kind.supportsVision ? attachedImagePaths : []))
        try request.validate(requireModelDirectory: true)
        return request
    }

    func apply(_ event: AppInferenceEvent) {
        switch event {
        case .prefillProgress(let done, let total):
            phase = .prefill
            livePrefillDone = done
            livePrefillTotal = total
        case .token(let token):
            phase = .decode
            liveTokenCount = token.index + 1
            liveElapsedDecodeSeconds = token.elapsedDecodeSeconds
            if let reporter = client as? any AppInferenceMemoryReporting {
                liveMemoryBytes = reporter.currentInferenceMemoryBytes
            } else {
                liveMemoryBytes = memorySampler.sample()
            }
            let chat = generatingChat ?? selectedChat
            if !token.textDelta.isEmpty {
                chat.outputText += token.textDelta
            }
            if !token.reasoningDelta.isEmpty {
                chat.outputReasoningText += token.reasoningDelta
            }
        case .toolCall(let call):
            pendingToolCalls.append(call)
            let chat = generatingChat ?? selectedChat
            chat.outputToolTrace.append(AppToolTraceEntry(
                id: call.id,
                name: call.name,
                subject: activeToolExecutor?.subject(of: call) ?? call.argumentsJSON))
        case .finished(let diagnostics):
            if !pendingToolCalls.isEmpty, activeToolExecutor != nil {
                continueToolLoop(after: diagnostics)
            } else {
                finishSuccessfully(diagnostics)
            }
        case .cancelled(let diagnostics):
            finishCancelled(diagnostics)
        case .failed(let appError, let partial):
            diagnostics = partial
            materializeServiceTranscript()
            finishWithError(appError)
        }
    }

    private func finishSuccessfully(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        finishTerminalRun()
    }

    private func finishCancelled(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        error = .cancelled
        finishTerminalRun()
    }

    private func materializeServiceTranscript() {
        guard let reporter = client as? any AppInferenceTranscriptReporting else { return }
        let chat = generatingChat ?? selectedChat
        chat.outputText = reporter.generationTranscriptMailbox.completeText
        chat.outputReasoningText = reporter.generationTranscriptMailbox.completeReasoningText
    }

    private func finishWithError(_ appError: AppInferenceError) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        error = appError
        finishTerminalRun()
    }

    private func finishStreamFailure(_ appError: AppInferenceError) {
        materializeServiceTranscript()
        finishWithError(appError)
    }

    private func finishTerminalRun() {
        phase = .idle
        runState = .idle
        isCancellationPending = false
        activeRunRuntimeKey = nil
        generatingChat = nil
        runTask = nil
        toolTask = nil
        activeToolExecutor = nil
        activeSystemPrompt = nil
        pendingToolCalls = []
        persistChatsNow()
    }

    private func clearLoadTask(generation: UInt64) {
        guard generation == loadGeneration else { return }
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
    }

    private func clearUnloadTask(generation: UInt64) {
        guard generation == unloadGeneration else { return }
        unloadTask = nil
    }
}
