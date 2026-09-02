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
    /// Whether a turn may leave this Mac (`AppNetworkMode`). Only Gemma
    /// declares tools (`toolsAvailable`); for another model the switch is
    /// kept but ignored.
    public var networkMode: AppNetworkMode = .offline
    /// Keys and limits for the web tools, one file for the app. Edited in
    /// the Inspector; `saveWebSearchConfiguration` writes it back.
    public var webSearchConfiguration: WebSearchConfiguration
    /// The instruction layer every turn carries (`AppPersona`), one file
    /// for the app. Edited in the Inspector; `savePersona` writes it back.
    public var persona: AppPersona
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
    private let toolExecutorProvider: (WebSearchConfiguration, AppNetworkMode) throws -> (any AppToolExecutor)?
    private let webSearchConfigurationURL: URL?
    private let personaURL: URL?
    private var activeToolExecutor: (any AppToolExecutor)?
    private var activeNetworkMode: AppNetworkMode = .offline
    private var activeSystemPrompt: String?
    /// Online with the web tools declared: the turn must search, then read
    /// a page, before it may answer (`AppModel.onlineToolChoice`). The
    /// model is not asked whether it needs to — with the switch on, the
    /// user has already said so.
    private var activeOnlinePolicy = false
    /// The request of the round now generating, kept so a structured-output
    /// failure (the model's first token was a stray control token, which the
    /// tool decoder fails closed on) can run the same round once more.
    private var activeRoundRequest: AppGenerationRequest?
    private var structuredRetriesUsed = 0
    /// Counts rounds; a stream from an earlier round (the one being
    /// retried) is ignored once a newer round has started.
    private var roundGeneration: UInt64 = 0
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
                personaURL: URL? = nil,
                toolExecutorProvider: ((WebSearchConfiguration, AppNetworkMode) throws -> (any AppToolExecutor)?)? = nil) {
        self.chatStore = chatStore
        let personaFileURL = personaURL
            ?? (settingsPersistenceEnabled ? AppPersonaStore.defaultFileURL : nil)
        self.personaURL = personaFileURL
        self.persona = personaFileURL.map { AppPersonaStore.load(from: $0) } ?? AppPersona()
        let configurationURL = webSearchConfigurationURL
            ?? (settingsPersistenceEnabled ? WebSearchConfigurationStore.defaultFileURL : nil)
        self.webSearchConfigurationURL = configurationURL
        self.webSearchConfiguration = configurationURL.map { WebSearchConfigurationStore.load(from: $0) }
            ?? WebSearchConfiguration()
        self.toolExecutorProvider = toolExecutorProvider
            ?? { try Self.makeToolExecutor(configuration: $0, mode: $1) }
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
        self.networkMode = settings.networkMode
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

    public var outputDirective: AppAnswerDirective? { selectedChat.outputDirective }

    public var outputVariants: [AppAnswerVariant] { selectedChat.outputVariants }

    public var selectedVariantIndex: Int { selectedChat.selectedVariantIndex }

    /// How many answers the live turn has to choose between.
    public var answerCount: Int { selectedChat.answerCount }

    /// What the answer on display leaned on, from its trace.
    public var outputGrounding: AppAnswerGrounding {
        AppAnswerGrounding.of(selectedChat.outputToolTrace)
    }

    /// True when the answer on display names no source although a web step
    /// ran: the model searched and then wrote from memory.
    public var outputLacksCitation: Bool {
        outputGrounding.webSteps > 0
            && !AppAnswerGrounding.citesSources(outputResponsePlainText)
    }

    /// Tokens the last round of the selected chat's turn occupied: the
    /// prompt as prefilled (history included) plus what was generated. Nil
    /// until a round has finished; the HUD shows it against
    /// `maxContextTokens`.
    public var contextUsedTokens: Int? {
        guard let diagnostics, let promptTokens = diagnostics.promptTokenCount else { return nil }
        return promptTokens + diagnostics.generatedTokens
    }

    /// A finished answer can be asked for again: same question, same
    /// images, optionally with a directive.
    public var canRegenerate: Bool {
        !isRunning && isModelAvailable && !loadState.isLoading && !hasStaleLoadedRuntime
            && !outputPromptText.isEmpty && !outputResponsePlainText.isEmpty
    }

    public var canAskFollowUp: Bool { canRegenerate }

    /// "Search and answer again" is offered when the model could have
    /// searched and did not: tools are this model's, the web is reachable
    /// with the keys on file, and the answer on display has no web step.
    public var canSearchAgain: Bool {
        canRegenerate && toolsAvailable && webSearchConfiguration.resolved().canSearch
            && outputGrounding.webSteps == 0
    }

    public func savePersona() {
        guard let personaURL else { return }
        try? AppPersonaStore.save(persona, to: personaURL)
    }

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
        networkMode = settings.networkMode
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
            networkMode: networkMode)
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
        chat.outputDirective = nil
        chat.outputVariants = []
        chat.selectedVariantIndex = 0
        if chat.id == mailboxOwnerChatID {
            generationTranscriptMailbox?.reset()
            mailboxOwnerChatID = nil
        }
        diagnostics = nil
        error = nil
        persistChatsNow()
    }

    /// Puts an earlier answer to the live turn on display. The answer now
    /// on display goes back among the variants at its own position, so the
    /// order stays the order the answers were made in. The next run folds
    /// whichever is on display.
    public func selectVariant(_ index: Int) {
        guard !isRunning else { return }
        let chat = selectedChat
        guard index != chat.selectedVariantIndex, (0...chat.outputVariants.count).contains(index) else { return }
        let displayed = AppAnswerVariant(
            directive: chat.outputDirective,
            text: responsePlainText(of: chat),
            reasoningText: chat.outputReasoningText,
            continuationTurns: chat.outputContinuationTurns,
            toolTrace: chat.outputToolTrace)
        chat.outputVariants.insert(displayed, at: chat.selectedVariantIndex)
        let chosen = chat.outputVariants.remove(at: index)
        if chat.id == mailboxOwnerChatID {
            // The mailbox holds the streamed text of the answer just set
            // aside; from here the chat's own fields are the source.
            mailboxOwnerChatID = nil
        }
        chat.outputDirective = chosen.directive
        chat.outputText = chosen.text
        chat.outputReasoningText = chosen.reasoningText
        chat.outputContinuationTurns = chosen.continuationTurns
        chat.outputToolTrace = chosen.toolTrace
        chat.selectedVariantIndex = index
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
                                                      text: chat.outputPromptAsSent,
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
        chat.outputDirective = nil
        chat.outputVariants = []
        chat.selectedVariantIndex = 0
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
        let question = promptText
        let images = selectedModelKind.supportsVision ? chat.attachedImagePaths : []
        foldCompletedTurnIntoHistory(of: chat)
        startTurn(question: question, imagePaths: images, directive: nil,
                  mode: effectiveNetworkMode, in: chat)
    }

    /// Asks the live turn's question again. The answer on display is set
    /// aside as a variant; the new one streams into the live fields. A
    /// `searched` directive goes online and makes the first round call a
    /// tool; the others keep the switch as it is.
    public func regenerate(_ directive: AppAnswerDirective) {
        guard directive == .searched ? canSearchAgain : canRegenerate else { return }
        let chat = selectedChat
        let displayed = AppAnswerVariant(
            directive: chat.outputDirective,
            text: responsePlainText(of: chat),
            reasoningText: chat.outputReasoningText,
            continuationTurns: chat.outputContinuationTurns,
            toolTrace: chat.outputToolTrace)
        chat.outputVariants.insert(displayed, at: chat.selectedVariantIndex)
        chat.selectedVariantIndex = chat.outputVariants.count
        let mode: AppNetworkMode = directive == .searched ? .online : effectiveNetworkMode
        startTurn(question: chat.outputPromptText, imagePaths: chat.outputImagePaths,
                  directive: directive == .again ? nil : directive, mode: mode, in: chat)
    }

    /// Opens a new turn under the answer with a stock question. The
    /// composer's draft is left as it was.
    public func askFollowUp(_ followUp: AppFollowUp) {
        guard canAskFollowUp else { return }
        let draft = promptText
        promptText = followUp.prompt
        run()
        promptText = draft
    }

    /// One turn of the chat: the question as the live user turn, then the
    /// first round (after the app's own lookups when tools are declared).
    /// `question` is what the transcript shows; the model gets it with the
    /// directive's line appended.
    private func startTurn(question: String, imagePaths: [String],
                           directive: AppAnswerDirective?,
                           mode: AppNetworkMode, in chat: AppChatSession) {
        let executor: (any AppToolExecutor)?
        let systemPrompt: String?
        let request: AppGenerationRequest
        if mode == .online, !webSearchConfiguration.resolved().canSearch {
            error = .invalidRequest(
                AppLocalization.string("Online needs a Serper or Brave API key. Add one in the Inspector, or switch to Offline."))
            return
        }
        do {
            executor = toolsAvailable ? try toolExecutorProvider(webSearchConfiguration, mode) : nil
            systemPrompt = makeSystemPrompt(tools: executor)
            request = try makeFirstRoundRequest(
                executor: executor, systemPrompt: systemPrompt,
                prompt: directive?.apply(to: question) ?? question,
                imagePaths: imagePaths, history: chat.conversationTurns)
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
        chat.outputPromptText = question
        chat.outputImagePaths = request.imagePaths
        chat.outputDirective = directive
        chat.outputText = ""
        chat.outputReasoningText = ""
        chat.outputContinuationTurns = []
        chat.outputToolTrace = []
        diagnostics = nil
        error = nil
        activeToolExecutor = executor
        activeNetworkMode = mode
        activeSystemPrompt = systemPrompt
        activeOnlinePolicy = mode == .online
            && request.tools.contains { $0.name == WebSearchToolExecutor.searchToolName }
        structuredRetriesUsed = 0
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
        guard let executor else {
            startRound(request)
            return
        }
        // Before the model's first round the app looks the prompt up itself
        // (the Wikipedia openings of what it names). The wait is a few
        // SQLite queries, on `toolTask` like any tool so `cancel` can stop it.
        phase = .tools
        let prompt = request.prompt
        let callIDPrefix = "lookup-" + UUID().uuidString.lowercased().prefix(8) + "-"
        toolTask = Task { [weak self] in
            let lookups = await executor.lookups(prompt: prompt, callIDPrefix: callIDPrefix)
            guard let self, !Task.isCancelled else { return }
            self.startFirstRound(seeding: lookups, in: chat)
        }
    }

    /// The first generation of a tool-loop turn, after the app's own
    /// lookups: with something found, the transcript starts with those
    /// calls (one assistant turn, as a model's round would be) and their
    /// results; without, as if no lookup had run.
    private func startFirstRound(seeding lookups: [AppToolLookup], in chat: AppChatSession) {
        toolTask = nil
        guard runState == .running, !isCancellationPending else { return }
        if !lookups.isEmpty {
            chat.outputContinuationTurns.append(
                AppChatTurn(role: .assistant, text: "", toolCalls: lookups.map(\.call)))
            for lookup in lookups {
                chat.outputContinuationTurns.append(
                    .toolResult(callID: lookup.call.id, name: lookup.call.name, content: lookup.result.content))
                chat.outputToolTrace.append(AppToolTraceEntry(
                    id: lookup.call.id, name: lookup.call.name, subject: lookup.subject,
                    status: lookup.result.isError ? .failed : .done, summary: lookup.result.summary))
            }
            persistChatsNow()
        }
        do {
            let request = try makeFirstRoundRequest(
                executor: activeToolExecutor,
                systemPrompt: activeSystemPrompt,
                toolChoice: onlineToolChoice(trace: chat.outputToolTrace,
                                             tools: activeToolExecutor?.definitions ?? []).choice,
                continuation: chat.outputContinuationTurns,
                prompt: chat.outputPromptAsSent, imagePaths: chat.outputImagePaths,
                history: chat.conversationTurns)
            startRound(request)
        } catch {
            finishWithError(error as? AppInferenceError ?? .unknown("\(error)"))
        }
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
        activeRoundRequest = request
        roundGeneration &+= 1
        let round = roundGeneration
        runTask = Task.detached { [weak self, client, request] in
            guard let self else { return }
            do {
                for try await event in client.generate(request) {
                    await self.apply(event, round: round)
                }
            } catch let appError as AppInferenceError {
                await self.finishStreamFailure(appError, round: round)
            } catch {
                await self.finishStreamFailure(.unknown("\(error)"), round: round)
            }
        }
    }

    /// The decode service's "structured_output_failure": the tool decoder
    /// refused what the model wrote (in practice a stray tool marker as the
    /// first token). Sampling makes the next draw different, and the prefix
    /// is cached, so the round is run once more before the turn fails.
    nonisolated static func isStructuredOutputFailure(_ error: AppInferenceError) -> Bool {
        if case .unknown(let message) = error {
            return message.hasPrefix("structured_output_failure")
        }
        return false
    }

    private func retryRoundAfterStructuredFailure() -> Bool {
        guard structuredRetriesUsed == 0, let request = activeRoundRequest,
              runState == .running, !isCancellationPending else { return false }
        structuredRetriesUsed += 1
        let chat = generatingChat ?? selectedChat
        chat.outputText = ""
        chat.outputReasoningText = ""
        startRound(request)
        return true
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
        let policy = onlineToolChoice(trace: chat.outputToolTrace,
                                      tools: activeToolExecutor?.definitions ?? [])
        do {
            request = try makeRequest(
                continuation: chat.outputContinuationTurns,
                systemPrompt: activeSystemPrompt,
                // Past the round budget the tools are withdrawn: the model
                // has to answer with what it has.
                tools: exhausted ? [] : policy.tools,
                toolChoice: exhausted ? .none : policy.choice,
                prompt: chat.outputPromptAsSent,
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
    /// Tools are a Gemma affair; Ornith declares none whatever the switch.
    public var toolsAvailable: Bool { selectedModelKind == .gemmaQATSym }

    public var effectiveNetworkMode: AppNetworkMode {
        toolsAvailable ? networkMode : .offline
    }

    public func saveWebSearchConfiguration() {
        guard let webSearchConfigurationURL else { return }
        try? WebSearchConfigurationStore.save(webSearchConfiguration, to: webSearchConfigurationURL)
    }

    /// The executor a turn declares. Offline: the local Wikipedia tools when
    /// an index is set, nothing otherwise (a plain turn). Online: those plus
    /// the web tools, which need a search key — going online without one
    /// is an error the user can act on, the Inspector names both ways out.
    nonisolated static func makeToolExecutor(configuration: WebSearchConfiguration,
                                             mode: AppNetworkMode,
                                             transport: any HTTPTransport = URLSessionTransport()) throws
        -> (any AppToolExecutor)? {
        let resolved = configuration.resolved()
        if mode == .online, !resolved.canSearch {
            throw AppInferenceError.invalidRequest(
                AppLocalization.string("Online needs a Serper or Brave API key. Add one in the Inspector, or switch to Offline."))
        }
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
        if mode == .online {
            executors.append(WebSearchToolExecutor(configuration: resolved, transport: transport))
        }
        guard !executors.isEmpty else { return nil }
        return executors.count == 1 ? executors[0] : CompositeToolExecutor(executors)
    }

    /// The persona first, then the tool instructions when tools are
    /// declared; nil when there is neither, so the request renders through
    /// the plain template exactly as before.
    private func makeSystemPrompt(tools executor: (any AppToolExecutor)?) -> String? {
        var sections: [String] = []
        if let personaSection = persona.promptSection {
            sections.append(personaSection)
        }
        if let executor {
            sections.append(WebSearchPrompt.system(
                maxRounds: webSearchConfiguration.resolved().maxToolRounds,
                tools: executor.promptFacts))
        }
        return sections.isEmpty ? nil : sections.joined(separator: "\n\n")
    }

    /// The request the next `run` would send for the current mode: the tools
    /// and system prompt when any are declared, nothing extra otherwise.
    public func makeRequest() throws -> AppGenerationRequest {
        let executor = toolsAvailable ? try toolExecutorProvider(webSearchConfiguration, effectiveNetworkMode) : nil
        let tools = executor?.definitions ?? []
        let online = effectiveNetworkMode == .online
            && tools.contains { $0.name == WebSearchToolExecutor.searchToolName }
        return try makeFirstRoundRequest(
            executor: executor,
            systemPrompt: makeSystemPrompt(tools: executor),
            toolChoice: onlineToolChoice(trace: [], tools: tools, online: online).choice)
    }

    /// The round that has no results yet. The thought channel is where
    /// Gemma 4 argues with the date instead of searching (`WebSearchPrompt`),
    /// and before the first result there is little else to think about, so
    /// with tools declared this round is held to `preSearchThinkingBudget`.
    /// Rounds that see results think without a bound.
    private func makeFirstRoundRequest(executor: (any AppToolExecutor)?,
                                       systemPrompt: String?,
                                       toolChoice: AppToolChoice = .auto,
                                       continuation: [AppChatTurn] = [],
                                       prompt: String? = nil,
                                       imagePaths: [String]? = nil,
                                       history: [AppChatTurn]? = nil) throws -> AppGenerationRequest {
        let plan = Self.firstRoundThinking(
            tools: executor != nil, thinking: thinkingEnabled,
            budget: webSearchConfiguration.resolved().preSearchThinkingBudget)
        return try makeRequest(
            continuation: continuation,
            systemPrompt: systemPrompt,
            tools: executor?.definitions ?? [],
            toolChoice: toolChoice,
            enableThinking: plan.enabled,
            reasoningBudgetTokens: plan.budget,
            prompt: prompt,
            imagePaths: imagePaths,
            history: history)
    }

    /// What the next round may do, under the online policy: no search yet
    /// and no page read → `web_search` is forced; searched but no page
    /// read → `fetch_page` is forced; a page read (by the model, or by the
    /// app for a URL in the question) → the model chooses. Every round
    /// declares all the tools — the declarations render at the top of the
    /// prompt, and changing them between rounds throws the prompt cache
    /// away; the grammar pins the named tool on its own. A fetch that
    /// failed counts as tried, so a dead site does not eat the round
    /// budget. Off the policy, always the model's choice.
    private func onlineToolChoice(trace: [AppToolTraceEntry], tools: [AppToolDefinition],
                                  online: Bool? = nil)
        -> (choice: AppToolChoice, tools: [AppToolDefinition]) {
        guard online ?? activeOnlinePolicy else { return (.auto, tools) }
        let fetchTried = trace.contains { $0.name == WebSearchToolExecutor.fetchToolName }
        let searched = trace.contains { $0.name == WebSearchToolExecutor.searchToolName && $0.status == .done }
        if fetchTried { return (.auto, tools) }
        if searched {
            guard tools.contains(where: { $0.name == WebSearchToolExecutor.fetchToolName }) else {
                return (.auto, tools)
            }
            return (.function(name: WebSearchToolExecutor.fetchToolName), tools)
        }
        return (.function(name: WebSearchToolExecutor.searchToolName), tools)
    }

    nonisolated static func firstRoundThinking(tools: Bool, thinking: Bool, budget: Int)
        -> (enabled: Bool, budget: Int) {
        guard thinking, tools else { return (thinking, -1) }
        if budget == 0 { return (false, -1) }
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

    func apply(_ event: AppInferenceEvent, round: UInt64? = nil) {
        if let round, round != roundGeneration { return }
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
            if Self.isStructuredOutputFailure(appError), retryRoundAfterStructuredFailure() {
                return
            }
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

    private func finishStreamFailure(_ appError: AppInferenceError, round: UInt64? = nil) {
        if let round, round != roundGeneration { return }
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
        activeOnlinePolicy = false
        activeRoundRequest = nil
        structuredRetriesUsed = 0
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
