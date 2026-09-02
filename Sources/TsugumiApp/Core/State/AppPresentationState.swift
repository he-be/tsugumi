import Foundation

public enum AppPresentationSeverity: Equatable, Sendable {
    case neutral
    case active
    case success
    case warning
    case error
}

public enum AppModelAction: Equatable, Sendable {
    case install
    case cancelInstall
    case load
    case retryLoad
    case cancelLoad
    case reload
    case unload
}

public struct AppPresentationSnapshot: Equatable, Sendable {
    public var requiresInstallation: Bool
    public var installState: AppModelInstallState
    public var installReadiness: AppModelInstallReadiness
    public var loadState: AppModelLoadState
    public var hasStaleRuntime: Bool
    public var isRunning: Bool
    public var isGenerationCancellationPending: Bool
    public var generationPhase: AppGenerationPhase
    public var livePrefillDone: Int
    public var livePrefillTotal: Int
    public var lastStopReason: AppStopReason?

    public init(requiresInstallation: Bool,
                installState: AppModelInstallState,
                installReadiness: AppModelInstallReadiness,
                loadState: AppModelLoadState,
                hasStaleRuntime: Bool,
                isRunning: Bool,
                isGenerationCancellationPending: Bool,
                generationPhase: AppGenerationPhase,
                livePrefillDone: Int = 0,
                livePrefillTotal: Int = 0,
                lastStopReason: AppStopReason? = nil) {
        self.requiresInstallation = requiresInstallation
        self.installState = installState
        self.installReadiness = installReadiness
        self.loadState = loadState
        self.hasStaleRuntime = hasStaleRuntime
        self.isRunning = isRunning
        self.isGenerationCancellationPending = isGenerationCancellationPending
        self.generationPhase = generationPhase
        self.livePrefillDone = livePrefillDone
        self.livePrefillTotal = livePrefillTotal
        self.lastStopReason = lastStopReason
    }
}

public struct AppPresentationState: Equatable, Sendable {
    public var label: String
    public var detail: String?
    public var severity: AppPresentationSeverity
    public var showsActivity: Bool
    public var primaryAction: AppModelAction?
    public var secondaryAction: AppModelAction?

    public init(label: String,
                detail: String? = nil,
                severity: AppPresentationSeverity = .neutral,
                showsActivity: Bool = false,
                primaryAction: AppModelAction? = nil,
                secondaryAction: AppModelAction? = nil) {
        self.label = label
        self.detail = detail
        self.severity = severity
        self.showsActivity = showsActivity
        self.primaryAction = primaryAction
        self.secondaryAction = secondaryAction
    }

    public var conversationAction: AppModelAction? {
        switch primaryAction {
        case .load, .retryLoad, .reload:
            return primaryAction
        case .install, .cancelInstall, .cancelLoad, .unload, nil:
            return nil
        }
    }

    public static func resolve(_ snapshot: AppPresentationSnapshot) -> Self {
        if snapshot.installState.isInstalling {
            if case .cancelling = snapshot.installState {
                return Self(label: AppLocalization.string("Cancelling installation"),
                            severity: .active, showsActivity: true)
            }
            if case .discarding = snapshot.installState {
                return Self(label: AppLocalization.string("Discarding download"),
                            severity: .active, showsActivity: true)
            }
            return Self(label: installLabel(snapshot.installState),
                        severity: .active, showsActivity: true,
                        primaryAction: snapshot.installState.canCancel ? .cancelInstall : nil)
        }

        if snapshot.requiresInstallation {
            if case .recoverable(let message) = snapshot.installState {
                return Self(label: AppLocalization.string("Saved download needs attention"), detail: message,
                            severity: .warning)
            }
            if case .failed(let message) = snapshot.installState {
                return Self(label: AppLocalization.string("Installation failed"), detail: message,
                            severity: .error, primaryAction: .install)
            }
            if case .cancelled = snapshot.installState {
                return Self(label: AppLocalization.string("Download paused"), severity: .warning,
                            primaryAction: .install)
            }
            if case .failed(let message) = snapshot.installReadiness {
                return Self(label: AppLocalization.string("Storage check failed"), detail: message,
                            severity: .error)
            }
            if case .insufficientSpace(let requirement) = snapshot.installReadiness {
                return Self(label: AppLocalization.string("Not enough storage"),
                            detail: AppLocalization.string("\(requirement.shortfallBytes) bytes more required"),
                            severity: .warning)
            }
            if case .checking = snapshot.installReadiness {
                return Self(label: AppLocalization.string("Checking available space"),
                            severity: .active, showsActivity: true)
            }
            if case .ready = snapshot.installReadiness {
                return Self(label: AppLocalization.string("Model required"), primaryAction: .install)
            }
            return Self(label: AppLocalization.string("Model required"))
        }

        switch snapshot.loadState {
        case .loading(let phase):
            return Self(label: phase.label, severity: .active, showsActivity: true,
                        primaryAction: .cancelLoad)
        case .cancelling:
            return Self(label: AppLocalization.string("Cancelling load"), severity: .active, showsActivity: true)
        case .unloading:
            return Self(label: AppLocalization.string("Unloading model"), severity: .active, showsActivity: true)
        case .failed(let error):
            return Self(label: AppLocalization.string("Model load failed"), detail: error.userMessage,
                        severity: .error, primaryAction: .retryLoad)
        case .notLoaded, .ready:
            break
        }

        if snapshot.isRunning {
            if snapshot.isGenerationCancellationPending {
                return Self(label: AppLocalization.string("Stopping"), severity: .active, showsActivity: true)
            }
            switch snapshot.generationPhase {
            case .prefill:
                let label = snapshot.livePrefillTotal > 0
                    ? AppLocalization.string(
                        "Prefill (\(snapshot.livePrefillDone)/\(snapshot.livePrefillTotal))")
                    : AppLocalization.string("Prefill")
                return Self(label: label, severity: .active)
            case .decode:
                return Self(label: AppLocalization.string("Generating"), severity: .active)
            case .tools:
                return Self(label: AppLocalization.string("Searching the web"), severity: .active,
                            showsActivity: true)
            case .idle:
                return Self(label: AppLocalization.string("Starting generation"), severity: .active,
                            showsActivity: true)
            }
        }

        if snapshot.hasStaleRuntime {
            return Self(label: AppLocalization.string("Reload required"), severity: .warning,
                        primaryAction: .reload, secondaryAction: .unload)
        }

        if case .ready = snapshot.loadState {
            if let reason = snapshot.lastStopReason {
                return Self(label: AppLocalization.string("Done · \(reason.rawValue)"), severity: .success,
                            secondaryAction: .unload)
            }
            return Self(label: AppLocalization.string("Ready"), severity: .success, secondaryAction: .unload)
        }

        return Self(label: AppLocalization.string("Installed · Not loaded"), primaryAction: .load)
    }

    private static func installLabel(_ state: AppModelInstallState) -> String {
        switch state {
        case .checking: return AppLocalization.string("Checking installation")
        case .downloadingMetadata: return AppLocalization.string("Downloading metadata")
        case .planning: return AppLocalization.string("Planning installation")
        case .reservingOutput: return AppLocalization.string("Reserving storage")
        case .copyingPayload: return AppLocalization.string("Downloading model")
        case .hashingOutput(let file): return AppLocalization.string("Verifying \(file)")
        case .finalizing: return AppLocalization.string("Finalizing installation")
        case .cancelling: return AppLocalization.string("Cancelling installation")
        case .discarding: return AppLocalization.string("Discarding download")
        case .idle, .cancelled, .recoverable, .installed, .failed:
            return AppLocalization.string("Model required")
        }
    }
}
