import AppKit
import TsugumiAppCore
import TsugumiMacPresentation
import SwiftUI

// Run as a regular foreground app even when launched as a bare SwiftPM
// executable (no .app bundle): Dock icon, click-to-activate, full main menu
// with Quit (Cmd+Q).
private final class ForegroundAppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        if let icon = MacAppIcon.load() {
            NSApp.applicationIconImage = icon
            NSApp.dockTile.display()
        }
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct TsugumiMacApp: App {
    @NSApplicationDelegateAdaptor private var appDelegate: ForegroundAppDelegate
    @State private var model: AppModel

    init() {
        _model = State(initialValue: AppModel(
            client: DecodeServiceInferenceClient(),
            settingsPersistenceEnabled: true,
            chatStore: .defaultStore))
    }

    var body: some Scene {
        Window("Tsugumi", id: "main") {
            RootView(model: model)
                .frame(minWidth: 1040, minHeight: 560)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1040, height: 720)
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button(L("About Tsugumi")) {
                    NSApp.orderFrontStandardAboutPanel(
                        options: AboutPanelPresentation.options(
                            infoDictionary: Bundle.main.infoDictionary,
                            icon: MacAppIcon.load()))
                }
            }
            CommandMenu(L("Generation")) {
                Button(L("Cancel Generation")) { model.cancel() }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!model.canCancel)
                Button(L("Cancel Model Installation")) { model.cancelInstall() }
                    .disabled(!model.canCancelInstall)
            }
            CommandMenu(L("Model")) {
                Button(L("Load Model"), action: model.loadModel)
                    .disabled(!model.canLoadModel)
                Button(L("Reload Model"), action: model.reloadModel)
                    .disabled(!model.canReloadModel)
                Button(L("Unload Model"), action: model.unloadModel)
                    .disabled(!model.canUnloadModel)
                Divider()
                Button(L("Reveal Model in Finder"), action: revealModel)
                    .disabled(modelRevealTarget == .unavailable)
            }
            CommandMenu(L("Settings")) {
                Picker(L("Send Message With"), selection: newlineShortcutBinding) {
                    ForEach(AppNewlineShortcut.sendMessageOptions) { shortcut in
                        Text(shortcut.sendMessageLabel).tag(shortcut)
                    }
                }
                Picker(L("After Sending"), selection: sentPromptBehaviorBinding) {
                    ForEach(AppSentPromptBehavior.allCases) { behavior in
                        Text(behavior.settingsLabel).tag(behavior)
                    }
                }
            }
        }
    }

    private var modelRevealTarget: ModelRevealTarget {
        ModelRevealPolicy.target(
            forModelPath: model.modelPathText,
            fileExists: FileManager.default.fileExists(atPath:))
    }

    private func revealModel() {
        switch modelRevealTarget {
        case .selectItem(let url):
            NSWorkspace.shared.activateFileViewerSelecting([url])
        case .openContainer(let url):
            NSWorkspace.shared.open(url)
        case .unavailable:
            break
        }
    }

    private var newlineShortcutBinding: Binding<AppNewlineShortcut> {
        Binding {
            model.newlineShortcut
        } set: { shortcut in
            model.setNewlineShortcut(shortcut)
        }
    }

    private var sentPromptBehaviorBinding: Binding<AppSentPromptBehavior> {
        Binding {
            model.sentPromptBehavior
        } set: { behavior in
            model.setSentPromptBehavior(behavior)
        }
    }
}
