import TsugumiAppCore
import SwiftUI

struct ModelStatusBadge: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 6) {
            statusDot
            Menu {
                ForEach(AppModelKind.allCases) { kind in
                    Button {
                        model.selectModel(kind)
                    } label: {
                        if kind == model.selectedModelKind {
                            Label(kind.displayName, systemImage: "checkmark")
                        } else {
                            Text(kind.displayName)
                        }
                    }
                }
            } label: {
                Text(model.selectedModelKind.shortName)
                    .font(.callout.weight(.semibold))
                    .lineLimit(1)
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .disabled(model.isRunning || model.isInstallingModel
                      || model.loadState.isLoading)
            .help(model.installDescriptor.repoID)
            .accessibilityLabel(L("Model"))
            .accessibilityValue(model.selectedModelKind.displayName)
        }
    }

    @ViewBuilder
    private var statusDot: some View {
        switch model.presentation.severity {
        case .neutral: dot(.gray)
        case .active, .warning: dot(.orange)
        case .success: dot(.green)
        case .error: dot(.red)
        }
    }

    private func dot(_ color: Color) -> some View {
        Circle().fill(color).frame(width: 8, height: 8).accessibilityHidden(true)
    }
}
