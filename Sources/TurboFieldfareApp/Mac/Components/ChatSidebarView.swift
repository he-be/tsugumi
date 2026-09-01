import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

/// The chat list: one row per conversation, the generating one marked with
/// a spinner. Switching is always allowed; the generating chat is the only
/// one that cannot be deleted.
struct ChatSidebarView: View {
    let model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(model.chats) { chat in
                        row(for: chat)
                    }
                }
                .padding(8)
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var header: some View {
        HStack {
            Text("Chats")
                .font(.headline)
            Spacer()
            Button {
                model.newChat()
            } label: {
                Image(systemName: "square.and.pencil")
                    .frame(width: 24, height: 24)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.borderless)
            .help("New chat")
            .accessibilityLabel("New chat")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private func row(for chat: AppChatSession) -> some View {
        let isSelected = chat.id == model.selectedChatID
        let isGenerating = chat.id == model.generatingChatID
        return Button {
            model.selectChat(chat.id)
        } label: {
            HStack(spacing: 8) {
                Text(chat.title)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .foregroundStyle(isSelected ? .primary : .secondary)
                Spacer(minLength: 0)
                if isGenerating {
                    ProgressView()
                        .controlSize(.small)
                        .accessibilityLabel("Generating")
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(RoundedRectangle(cornerRadius: 8))
            .background {
                if isSelected {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(TurboFieldfareMacTheme.accentColor.opacity(0.14))
                }
            }
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button("Delete chat") {
                model.deleteChat(chat.id)
            }
            .disabled(!model.canDeleteChat(chat.id))
        }
    }
}
