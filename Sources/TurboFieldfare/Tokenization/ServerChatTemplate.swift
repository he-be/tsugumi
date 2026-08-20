import Foundation

/// The repo-owned chat template the server renders with (SPEC §12 DEV-12).
///
/// It is the pinned checkpoint's own `chat_template.jinja` with one hunk
/// changed, so `diff` against the bundled file is the whole deviation and
/// stays readable when the checkpoint moves: a finished model turn is drawn
/// with the thought channel that was in the KV when it was generated, which is
/// what SPEC INV-1 asks for. The bundled template draws that channel only for
/// a turn newer than the last user message that also carries tool calls, so
/// every redraw of a finished answer diverges from the tokens the model
/// actually produced and the prompt cache's common prefix stops there.
///
/// Only the server path uses it (`ChatTemplateVariant.serverRedraw`). The CLI
/// and the Mac app keep the bundled rendering, so their token streams — and
/// every measurement taken through them — stay byte for byte what they were.
public enum ServerChatTemplate {
    private static let resource = "server_chat_template"
    private static let ext = "jinja"
    private static let subdirectory = "Templates"

    private static let loaded: Result<String, GFTokenizerError> = {
        guard let url = Bundle.module.url(forResource: resource,
                                          withExtension: ext,
                                          subdirectory: subdirectory)
            ?? Bundle.module.url(forResource: resource, withExtension: ext) else {
            return .failure(.invalidChatTemplate(
                "\(resource).\(ext) is missing from the package resources"))
        }
        do {
            return .success(try String(contentsOf: url, encoding: .utf8))
        } catch {
            return .failure(.invalidChatTemplate(
                "\(resource).\(ext) could not be read: \(error)"))
        }
    }()

    /// The template source. Read once; the failure is the same every time.
    public static func jinja() throws -> String { try loaded.get() }
}
