import Foundation
import Tsugumi

/// SPEC §6 for the Ornith (Qwen 3.5-MoE) family: a request's `tools` /
/// `tool_choice` / `parallel_tool_calls` / `response_format` turned into the
/// constraint that governs generation.
///
/// The sibling of `ChatGrammarBuilder`, not a branch of it — Gemma's call is
/// one JSON-ish object behind a name (`call:get_weather{city:…}`), Ornith's is
/// the XML its own `chat_template.jinja` writes. **The grammar text itself
/// lives in `QwenToolCallGrammar`**, next to the tokenizer, because the format
/// belongs to the checkpoint rather than to the HTTP layer and the kernel-check
/// tool exercises it against the real vocabulary. What is left here is the part
/// that is genuinely about a *request*: which tools a `tool_choice` selects,
/// whether the grammar is lazy, and how a response format collides with a
/// forced choice.
public enum QwenChatGrammarBuilder {
    /// The whole stage. Returns `nil` when the request asks for no constraint.
    ///
    /// The response-format / tools precedence is `ChatGrammarBuilder`'s
    /// (GEN-12), unchanged: with `auto` or `none` the response format wins and
    /// no tool grammar is emitted; with `required` or a named function the two
    /// asks collide, which the request layer turns into a 400 before it ever
    /// gets here, so the branch below only records it.
    public static func constraint(
        tools: [GFTokenizer.FunctionDefinition],
        toolChoice: ChatToolChoice,
        parallelToolCalls: Bool,
        responseFormat: ChatGrammarBuilder.ResponseFormat,
        markers: QwenToolCallMarkers
    ) -> ChatGrammarConstraint? {
        if let schema = responseFormat.schema {
            return responseFormatConstraint(schema: schema,
                                            tools: tools,
                                            toolChoice: toolChoice,
                                            markers: markers)
        }
        return toolConstraint(tools: tools,
                              toolChoice: toolChoice,
                              parallelToolCalls: parallelToolCalls,
                              markers: markers)
    }

    // MARK: - GEN-3

    private static func responseFormatConstraint(
        schema: JSONValue,
        tools: [GFTokenizer.FunctionDefinition],
        toolChoice: ChatToolChoice,
        markers: QwenToolCallMarkers
    ) -> ChatGrammarConstraint {
        var approximations: [String] = []
        switch toolChoice {
        case .required, .function:
            if !tools.isEmpty {
                approximations.append(
                    "response-format-overrides-tool-choice: the response format "
                    + "constrains generation, so no tool call can be produced")
            }
        case .auto, .none:
            break
        }
        let result = QwenToolCallGrammar.responseFormatGrammar(schema: schema,
                                                               markers: markers)
        return ChatGrammarConstraint(grammar: result.grammar,
                                     isLazy: false,
                                     trigger: nil,
                                     approximations: approximations + result.approximations)
    }

    // MARK: - GEN-1 / GEN-4 / GEN-5

    private static func toolConstraint(
        tools: [GFTokenizer.FunctionDefinition],
        toolChoice: ChatToolChoice,
        parallelToolCalls: Bool,
        markers: QwenToolCallMarkers
    ) -> ChatGrammarConstraint? {
        let selected: [GFTokenizer.FunctionDefinition]
        switch toolChoice {
        // GEN-4: `none` is no tool grammar at all.
        case .none:
            return nil
        case .auto, .required:
            selected = tools
        // DEV-17: a named choice pins that one function.
        case .function(let name):
            selected = tools.filter { $0.name == name }
        }
        // No alternative to spell. A named choice that names an undeclared
        // tool lands here too; refusing that request is the request layer's
        // job, not the grammar's.
        guard !selected.isEmpty else { return nil }

        let isLazy = toolChoice == .auto
        let result = QwenToolCallGrammar.grammar(tools: selected,
                                                 parallelToolCalls: parallelToolCalls,
                                                 withPreamble: !isLazy,
                                                 markers: markers)
        return ChatGrammarConstraint(
            grammar: result.grammar,
            isLazy: isLazy,
            // GEN-5: the section-start token.
            trigger: isLazy
                ? ChatGrammarTrigger(tokenID: markers.toolCallStartTokenID,
                                     text: markers.toolCallStart)
                : nil,
            approximations: result.approximations)
    }
}
