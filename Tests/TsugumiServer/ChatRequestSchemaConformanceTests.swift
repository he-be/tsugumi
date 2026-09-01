import Foundation
import Testing
@testable import Tsugumi
@testable import TsugumiServerCore

/// C0 (CONFORMANCE §1): the SPEC §4 table, one case per row, against the
/// schema alone — no HTTP, no tokenizer, no weights. Every test name carries
/// the SPEC ID it checks so a row and its test can be found from each other.
@Suite("C0 request schema")
struct ChatRequestSchemaConformanceTests {
    // MARK: - helpers

    private static func json(_ extra: String = "") -> String {
        let tail = extra.isEmpty ? "" : ",\(extra)"
        return #"{"model":"any-name","messages":[{"role":"user","content":"hi"}]"# + tail + "}"
    }

    private static func decoded(_ text: String) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self, from: Data(text.utf8))
    }

    /// The table applied to a body that is the minimum valid request plus
    /// whatever the case is about.
    private static func normalized(
        _ extra: String = "",
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws -> NormalizedChatRequest {
        try ChatRequestSchema.normalize(decoded(json(extra)))
    }

    private static func refusal(
        _ extra: String = "",
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws -> ServerRequestError {
        let body = try decoded(json(extra))
        let error = #expect(throws: ServerRequestError.self, sourceLocation: sourceLocation) {
            _ = try ChatRequestSchema.normalize(body)
        }
        return try #require(error, sourceLocation: sourceLocation)
    }

    // MARK: - R1, R2, R5: the acceptance rules

    @Test("R1: an unknown key is ignored, not refused")
    func R1_unknown_keys_are_ignored() throws {
        let request = try Self.normalized(#""user":"someone","service_tier":"auto""#)
        #expect(request["user"] == nil)
        #expect(request["service_tier"] == nil)
        #expect(request.double("temperature") == 1.0)
    }

    @Test("R2: null means unspecified, so the default applies")
    func R2_null_is_unspecified() throws {
        let request = try Self.normalized(
            #""temperature":null,"top_p":null,"seed":null,"max_tokens":null,"stop":null"#)
        #expect(request.double("temperature") == 1.0)
        #expect(request.double("top_p") == 1.0)
        #expect(request.int("seed") == -1)
        #expect(request.int("max_tokens") == -1)
        #expect(request["stop"] == .array([]))
    }

    @Test("REQ-model / R5: any model name is accepted verbatim")
    func REQ_model_is_not_checked() throws {
        let request = try ChatRequestSchema.normalize(
            Self.decoded(#"{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"hi"}]}"#))
        #expect(request.string("model") == "gpt-3.5-turbo")
    }

    @Test("REQ-messages: a body without messages is a 400 that names the field")
    func REQ_messages_is_required() throws {
        let body = try Self.decoded(#"{"model":"m"}"#)
        let error = #expect(throws: ServerRequestError.self) {
            _ = try ChatRequestSchema.normalize(body)
        }
        #expect(error?.type == .invalidRequest)
        #expect(error?.param == "messages")
    }

    // MARK: - stream

    @Test("REQ-stream: defaults to false and is honored when sent")
    func REQ_stream_default_and_value() throws {
        #expect(try Self.normalized().bool("stream") == false)
        #expect(try Self.normalized(#""stream":true"#).bool("stream") == true)
    }

    @Test("REQ-stream-usage: include_usage rides inside stream_options")
    func REQ_stream_usage() throws {
        let request = try Self.normalized(#""stream_options":{"include_usage":true}"#)
        #expect(request.object("stream_options")?["include_usage"] == .bool(true))
    }

    // MARK: - max_tokens

    @Test("REQ-max-tokens: the default is -1, meaning unlimited")
    func REQ_max_tokens_defaults_to_unlimited() throws {
        #expect(try Self.normalized().int("max_tokens") == -1)
    }

    @Test("REQ-max-tokens: -1 and 0 are both accepted values")
    func REQ_max_tokens_accepts_minus_one_and_zero() throws {
        #expect(try Self.normalized(#""max_tokens":-1"#).int("max_tokens") == -1)
        #expect(try Self.normalized(#""max_tokens":0"#).int("max_tokens") == 0)
    }

    @Test("REQ-max-tokens: below -1 is hard, so it is a 400 naming max_tokens")
    func REQ_max_tokens_below_minus_one_is_refused() throws {
        let error = try Self.refusal(#""max_tokens":-2"#)
        #expect(error.type == .invalidRequest)
        #expect(error.param == "max_tokens")
    }

    @Test("REQ-max-tokens: max_completion_tokens is the same field")
    func REQ_max_tokens_alias() throws {
        let request = try Self.normalized(#""max_completion_tokens":128"#)
        #expect(request.int("max_tokens") == 128)
    }

    // MARK: - sampling clamps (R3)

    @Test("REQ-temp: the default is 1.0")
    func REQ_temp_default() throws {
        #expect(try Self.normalized().double("temperature") == 1.0)
    }

    @Test("REQ-temp: clamp [0, ∞) rounds a negative up and leaves a large value alone")
    func REQ_temp_clamps_at_zero_only() throws {
        #expect(try Self.normalized(#""temperature":-1"#).double("temperature") == 0)
        #expect(try Self.normalized(#""temperature":3"#).double("temperature") == 3)
    }

    @Test("REQ-top-p: the default is 1.0 and both ends clamp")
    func REQ_top_p_clamps() throws {
        #expect(try Self.normalized().double("top_p") == 1.0)
        #expect(try Self.normalized(#""top_p":1.5"#).double("top_p") == 1.0)
        #expect(try Self.normalized(#""top_p":-0.5"#).double("top_p") == 0)
    }

    @Test("REQ-top-k: the default is 0 (disabled) and the ceiling is DEV-9's 256")
    func REQ_top_k_clamps() throws {
        #expect(try Self.normalized().int("top_k") == 0)
        #expect(try Self.normalized(#""top_k":0"#).int("top_k") == 0)
        #expect(try Self.normalized(#""top_k":-5"#).int("top_k") == 0)
        #expect(try Self.normalized(#""top_k":1000"#).int("top_k") == 256)
    }

    @Test("REQ-seed: -1 is a value, not a decode failure")
    func REQ_seed_negative_one_is_random() throws {
        #expect(try Self.normalized().int64("seed") == -1)
        #expect(try Self.normalized(#""seed":-1"#).int64("seed") == -1)
        #expect(try Self.normalized(#""seed":42"#).int64("seed") == 42)
    }

    @Test("REQ-stop: a bare string and a long array are both taken as sent")
    func REQ_stop_takes_string_or_array() throws {
        #expect(try Self.normalized(#""stop":"END""#)["stop"] == .string("END"))
        let many = try Self.normalized(#""stop":["a","b","c","d","e"]"#)
        #expect(many.array("stop")?.count == 5)
    }

    @Test("REQ-repeat-penalty: passed through as sent")
    func REQ_repeat_penalty() throws {
        #expect(try Self.normalized().double("repeat_penalty") == 1.0)
        #expect(try Self.normalized(#""repeat_penalty":1.1"#).double("repeat_penalty") == 1.1)
    }

    @Test("REQ-n: hard [1, 1] — one is fine, two is a 400")
    func REQ_n_is_one_only() throws {
        #expect(try Self.normalized(#""n":1"#).int("n") == 1)
        let error = try Self.refusal(#""n":2"#)
        #expect(error.type == .invalidRequest)
        #expect(error.param == "n")
    }

    @Test("REQ-ignored: DEV-5's sampling knobs are accepted and dropped")
    func REQ_ignored_samplers_are_accepted() throws {
        let request = try Self.normalized(
            #""presence_penalty":0.5,"frequency_penalty":1,"min_p":0.05,"typical_p":0.9,"#
            + #""mirostat":2,"dry_multiplier":0.8,"xtc_probability":0.5,"#
            + #""dynatemp_range":0.5,"samplers":["top_k"],"logit_bias":{"1":1},"ignore_eos":true"#)
        for name in ChatRequestSchema.ignoredNames {
            #expect(request[name] == nil, "\(name) should not reach the request")
        }
    }

    // MARK: - contract parameters (R4)

    @Test("REQ-logprobs: false is fine; asking for them is 501, never 400")
    func REQ_logprobs_is_not_supported() throws {
        #expect(throws: Never.self) { _ = try Self.normalized(#""logprobs":false"#) }
        #expect(try Self.refusal(#""logprobs":true"#).type == .notSupported)
        #expect(try Self.refusal(#""top_logprobs":3"#).type == .notSupported)
    }

    /// GEN-4: all four shapes are the grammar's now, so the table accepts them
    /// all and hands them on unchanged. What the four *mean* is
    /// `ChatRequestConstraintTests`; this row only says none of them is refused.
    @Test("REQ-tool-choice / GEN-4: the table accepts all four shapes")
    func REQ_tool_choice_shapes() throws {
        #expect(try Self.normalized()["tool_choice"] == .string("auto"))
        #expect(try Self.normalized(#""tool_choice":"none""#)["tool_choice"] == .string("none"))
        #expect(try Self.normalized(#""tool_choice":"required""#)["tool_choice"]
            == .string("required"))
        let named = #""tool_choice":{"type":"function","function":{"name":"f"}}"#
        #expect(try Self.normalized(named)["tool_choice"]
            == .object(["type": .string("function"),
                        "function": .object(["name": .string("f")])]))
    }

    @Test("REQ-tool-choice: a value outside the four shapes is a 400")
    func REQ_tool_choice_unknown_value() throws {
        let error = try Self.refusal(#""tool_choice":"banana""#)
        #expect(error.type == .invalidRequest)
        #expect(error.param == "tool_choice")
    }

    @Test("REQ-parallel: parallel_tool_calls=false is accepted")
    func REQ_parallel_tool_calls() throws {
        #expect(try Self.normalized().bool("parallel_tool_calls") == true)
        #expect(try Self.normalized(#""parallel_tool_calls":false"#)
            .bool("parallel_tool_calls") == false)
    }

    /// GEN-3: structured output rides the grammar, which now exists, so all
    /// three types are accepted here. What each one constrains is
    /// `ChatRequestConstraintTests`.
    @Test("GEN-3: the table accepts text, json_object and json_schema")
    func GEN_3_structured_output_types_are_accepted() throws {
        for format in [#"{"type":"text"}"#,
                       #"{"type":"json_object"}"#,
                       #"{"type":"json_schema","json_schema":{"name":"s","schema":{}}}"#] {
            #expect(throws: Never.self, "\(format)") {
                _ = try Self.normalized(#""response_format":"# + format)
            }
        }
    }

    @Test("REQ-response-format: an unknown type is a 400")
    func REQ_response_format_unknown_type() throws {
        let error = try Self.refusal(#""response_format":{"type":"banana"}"#)
        #expect(error.type == .invalidRequest)
        #expect(error.param == "response_format")
    }

    // MARK: - reasoning

    @Test("REQ-reasoning-effort: any string rides through; only none has a meaning here")
    func REQ_reasoning_effort_is_not_enumerated() throws {
        #expect(try Self.normalized(#""reasoning_effort":"none""#)
            .string("reasoning_effort") == "none")
        #expect(try Self.normalized(#""reasoning_effort":"ultra""#)
            .string("reasoning_effort") == "ultra")
    }

    @Test("REQ-template-kwargs: taken whole, including keys this template lacks")
    func REQ_template_kwargs() throws {
        let request = try Self.normalized(
            #""chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true}"#)
        #expect(request.object("chat_template_kwargs")?["enable_thinking"] == .bool(true))
        #expect(request.object("chat_template_kwargs")?["preserve_thinking"] == .bool(true))
    }

    @Test("REQ-cache-prompt: on by default, off on request")
    func REQ_cache_prompt() throws {
        #expect(try Self.normalized().bool("cache_prompt") == true)
        #expect(try Self.normalized(#""cache_prompt":false"#).bool("cache_prompt") == false)
    }

    @Test("REQ-reasoning-budget: -1 by default, hard below -1")
    func REQ_reasoning_budget() throws {
        #expect(try Self.normalized().int("reasoning_budget_tokens") == -1)
        #expect(try Self.normalized(#""reasoning_budget_tokens":0"#)
            .int("reasoning_budget_tokens") == 0)
        let error = try Self.refusal(#""reasoning_budget_tokens":-2"#)
        #expect(error.param == "reasoning_budget_tokens")
    }

    @Test("REQ-reasoning-format: auto by default")
    func REQ_reasoning_format() throws {
        #expect(try Self.normalized().string("reasoning_format") == "auto")
        #expect(try Self.normalized(#""reasoning_format":"none""#)
            .string("reasoning_format") == "none")
    }

    @Test("REQ-timings: off by default")
    func REQ_timings_per_token() throws {
        #expect(try Self.normalized().bool("timings_per_token") == false)
        #expect(try Self.normalized(#""timings_per_token":true"#)
            .bool("timings_per_token") == true)
    }

    // MARK: - ERR-3

    @Test("ERR-3: a type mismatch names the field instead of blaming the JSON")
    func ERR_3_type_mismatch_names_the_field() throws {
        let error = try Self.refusal(#""temperature":"hot""#)
        #expect(error.type == .invalidRequest)
        #expect(error.param == "temperature")
        #expect(error.message.contains("temperature"))
        #expect(!error.message.lowercased().contains("malformed json"))
    }

    @Test("ERR-3: a body that is not an object is a 400 that says so")
    func ERR_3_non_object_body() throws {
        let body = try Self.decoded("[1,2,3]")
        let error = #expect(throws: ServerRequestError.self) {
            _ = try ChatRequestSchema.normalize(body)
        }
        #expect(error?.type == .invalidRequest)
    }

    // MARK: - the table itself

    /// EP-4: `/props.default_generation_settings` is "the effective value of
    /// every SPEC §4 row that has a default", and it is built by walking this
    /// table. So a row whose §4 line prints a default and whose `defaultValue`
    /// is nil is a default the client cannot read — four of them were.
    @Test("EP-4: every default SPEC §4 prints is in the table, so /props reports it")
    func EP_4_the_table_carries_every_printed_default() throws {
        let printed: [String: JSONValue] = [
            "stop": .array([]),
            "tools": .array([]),
            "response_format": .object(["type": .string("text")]),
            "chat_template_kwargs": .object([:]),
        ]
        for (name, expected) in printed {
            let field = try #require(ChatRequestSchema.field(named: name), "\(name)")
            #expect(field.defaultValue == expected, "\(name)")
            // And the default is what a request that says nothing gets.
            #expect(try Self.normalized()[name] == expected, "\(name)")
        }
    }

    @Test("the table covers every field the parser reads, with no duplicate spelling")
    func schema_table_is_well_formed() {
        var seen = Set<String>()
        for field in ChatRequestSchema.fields {
            for spelling in field.spellings {
                #expect(seen.insert(spelling).inserted, "\(spelling) appears twice")
            }
        }
    }
}

/// C0, second half: the clamped table values as the sampler actually takes
/// them. Two of the mappings are approximations this engine forces (DEV-10),
/// and an approximation that silently produced an unrunnable config would be
/// worse than the refusal R3 forbids — so every case here also asserts that
/// the engine would accept what it built.
@Suite("C0 sampler mapping")
struct ChatRequestSamplerMappingTests {
    private static func config(_ extra: String = "") throws -> GenerationConfig {
        let tail = extra.isEmpty ? "" : ",\(extra)"
        let body = #"{"model":"m","messages":[{"role":"user","content":"hi"}]"# + tail + "}"
        let config = try ChatRequestParser.parse(Data(body.utf8)).generationConfig
        try config.validate()
        return config
    }

    @Test("the defaults are a plain untruncated draw at temperature 1")
    func defaults_are_runnable() throws {
        let config = try Self.config()
        #expect(config.temperature == 1.0)
        #expect(config.topK == nil)
        #expect(config.topP == nil)
        #expect(config.seed == nil)
        #expect(config.repetitionPenalty == 1.0)
    }

    @Test("DEV-10: top_p below 1 borrows the widest top-k this sampler has")
    func DEV_10_nucleus_needs_a_top_k() throws {
        let config = try Self.config(#""top_p":0.9"#)
        #expect(config.topP == 0.9)
        #expect(config.topK == ChatRequestSchema.topKCeiling)
    }

    @Test("DEV-10: an empty nucleus is the greedy draw it describes")
    func DEV_10_top_p_zero_is_greedy() throws {
        let config = try Self.config(#""top_p":0"#)
        #expect(config.topP == nil)
        #expect(config.topK == 1)
    }

    @Test("an explicit top_k is kept beside top_p")
    func top_k_and_top_p_together() throws {
        let config = try Self.config(#""top_k":40,"top_p":0.8"#)
        #expect(config.topK == 40)
        #expect(config.topP == 0.8)
    }

    @Test("REQ-seed: -1 is a random draw, and every other value is used as sent")
    func REQ_seed_maps_to_the_sampler() throws {
        #expect(try Self.config(#""seed":-1"#).seed == nil)
        #expect(try Self.config(#""seed":42"#).seed == 42)
        #expect(try Self.config(#""seed":-2"#).seed == UInt64(bitPattern: -2))
    }

    @Test("REQ-stop: both spellings reach the sampler")
    func REQ_stop_maps_to_the_sampler() throws {
        #expect(try Self.config(#""stop":"END""#).stopStrings == ["END"])
        #expect(try Self.config(#""stop":["a","b"]"#).stopStrings == ["a", "b"])
    }

    @Test("every clamped corner of the table still builds a config the engine runs")
    func clamped_corners_stay_runnable() throws {
        for extra in [#""temperature":-1"#, #""temperature":100"#, #""top_p":1.5"#,
                      #""top_p":-1"#, #""top_k":-5"#, #""top_k":100000"#,
                      #""top_p":0.5,"top_k":0"#, #""repeat_penalty":0.1"#] {
            #expect(throws: Never.self, "\(extra) should stay runnable") {
                _ = try Self.config(extra)
            }
        }
    }
}
