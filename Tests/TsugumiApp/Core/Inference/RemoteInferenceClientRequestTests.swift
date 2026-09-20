import Foundation
import Testing
@testable import TsugumiAppCore

/// What `RemoteInferenceClient` puts on the wire (`docs/qwen38-27b/10` S2): a llama-server fills a missing `min_p`
/// and `presence_penalty` with 0.05 and 0.0, which are not Qwen3.8's official values.
@Suite struct RemoteInferenceClientRequestTests {
    private func client(_ routing: RemoteInferenceClient.Routing = .direct) -> RemoteInferenceClient {
        RemoteInferenceClient(endpoint: URL(string: "http://127.0.0.1:1")!, modelID: "m", dialect: .qwen,
                              routing: routing)
    }

    @Test func llamaSwapReachesTheServerRoutesThroughUpstream() {
        #expect(client(.llamaSwap).serverRoute("props") == "upstream/m/props")
        #expect(client(.llamaSwap).serverRoute("apply-template") == "upstream/m/apply-template")
        #expect(client(.direct).serverRoute("props") == "props")
        // The default is what every run before `--remote-direct` used.
        #expect(RemoteInferenceClient(endpoint: URL(string: "http://127.0.0.1:1")!, modelID: "m", dialect: .gemma)
            .routing == .llamaSwap)
    }

    @Test func theOfficialSamplerGoesOutWhole() {
        let official = AppModelKind.qwen38.officialSampling(thinking: false)
        let request = AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/tmp"), prompt: "hi",
                                           temperature: Float(official.temperature), topK: official.topK,
                                           topP: Float(official.topP), minP: official.minP.map(Float.init),
                                           presencePenalty: official.presencePenalty.map(Float.init))
        let body = client().requestBody(request)
        #expect(body["temperature"] as? Float == 0.7)
        #expect(body["top_k"] as? Int == 20)
        #expect(body["top_p"] as? Float == 0.8)
        #expect(body["min_p"] as? Float == 0.0)
        #expect(body["presence_penalty"] as? Float == 1.5)
    }

    @Test func aRequestThatNamesNeitherLeavesTheServerDefaults() {
        let request = AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/tmp"), prompt: "hi")
        let body = client().requestBody(request)
        #expect(body["min_p"] == nil)
        #expect(body["presence_penalty"] == nil)
        #expect(body["temperature"] as? Float == 1.0)
    }

    @Test func everySamplingRouteCarriesTheSameValues() {
        let request = AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/tmp"), prompt: "hi",
                                           temperature: 0.7, topK: 20, topP: 0.8, minP: 0, presencePenalty: 1.5)
        var body: [String: Any] = [:]
        RemoteInferenceClient.setSampling(request, in: &body)
        #expect(Set(body.keys) == ["temperature", "top_k", "top_p", "min_p", "presence_penalty"])
    }

    @Test func outOfRangeValuesAreRefused() {
        var request = AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/tmp"), prompt: "hi", minP: 1.5)
        #expect(throws: AppInferenceError.self) { try request.validate(requireModelDirectory: false) }
        request.minP = nil
        request.presencePenalty = 3
        #expect(throws: AppInferenceError.self) { try request.validate(requireModelDirectory: false) }
    }
}
