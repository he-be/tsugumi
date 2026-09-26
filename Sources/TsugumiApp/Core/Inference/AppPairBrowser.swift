import Foundation
import Network

/// The M6 as its pair serve advertises itself (`Qwen38Pair.bonjourType`, registered only on the interface of the
/// address it binds, bridge0): the address a round connects to.
public struct AppPairPeer: Equatable, Sendable {
    /// The bridge0 IPv4 and port the serve binds, from the TXT record.
    public var host: String
    public var port: Int
    /// The Bonjour name (the M6's computer name).
    public var name: String

    public var address: String { "\(host):\(port)" }
}

/// Watches for the M6's pair serve (docs/qfn-prefill/33 §1 #3). Bonjour and not a connection: the serve takes one
/// connection at a time, and a probe would queue behind the link a round holds.
public final class AppPairBrowser: @unchecked Sendable {
    public static let serviceType = "_tsugumi-pair._tcp"

    private let browser: NWBrowser
    private let queue = DispatchQueue(label: "Tsugumi.PairBrowser")

    /// `onChange` gets the peer (nil when none is advertised) on the main queue.
    public init(onChange: @escaping @MainActor @Sendable (AppPairPeer?) -> Void) {
        let parameters = NWParameters()
        parameters.includePeerToPeer = false
        browser = NWBrowser(for: .bonjourWithTXTRecord(type: Self.serviceType, domain: "local."), using: parameters)
        browser.browseResultsChangedHandler = { results, _ in
            let peer = results.lazy.compactMap(Self.peer).first
            DispatchQueue.main.async { MainActor.assumeIsolated { onChange(peer) } }
        }
        browser.stateUpdateHandler = { state in
            if case .failed = state {
                DispatchQueue.main.async { MainActor.assumeIsolated { onChange(nil) } }
            }
        }
        browser.start(queue: queue)
    }

    deinit {
        browser.cancel()
    }

    private static func peer(_ result: NWBrowser.Result) -> AppPairPeer? {
        guard case .service(let name, _, _, _) = result.endpoint,
              case .bonjour(let txt) = result.metadata,
              let host = txt["ip"], !host.isEmpty else { return nil }
        // The TXT record carries the port too: NWBrowser does not resolve the SRV.
        guard let port = txt["port"].flatMap(Int.init) else { return nil }
        return AppPairPeer(host: host, port: port, name: name)
    }
}
