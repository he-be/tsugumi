import TsugumiAppCore
import TsugumiMacPresentation
import SwiftUI

/// The speedometer in the HUD. The needle is `AppMachineHeadroom.level`:
/// how much of the streamed weights this Mac can hold right now, which is
/// what decides how fast an answer comes. Not the decode rate — with MTP
/// that swings with the task — but the room the machine gives the model.
/// A dial says "faster to the right" without a caption; the numbers behind
/// it open on click.
struct HeadroomGaugeView: View {
    let headroom: AppMachineHeadroom?
    @State private var showsDetail = false

    var body: some View {
        Button {
            showsDetail.toggle()
        } label: {
            HeadroomDial(level: headroom?.level ?? 0, band: headroom?.band ?? .full,
                         radius: 12, lineWidth: 3)
                .padding(.horizontal, 2)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .opacity(headroom == nil ? 0.35 : 1)
        .help(HeadroomText.explanation)
        .popover(isPresented: $showsDetail, arrowEdge: .bottom) {
            if let headroom {
                HeadroomDetailView(headroom: headroom)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(L("Room for the model"))
        .accessibilityValue(headroom.map { HeadroomText.percent($0) } ?? "\u{2014}")
    }
}

/// A 240° dial with the gap at the bottom: the track, the filled share,
/// five ticks, and a needle that swings smoothly between samples.
struct HeadroomDial: View {
    let level: Double
    let band: AppMachineHeadroom.Band
    var radius: CGFloat
    var lineWidth: CGFloat

    private static let sweep = 240.0
    private static let start = 150.0

    var body: some View {
        let diameter = radius * 2
        let angle = Self.start + Self.sweep * level
        ZStack {
            Circle()
                .trim(from: 0, to: Self.sweep / 360)
                .stroke(.quaternary, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
                .rotationEffect(.degrees(Self.start))
            Circle()
                .trim(from: 0, to: Self.sweep / 360 * level)
                .stroke(HeadroomText.color(for: band),
                        style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
                .rotationEffect(.degrees(Self.start))
            ForEach([0.0, 0.25, 0.5, 0.75, 1.0], id: \.self) { fraction in
                Rectangle()
                    .fill(.secondary)
                    .frame(width: 1, height: lineWidth)
                    .offset(y: -(radius + lineWidth * 1.5))
                    .rotationEffect(.degrees(Self.start + Self.sweep * fraction + 90))
            }
            Capsule()
                .fill(.primary)
                .frame(width: max(1.5, lineWidth * 0.55), height: radius * 0.85)
                .offset(y: -radius * 0.425)
                .rotationEffect(.degrees(angle + 90))
            Circle()
                .fill(.primary)
                .frame(width: lineWidth * 1.2, height: lineWidth * 1.2)
        }
        .frame(width: diameter, height: diameter)
        .padding(lineWidth * 2)
        .animation(.smooth(duration: 0.8), value: level)
    }
}

/// What the dial stands for, in numbers: the room, the model, and who
/// holds the rest. Opens from the dial.
struct HeadroomDetailView: View {
    let headroom: AppMachineHeadroom

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 14) {
                HeadroomDial(level: headroom.level, band: headroom.band, radius: 26, lineWidth: 6)
                VStack(alignment: .leading, spacing: 2) {
                    Text(HeadroomText.percent(headroom))
                        .font(.system(.title, design: .rounded).weight(.semibold))
                        .monospacedDigit()
                    Text(HeadroomText.verdict(headroom))
                        .font(.callout)
                }
            }
            Text(HeadroomText.explanation)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Divider()
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                row(L("Free for the model"), MetricFormat.memory(headroom.borrowableBytes))
                row(L("The model"), MetricFormat.memory(headroom.wantedBytes))
                row(L("Other apps and macOS"), MetricFormat.memory(headroom.host.usedBytes))
                row(L("This Mac"), MetricFormat.memory(headroom.host.physicalBytes))
            }
        }
        .padding(16)
        .frame(width: 320)
    }

    private func row(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).foregroundStyle(.secondary)
            Text(value)
                .monospacedDigit()
                .gridColumnAlignment(.trailing)
        }
        .font(.callout)
    }
}

@MainActor
enum HeadroomText {
    static var explanation: String {
        L("How much of the model this Mac can hold in memory right now. More room means faster answers; less only means slower, never broken.")
    }

    static func percent(_ headroom: AppMachineHeadroom) -> String {
        "\(Int((headroom.level * 100).rounded()))%"
    }

    static func verdict(_ headroom: AppMachineHeadroom) -> String {
        switch headroom.band {
        case .full: L("The whole model is in memory.")
        case .partial: L("Part of the model streams from the SSD.")
        case .tight: L("Most of the model streams from the SSD.")
        }
    }

    static func color(for band: AppMachineHeadroom.Band) -> Color {
        switch band {
        case .full, .partial: TsugumiMacTheme.accentColor
        case .tight: .orange
        }
    }
}
