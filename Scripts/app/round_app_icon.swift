// ロゴ (docs/assets/tsugumi.png) を macOS のアプリアイコンの形に整える。
//
//   swift Scripts/app/round_app_icon.swift \
//       docs/assets/tsugumi.png \
//       Sources/TsugumiApp/Mac/Resources/tsugumi-app-icon.png
//
// macOS Big Sur 以降のアイコンは 1024 の canvas の中央に 824 角の角丸正方形が
// 置かれ、周りは透明の余白になっている。全面を絵で埋めると Finder や Dock で
// 一つだけ角の立った四角いタイルになるので、その格子に合わせて切り抜く。
// 角は円弧ではなく連続曲線 (cornerCurve = .continuous) — システムのアイコンと
// 同じ曲がり方にするため、CALayer に描かせたものをマスクとして使っている。
//
// 出来上がりを Sources/... に置いてコミットする。アプリが実行時に読む画像
// (MacAppIcon) と make_app.sh が作る .icns はどちらもこの 1 枚から来るので、
// Dock も Finder も About パネルも同じ絵になる。
import AppKit

let canvas: CGFloat = 1024      // アイコン全体
let body: CGFloat = 824         // 角丸正方形の一辺 (Apple の格子)
let cornerRadius: CGFloat = 185.4
let inset = (canvas - body) / 2

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

let arguments = CommandLine.arguments
guard arguments.count == 3 else {
    fail("usage: round_app_icon.swift <source.png> <destination.png>")
}
let sourceURL = URL(fileURLWithPath: arguments[1])
let destinationURL = URL(fileURLWithPath: arguments[2])

guard let source = CGImageSourceCreateWithURL(sourceURL as CFURL, nil),
      let logo = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
    fail("\(sourceURL.path) を画像として読めなかった")
}

// 角丸の形そのものを 8bit のマスクにする。CALayer に描かせているのは、
// 連続曲線の角を自分で書き起こさずにシステムと同じ形を得るため。
// CGImage のマスクは「黒いところが通る」ので、地を白、形を黒で描く。
guard let maskContext = CGContext(
    data: nil,
    width: Int(body), height: Int(body),
    bitsPerComponent: 8, bytesPerRow: 0,
    space: CGColorSpaceCreateDeviceGray(),
    bitmapInfo: CGImageAlphaInfo.none.rawValue
) else { fail("マスクの context を作れなかった") }
maskContext.setFillColor(gray: 1, alpha: 1)
maskContext.fill(CGRect(x: 0, y: 0, width: body, height: body))

let shape = CALayer()
shape.frame = CGRect(x: 0, y: 0, width: body, height: body)
shape.backgroundColor = CGColor(gray: 0, alpha: 1)
shape.cornerRadius = cornerRadius
shape.cornerCurve = .continuous
shape.masksToBounds = true
shape.render(in: maskContext)

guard let maskImage = maskContext.makeImage(),
      let mask = CGImage(
          maskWidth: maskImage.width, height: maskImage.height,
          bitsPerComponent: 8, bitsPerPixel: 8,
          bytesPerRow: maskImage.bytesPerRow,
          provider: maskImage.dataProvider!, decode: nil,
          shouldInterpolate: true
      ) else { fail("マスクを作れなかった") }

guard let context = CGContext(
    data: nil,
    width: Int(canvas), height: Int(canvas),
    bitsPerComponent: 8, bytesPerRow: 0,
    space: CGColorSpace(name: CGColorSpace.sRGB)!,
    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
) else { fail("出力の context を作れなかった") }

let bodyRect = CGRect(x: inset, y: inset, width: body, height: body)

// システムのアイコンと同じく、下に薄く影を落とす。明るい壁紙の上で
// 白っぽい地の絵が背景に溶けないようにするため。
context.setShadow(
    offset: CGSize(width: 0, height: -8),
    blur: 24,
    color: CGColor(gray: 0, alpha: 0.18)
)
context.saveGState()
context.clip(to: bodyRect, mask: mask)
context.draw(logo, in: bodyRect)
context.restoreGState()

guard let output = context.makeImage(),
      let destination = CGImageDestinationCreateWithURL(
          destinationURL as CFURL, "public.png" as CFString, 1, nil
      ) else { fail("\(destinationURL.path) に書けなかった") }
CGImageDestinationAddImage(destination, output, nil)
guard CGImageDestinationFinalize(destination) else {
    fail("\(destinationURL.path) の書き出しに失敗した")
}
print("\(destinationURL.path) — \(output.width)x\(output.height)")
