# SwiftMath (vendored)

Copied from https://github.com/mgriebling/SwiftMath at tag 1.7.3 (MIT, see LICENSE).
Only the source under `Sources/SwiftMath/{MathBundle,MathRender}` and the Latin Modern
Math font (`mathFonts.bundle`, GUST Font License) are kept; the other fonts are not
shipped.

Local change: `MathResources.swift` replaces the three `Bundle.module` lookups with
`PackagedResourceBundle`, because the accessor SwiftPM generates for `Bundle.module`
does not find the resource bundle inside a `.app` (it looks at the bundle root, where
a sealed app may not carry files) and otherwise falls back to the build machine's
absolute path. See `Sources/TsugumiBundleLocation/PackagedResourceBundle.swift`.
