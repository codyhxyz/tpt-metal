// swift-tools-version:6.0
import PackageDescription

let package = Package(
  name: "BigSand",
  platforms: [.macOS("26.0")],
  targets: [
    .executableTarget(name: "BigSand", path: "Sources/BigSand",
                      swiftSettings: [.unsafeFlags(["-Ounchecked"])])
  ],
  swiftLanguageModes: [.v5]
)
