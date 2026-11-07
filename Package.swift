// swift-tools-version:6.0
import PackageDescription

let package = Package(
  name: "BigSand",
  platforms: [.macOS(.v15)],
  targets: [
    .executableTarget(name: "BigSand", path: "Sources/BigSand",
                      swiftSettings: [.unsafeFlags(["-Ounchecked"])])
  ],
  swiftLanguageModes: [.v5]
)
