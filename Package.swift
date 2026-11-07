// swift-tools-version:6.0
import PackageDescription

let package = Package(
  name: "BigSand",
  platforms: [.macOS("26.0")],
  targets: [
    // binary greedy meshing v2 (MIT), vendored with tpt-metal's changes; see Sources/Mesher/mesher.h
    .target(name: "Mesher", path: "Sources/Mesher", exclude: ["LICENSE"],
            cxxSettings: [.unsafeFlags(["-O3", "-Wno-shorten-64-to-32"])]),
    .executableTarget(name: "BigSand", dependencies: ["Mesher"], path: "Sources/BigSand",
                      swiftSettings: [.unsafeFlags(["-Ounchecked"])])
  ],
  swiftLanguageModes: [.v5],
  cxxLanguageStandard: .cxx17
)
