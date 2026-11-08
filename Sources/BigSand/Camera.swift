import Metal
import simd

struct Cam {
  var pos = SIMD4<Float>(), fwd = SIMD4<Float>(), right = SIMD4<Float>(), up = SIMD4<Float>(), sun = SIMD4<Float>()
  var tanHalf: Float = 0, aspect: Float = 1
  var jitter = SIMD2<Float>()
}

/// A camera that flies: position plus yaw/pitch.
struct FlyCam {
  var pos = SIMD3<Float>(-1400, 1500, -1400)
  var yaw: Float = 0.785, pitch: Float = -0.25
  var fovY: Float = 1.0
  /// Cells per second when flying (Shift: 4x).
  var speed: Float = 600
  init() {}
  var forward: SIMD3<Float> { SIMD3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)) }
  func uniforms(aspect: Float) -> Cam {
    let f = forward, r = simd_normalize(simd_cross(f, SIMD3<Float>(0, 1, 0))), u = simd_cross(r, f)
    let sun = simd_normalize(SIMD3<Float>(-0.55, 0.70, -0.35))
    return Cam(pos: SIMD4(pos, 0), fwd: SIMD4(f, 0), right: SIMD4(r, 0), up: SIMD4(u, 0), sun: SIMD4(sun, 0),
               tanHalf: tan(fovY / 2), aspect: aspect)
  }
}

/// All the shaders, from one or more libraries.
struct ShaderLib {
  let libs: [MTLLibrary]
  func makeFunction(name: String, constantValues: MTLFunctionConstantValues) throws -> MTLFunction {
    for l in libs where l.functionNames.contains(name) { return try l.makeFunction(name: name, constantValues: constantValues) }
    throw NSError(domain: "BigSand", code: 2, userInfo: [NSLocalizedDescriptionKey: "no shader function \(name)"])
  }
}

/// The shaders: the precompiled BigSand.metallib inside the .app, else the one scripts/build-shaders.sh leaves in
/// .build/, else every Shaders/*.metal compiled from source now (so a plain `swift run` still works).
func loadShaderLibrary(_ dev: MTLDevice) throws -> ShaderLib {
  let fm = FileManager.default
  // The package's .build directory, found by walking up from the executable (.build/<triple>/release/BigSand).
  var buildDir: URL? = nil
  if let exe = Bundle.main.executableURL {
    var dir = exe.deletingLastPathComponent()
    while dir.path != "/" {
      if dir.lastPathComponent == ".build" { buildDir = dir; break }
      dir = dir.deletingLastPathComponent()
    }
  }
  let packageRoot = buildDir?.deletingLastPathComponent()

  let libs = [Bundle.main.resourceURL?.appendingPathComponent("BigSand.metallib"),
              buildDir?.appendingPathComponent("BigSand.metallib")].compactMap { $0 }
  for url in libs where fm.fileExists(atPath: url.path) {
    let lib = try dev.makeLibrary(URL: url)
    print("shaders: precompiled \(url.path)")
    return ShaderLib(libs: [lib])
  }
  let dirs = [Bundle.main.resourceURL, packageRoot?.appendingPathComponent("Shaders"),
              URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("Shaders")].compactMap { $0 }
  for dir in dirs {
    let files = ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? []).filter { $0.hasSuffix(".metal") }.sorted()
    if files.isEmpty { continue }
    let compiled = try files.map { try dev.makeLibrary(source: try String(contentsOf: dir.appendingPathComponent($0), encoding: .utf8), options: nil) }
    print("shaders: compiled from source in \(dir.path) (run scripts/build-shaders.sh to skip this)")
    return ShaderLib(libs: compiled)
  }
  throw NSError(domain: "BigSand", code: 1, userInfo: [NSLocalizedDescriptionKey:
    "no BigSand.metallib or Shaders/*.metal found (looked in \((libs + dirs).map(\.path)))"])
}
