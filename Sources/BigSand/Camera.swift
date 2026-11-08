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

/// The shaders: .build/BigSand.metallib, which scripts/build-shaders.sh compiles (the launcher runs it whenever a
/// file in Shaders/ changed). Found by walking up from the executable (.build/<triple>/release/BigSand).
func loadShaderLibrary(_ dev: MTLDevice) throws -> MTLLibrary {
  var dir = Bundle.main.executableURL!.deletingLastPathComponent()
  while dir.lastPathComponent != ".build" {
    if dir.path == "/" { fatalError("BigSand isn't running from the package's .build directory") }
    dir = dir.deletingLastPathComponent()
  }
  return try dev.makeLibrary(URL: dir.appendingPathComponent("BigSand.metallib"))
}
