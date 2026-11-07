import Metal
import simd

struct Tick { var tick: UInt32; var sub: UInt32; var pad0: UInt32 = 0; var pad1: UInt32 = 0 }
struct Seed { var x0: Int32; var z0: Int32; var size: Int32; var y0: Int32; var y1: Int32 }
struct Cam {
  var pos = SIMD4<Float>(), fwd = SIMD4<Float>(), right = SIMD4<Float>(), up = SIMD4<Float>(), sun = SIMD4<Float>()
  var tanHalf: Float = 0, aspect: Float = 1, shadows: UInt32 = 1, stats: UInt32 = 0
  var jitter = SIMD2<Float>(), pad = SIMD2<Float>()
}

/// The world, its physics tick and its rays. Everything lives on the GPU; the CPU only encodes.
///
/// State: airborne sand as one run per column (8 bytes: its two ends, speed and sub-cell position); the pile as a
/// shallow granular flow on 8x8-column tiles (grain count + momentum per tile) and its surface (one height per column).
/// A tick: plan (which tiles run) -> fall -> `substeps` flow steps -> surface -> bounds for the rays.
///
/// The world is sized from the cube: 4x its side across, its side plus max(1024, side/2) tall. A 1024 cube (1.07 G
/// grains) falls from 960 cells in a 4096 x 2048 x 4096 world; a 4096 cube (68.7 G grains) from 1984 cells in a
/// 16384 x 6144 x 16384 one.
final class Engine {
  let WX: Int, WZ: Int, HY: Int
  /// Flow steps per tick: the stability limit (|u| + sqrt(g h)) dt/dx <= 0.5 at the deepest pile the cube can make
  /// (taken as 1.25x its side), rounded up to an even number so the flow's state ends back in V[0], Q[0].
  let substeps: Int
  var TILES: Int { (WX / 8) * (WZ / 8) }
  var SUPERS: Int { (WX / 64) * (WZ / 64) }

  let device: MTLDevice, queue: MTLCommandQueue
  let runs, HF, air, countdown, awake, active, args, args2, args3, tileBounds, superBounds, superAir, sunMap, pileTop, stepsOut, censusOut: MTLBuffer
  let V: [MTLBuffer], Q: [MTLBuffer], changed: [MTLBuffer]
  let pPlan, pFlowArgs, pFall, pFlow, pSurface, pSuper, pSun, pSeed, pCensus, pRender: MTLComputePipelineState
  let pPresent: MTLRenderPipelineState
  var cur = 0
  var tick: UInt32 = 0
  let cube: Int
  var grains: Int { cube * cube * cube }

  init(cube: Int = 1024) throws {
    precondition(cube >= 64 && cube % 64 == 0, "the cube's side must be a multiple of 64")
    self.cube = cube
    WX = 4 * cube; WZ = 4 * cube; HY = cube + max(1024, cube / 2)
    precondition(HY <= 65535, "a run's ends are 16 bits")
    let hMax = 1.25 * Double(cube), umax = 25.0, g = 0.2725, dx = 8.0
    substeps = (Int(((umax + (g * hMax).squareRoot()) / (0.5 * dx)).rounded(.up)) + 1) & ~1
    let dev = MTLCreateSystemDefaultDevice()!
    device = dev
    queue = dev.makeCommandQueue()!
    let lib = try loadShaderLibrary(dev)
    let fc = MTLFunctionConstantValues()
    var consts: [Int32] = [Int32(WX), Int32(WZ), Int32(HY), Int32(substeps)]
    for i in 0..<consts.count { fc.setConstantValue(&consts[i], type: .int, index: i) }
    func fn(_ n: String) throws -> MTLFunction { try lib.makeFunction(name: n, constantValues: fc) }
    func k(_ n: String) throws -> MTLComputePipelineState { try dev.makeComputePipelineState(function: try fn(n)) }
    pPlan = try k("plan"); pFlowArgs = try k("flowArgs"); pFall = try k("fall"); pFlow = try k("flow")
    pSurface = try k("surface"); pSuper = try k("superBounds"); pSun = try k("sunShadow"); pSeed = try k("seed"); pCensus = try k("census"); pRender = try k("render")
    let rp = MTLRenderPipelineDescriptor()
    rp.vertexFunction = try fn("presentV"); rp.fragmentFunction = try fn("presentF")
    rp.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
    pPresent = try dev.makeRenderPipelineState(descriptor: rp)

    func buf(_ len: Int, _ opt: MTLResourceOptions = .storageModePrivate) -> MTLBuffer { dev.makeBuffer(length: len, options: opt)! }
    let tiles = (WX / 8) * (WZ / 8), cols = WX * WZ, supers = (WX / 64) * (WZ / 64)
    runs = buf(cols * 8)
    HF = buf(cols * 4)
    V = [buf(tiles * 4), buf(tiles * 4)]; Q = [buf(tiles * 8), buf(tiles * 8)]
    changed = [buf(tiles * 4), buf(tiles * 4)]
    air = buf(tiles * 8); countdown = buf(tiles * 4); awake = buf(tiles * 4); active = buf(tiles * 4)
    args = buf(12, .storageModeShared); args2 = buf(12); args3 = buf(12)
    let a = args.contents().bindMemory(to: UInt32.self, capacity: 3); a[0] = 0; a[1] = 1; a[2] = 1
    tileBounds = buf(tiles * 8); superBounds = buf(supers * 8); superAir = buf(supers * 8)
    // the sun-height map: per tile column, one value per sun line (see sunShadow); lines = TZ + ceil(0.35/0.55 (TX-1)) + 1
    let sunLines = WZ / 8 + Int((0.35 / 0.55 * Double(WX / 8 - 1)).rounded(.up)) + 4
    sunMap = buf(sunLines * (WX / 8) * 4); pileTop = buf(4)
    stepsOut = buf(4, .storageModeShared); censusOut = buf(1024 * 4, .storageModeShared)
    reset()
  }

  /// Bytes of GPU memory the world takes.
  var bytes: Int {
    [runs, HF, air, countdown, awake, active, tileBounds, superBounds, superAir, sunMap].reduce(0) { $0 + $1.length }
      + (V + Q + changed).reduce(0) { $0 + $1.length }
  }

  /// Empty world, then the cube of sand hanging in the sky, at rest.
  func reset() {
    let cb = queue.makeCommandBuffer()!
    let b = cb.makeBlitCommandEncoder()!
    for buf in [runs, HF, V[0], V[1], Q[0], Q[1], changed[0], changed[1], air, countdown, awake, tileBounds, superBounds, superAir, sunMap, pileTop] {
      b.fill(buffer: buf, range: 0..<buf.length, value: 0)
    }
    b.endEncoding()
    let e = cb.makeComputeCommandEncoder()!
    var s = Seed(x0: Int32((WX - cube) / 2), z0: Int32((WZ - cube) / 2), size: Int32(cube),
                 y0: Int32(HY - cube - 64), y1: Int32(HY - 64))
    e.setComputePipelineState(pSeed)
    e.setBuffer(runs, offset: 0, index: 0); e.setBuffer(changed[0], offset: 0, index: 1)
    e.setBuffer(air, offset: 0, index: 2); e.setBuffer(tileBounds, offset: 0, index: 3)
    e.setBytes(&s, length: MemoryLayout<Seed>.stride, index: 4)
    e.dispatchThreads(MTLSize(width: cube, height: cube, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    encodeSuper(e)
    e.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    cur = 0; tick = 0
  }

  private func encodeSuper(_ e: MTLComputeCommandEncoder) {
    e.setComputePipelineState(pSuper)
    e.setBuffer(tileBounds, offset: 0, index: 0); e.setBuffer(superBounds, offset: 0, index: 1)
    e.setBuffer(air, offset: 0, index: 2); e.setBuffer(superAir, offset: 0, index: 3)
    e.dispatchThreads(MTLSize(width: SUPERS, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
  }
  private func perTile(_ e: MTLComputeCommandEncoder) {   // one 64-thread group per active tile (a thread per column)
    e.dispatchThreadgroups(indirectBuffer: args, indirectBufferOffset: 0, threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
  }

  /// One physics tick (1/60 s), all in `cb`.
  func encodeTick(_ cb: MTLCommandBuffer) { encodeTick { _ in cb } }

  /// One physics tick, its passes in the command buffers `cbFor` hands out by name: "plan", "fall", "flow",
  /// "surface", "sun" (the sun-height map) and "bounds" (height bounds for the rays). The headless benchmark gives each pass its own command
  /// buffer to time it; the window gives them all the same one.
  func encodeTick(_ cbFor: (String) -> MTLCommandBuffer) {
    let nxt = 1 - cur
    var t = Tick(tick: tick, sub: 0)
    var cb = cbFor("plan")
    let bl = cb.makeBlitCommandEncoder()!
    bl.fill(buffer: args, range: 0..<4, value: 0)
    bl.endEncoding()
    var e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pPlan)
    e.setBuffer(changed[cur], offset: 0, index: 0); e.setBuffer(changed[nxt], offset: 0, index: 1)
    e.setBuffer(countdown, offset: 0, index: 2); e.setBuffer(awake, offset: 0, index: 3)
    e.setBuffer(args, offset: 0, index: 4); e.setBuffer(active, offset: 0, index: 5)
    e.dispatchThreads(MTLSize(width: TILES, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.setComputePipelineState(pFlowArgs)
    e.setBuffer(args, offset: 0, index: 0); e.setBuffer(args2, offset: 0, index: 1); e.setBuffer(args3, offset: 0, index: 2)
    e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    e.endEncoding()

    // air -> pile (grains that land are added to their tile's count in V[0])
    cb = cbFor("fall"); e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pFall)
    e.setBuffer(runs, offset: 0, index: 0); e.setBuffer(HF, offset: 0, index: 1); e.setBuffer(pileTop, offset: 0, index: 2)
    e.setBuffer(V[0], offset: 0, index: 3); e.setBuffer(active, offset: 0, index: 4); e.setBuffer(changed[nxt], offset: 0, index: 5)
    e.setBuffer(air, offset: 0, index: 6)
    perTile(e)
    e.endEncoding()

    // the pile flows (an even number of steps: the state ends back in V[0], Q[0])
    cb = cbFor("flow"); e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pFlow)
    for s in 0..<substeps {
      t.sub = UInt32(s)
      let a = s & 1, b = 1 - a
      e.setBuffer(V[a], offset: 0, index: 0); e.setBuffer(Q[a], offset: 0, index: 1)
      e.setBuffer(V[b], offset: 0, index: 2); e.setBuffer(Q[b], offset: 0, index: 3)
      e.setBuffer(active, offset: 0, index: 4); e.setBuffer(args, offset: 0, index: 5)
      e.setBuffer(awake, offset: 0, index: 6); e.setBuffer(changed[nxt], offset: 0, index: 7)
      e.setBytes(&t, length: MemoryLayout<Tick>.stride, index: 8)
      e.dispatchThreadgroups(indirectBuffer: args2, indirectBufferOffset: 0, threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }
    e.endEncoding()

    // its surface, and the rays' bounds
    cb = cbFor("surface"); e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pSurface)
    e.setBuffer(V[0], offset: 0, index: 0); e.setBuffer(HF, offset: 0, index: 1); e.setBuffer(active, offset: 0, index: 2)
    e.setBuffer(air, offset: 0, index: 3); e.setBuffer(tileBounds, offset: 0, index: 4); e.setBuffer(pileTop, offset: 0, index: 5)
    perTile(e)
    e.endEncoding()
    // the sun's height over the pile, only on ticks where something ran
    cb = cbFor("sun"); e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pSun)
    e.setBuffer(V[0], offset: 0, index: 0); e.setBuffer(sunMap, offset: 0, index: 1)
    e.dispatchThreadgroups(indirectBuffer: args3, indirectBufferOffset: 0, threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    e.endEncoding()
    cb = cbFor("bounds"); e = cb.makeComputeCommandEncoder()!
    encodeSuper(e)
    e.endEncoding()
    cur = nxt; tick &+= 1
  }

  /// Active tiles of the last completed tick (read after its command buffer finished).
  var activeTiles: Int { Int(args.contents().load(as: UInt32.self)) }

  /// Rays at the input resolution: colour, reversed depth and motion (from `prev`, last frame's camera) for MetalFX.
  func encodeRender(_ cb: MTLCommandBuffer, to tex: MTLTexture, depth: MTLTexture, motion: MTLTexture, cam: inout Cam, prev: inout Cam) {
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pRender)
    e.setTexture(tex, index: 0); e.setTexture(depth, index: 1); e.setTexture(motion, index: 2)
    e.setBytes(&prev, length: MemoryLayout<Cam>.stride, index: 6)
    e.setBuffer(sunMap, offset: 0, index: 7); e.setBuffer(air, offset: 0, index: 8); e.setBuffer(superAir, offset: 0, index: 9)
    e.setBuffer(runs, offset: 0, index: 0); e.setBuffer(HF, offset: 0, index: 1); e.setBuffer(tileBounds, offset: 0, index: 2)
    e.setBuffer(superBounds, offset: 0, index: 3); e.setBytes(&cam, length: MemoryLayout<Cam>.stride, index: 4)
    e.setBuffer(stepsOut, offset: 0, index: 5)
    e.dispatchThreads(MTLSize(width: tex.width, height: tex.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    e.endEncoding()
  }

  func encodePresent(_ cb: MTLCommandBuffer, from tex: MTLTexture, pass: MTLRenderPassDescriptor) {
    let r = cb.makeRenderCommandEncoder(descriptor: pass)!
    r.setRenderPipelineState(pPresent); r.setFragmentTexture(tex, index: 0)
    r.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    r.endEncoding()
  }

  /// Total sand grains: airborne run lengths + the pile's counts. Must equal `grains` forever.
  func census() -> Int {
    memset(censusOut.contents(), 0, censusOut.length)
    let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pCensus)
    e.setBuffer(runs, offset: 0, index: 0); e.setBuffer(V[0], offset: 0, index: 1); e.setBuffer(censusOut, offset: 0, index: 2)
    e.dispatchThreads(MTLSize(width: 10 * 1024 * 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let p = censusOut.contents().bindMemory(to: UInt32.self, capacity: 1024)
    return (0..<1024).reduce(0) { $0 + Int(p[$1]) }
  }
}

/// A camera that flies: position plus yaw/pitch.
struct FlyCam {
  var pos = SIMD3<Float>(-1400, 1500, -1400)
  var yaw: Float = 0.785, pitch: Float = -0.25
  var fovY: Float = 1.0
  /// Cells per second when flying (Shift: 4x).
  var speed: Float = 600
  init() {}
  /// The default view of `e`'s world, and a fly speed to match its size.
  init(world e: Engine) {
    let k = Float(e.WX) / 4096
    pos = SIMD3(-1400 * k, 1500 * Float(e.HY) / 2048, -1400 * k); speed = 600 * k
  }
  var forward: SIMD3<Float> { SIMD3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)) }
  func uniforms(aspect: Float, shadows: Bool, stats: Bool = false) -> Cam {
    let f = forward, r = simd_normalize(simd_cross(f, SIMD3<Float>(0, 1, 0))), u = simd_cross(r, f)
    let sun = simd_normalize(SIMD3<Float>(-0.55, 0.70, -0.35))   // informational: the shaders use SUN_DIR in Shaders/BigSand.metal
    return Cam(pos: SIMD4(pos, 0), fwd: SIMD4(f, 0), right: SIMD4(r, 0), up: SIMD4(u, 0), sun: SIMD4(sun, 0),
               tanHalf: tan(fovY / 2), aspect: aspect, shadows: shadows ? 1 : 0, stats: stats ? 1 : 0)
  }
}

/// The shaders: the precompiled BigSand.metallib inside the .app, else the one scripts/build-shaders.sh leaves in
/// .build/, else Shaders/BigSand.metal compiled from source now (so a plain `swift run` still works).
private func loadShaderLibrary(_ dev: MTLDevice) throws -> MTLLibrary {
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
    return lib
  }
  let sources = [Bundle.main.resourceURL?.appendingPathComponent("BigSand.metal"),
                 packageRoot?.appendingPathComponent("Shaders/BigSand.metal"),
                 URL(fileURLWithPath: fm.currentDirectoryPath).appendingPathComponent("Shaders/BigSand.metal")].compactMap { $0 }
  for url in sources where fm.fileExists(atPath: url.path) {
    let lib = try dev.makeLibrary(source: try String(contentsOf: url, encoding: .utf8), options: nil)
    print("shaders: compiled from source \(url.path) (run scripts/build-shaders.sh to skip this)")
    return lib
  }
  throw NSError(domain: "BigSand", code: 1, userInfo: [NSLocalizedDescriptionKey:
    "no BigSand.metallib or BigSand.metal found (looked in \((libs + sources).map(\.path)))"])
}
