import Metal
import simd

struct Tick { var tick: UInt32; var fall: UInt32; var sub: UInt32; var pad: UInt32 = 0 }
struct Seed { var x0: Int32; var z0: Int32; var size: Int32; var y0: Int32; var y1: Int32 }
struct Cam {
  var pos = SIMD4<Float>(), fwd = SIMD4<Float>(), right = SIMD4<Float>(), up = SIMD4<Float>(), sun = SIMD4<Float>()
  var tanHalf: Float = 0, aspect: Float = 1, shadows: UInt32 = 1, stats: UInt32 = 0
}

/// The world, its physics tick and its rays. Everything lives on the GPU; the CPU only encodes.
///
/// State: one copy of the cells (1 bit each, 4 GiB) + per column the grounded run's height and free space (64 MB).
/// A tick: plan (which 8x8-column tiles run) -> fall (columns with sand in the air) -> topple x `rounds`
/// (grounded heights flow to 45 degrees) -> commit (heights back into cells) -> bounds for the rays.
final class Engine {
  let WX = 4096, WZ = 4096, WORDS = 32
  var TILES: Int { (WX / 8) * (WZ / 8) }
  var SUPERS: Int { (WX / 64) * (WZ / 64) }
  var COLS: Int { WX * WZ }
  var cells: Int { COLS * WORDS * 64 }

  let device: MTLDevice, queue: MTLCommandQueue
  let vox: MTLBuffer
  let H, F: MTLBuffer
  var changed: [MTLBuffer]
  let h0, countdown, awake, active, args, tileBounds, superBounds, stepsOut, censusOut: MTLBuffer
  let pPlan, pFall, pFlux, pHeights, pCommit, pSuper, pSeed, pCensus, pRender: MTLComputePipelineState
  let pPresent: MTLRenderPipelineState
  var cur = 0
  var tick: UInt32 = 0
  var fall: UInt32 = 6     // cells a falling run drops per tick (60 ticks/s -> 360 cells/s)
  var rounds = 4           // flow rounds per tick: a collapse front moves up to this many columns per tick
  var cube = 1024
  var grains: Int { cube * cube * cube }

  init(cube: Int = 1024) throws {
    self.cube = cube
    let dev = MTLCreateSystemDefaultDevice()!
    device = dev
    queue = dev.makeCommandQueue()!
    let lib = try dev.makeLibrary(source: shaderSource(wx: 4096, wz: 4096, words: 32), options: nil)
    func k(_ n: String) throws -> MTLComputePipelineState { try dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
    pPlan = try k("plan"); pFall = try k("fallStep"); pFlux = try k("flux"); pHeights = try k("heights"); pCommit = try k("commit")
    pSuper = try k("superBounds"); pSeed = try k("seed"); pCensus = try k("census"); pRender = try k("render")
    let rp = MTLRenderPipelineDescriptor()
    rp.vertexFunction = lib.makeFunction(name: "presentV"); rp.fragmentFunction = lib.makeFunction(name: "presentF")
    rp.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
    pPresent = try dev.makeRenderPipelineState(descriptor: rp)

    func buf(_ len: Int, _ opt: MTLResourceOptions = .storageModePrivate) -> MTLBuffer { dev.makeBuffer(length: len, options: opt)! }
    let tiles = (4096 / 8) * (4096 / 8), cols = 4096 * 4096
    vox = buf(cols * 32 * 8)
    H = buf(cols * 4); F = buf(cols * 4); h0 = buf(cols * 4)
    changed = [buf(tiles * 4), buf(tiles * 4)]
    countdown = buf(tiles * 4); awake = buf(tiles * 4); active = buf(tiles * 4)
    args = buf(12, .storageModeShared)
    let a = args.contents().bindMemory(to: UInt32.self, capacity: 3); a[0] = 0; a[1] = 1; a[2] = 1
    tileBounds = buf(tiles * 8); superBounds = buf((4096 / 64) * (4096 / 64) * 8)
    stepsOut = buf(4, .storageModeShared); censusOut = buf(1024 * 4, .storageModeShared)
    reset()
  }

  /// Empty world, then the cube of sand hanging in the sky.
  func reset() {
    let cb = queue.makeCommandBuffer()!
    let b = cb.makeBlitCommandEncoder()!
    for buf in [vox, H, F, h0, changed[0], changed[1], countdown, awake, tileBounds, superBounds] {
      b.fill(buffer: buf, range: 0..<buf.length, value: 0)
    }
    b.endEncoding()
    let e = cb.makeComputeCommandEncoder()!
    var s = Seed(x0: Int32((WX - cube) / 2), z0: Int32((WZ - cube) / 2), size: Int32(cube),
                 y0: Int32(WORDS * 64 - cube - 64), y1: Int32(WORDS * 64 - 64))
    e.setComputePipelineState(pSeed)
    e.setBuffer(vox, offset: 0, index: 0); e.setBuffer(H, offset: 0, index: 1)
    e.setBuffer(changed[0], offset: 0, index: 2); e.setBuffer(tileBounds, offset: 0, index: 3)
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
    e.dispatchThreads(MTLSize(width: SUPERS, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
  }
  private func perTile(_ e: MTLComputeCommandEncoder) {
    e.dispatchThreadgroups(indirectBuffer: args, indirectBufferOffset: 0, threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
  }

  /// One physics tick, in two command buffers so each half gets its own GPU time:
  /// `a` plans and drops falling sand; `b` topples the grounded sand and commits it.
  func encodeTick(_ a: MTLCommandBuffer, _ b: MTLCommandBuffer) {
    let nxt = 1 - cur
    let bl = a.makeBlitCommandEncoder()!
    bl.fill(buffer: args, range: 0..<4, value: 0)
    bl.endEncoding()
    var t = Tick(tick: tick, fall: fall, sub: 0)
    let e = a.makeComputeCommandEncoder()!
    e.setComputePipelineState(pPlan)
    e.setBuffer(changed[cur], offset: 0, index: 0); e.setBuffer(changed[nxt], offset: 0, index: 1)
    e.setBuffer(countdown, offset: 0, index: 2); e.setBuffer(awake, offset: 0, index: 3)
    e.setBuffer(args, offset: 0, index: 4); e.setBuffer(active, offset: 0, index: 5)
    e.dispatchThreads(MTLSize(width: TILES, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.setComputePipelineState(pFall)
    e.setBuffer(vox, offset: 0, index: 0); e.setBuffer(H, offset: 0, index: 1)
    e.setBuffer(active, offset: 0, index: 2); e.setBuffer(changed[nxt], offset: 0, index: 3)
    e.setBuffer(tileBounds, offset: 0, index: 4); e.setBytes(&t, length: MemoryLayout<Tick>.stride, index: 5)
    perTile(e)
    e.endEncoding()

    let f = b.makeComputeCommandEncoder()!
    for r in 0..<rounds {
      t.sub = UInt32(r)
      f.setComputePipelineState(pFlux)
      f.setBuffer(H, offset: 0, index: 0); f.setBuffer(F, offset: 0, index: 1)
      f.setBuffer(h0, offset: 0, index: 2); f.setBuffer(active, offset: 0, index: 3)
      f.setBuffer(awake, offset: 0, index: 4); f.setBuffer(changed[nxt], offset: 0, index: 5)
      f.setBytes(&t, length: MemoryLayout<Tick>.stride, index: 6)
      perTile(f)
      f.setComputePipelineState(pHeights)
      f.setBuffer(H, offset: 0, index: 0); f.setBuffer(F, offset: 0, index: 1); f.setBuffer(active, offset: 0, index: 2)
      perTile(f)
    }
    f.setComputePipelineState(pCommit)
    f.setBuffer(vox, offset: 0, index: 0); f.setBuffer(H, offset: 0, index: 1); f.setBuffer(h0, offset: 0, index: 2)
    f.setBuffer(active, offset: 0, index: 3); f.setBuffer(tileBounds, offset: 0, index: 4)
    perTile(f)
    encodeSuper(f)
    f.endEncoding()
    cur = nxt; tick &+= 1
  }

  /// Active tiles of the last completed tick (read after its command buffers finished).
  var activeTiles: Int { Int(args.contents().load(as: UInt32.self)) }

  /// The model (bytes of main-memory traffic), per pass:
  /// fall: each active column reads and writes its full height at most (an upper bound: most skip);
  /// flow: per round each active column reads its height and flows and writes them back (neighbours from cache): ~16 B.
  func fallBytes(_ tiles: Int) -> Double { Double(tiles * 64 * WORDS * 8 * 2) }
  func toppleBytes(_ tiles: Int) -> Double { Double(tiles * 64 * rounds * 16) }

  func encodeRender(_ cb: MTLCommandBuffer, to tex: MTLTexture, cam: inout Cam) {
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pRender)
    e.setTexture(tex, index: 0)
    e.setBuffer(vox, offset: 0, index: 0); e.setBuffer(tileBounds, offset: 0, index: 1); e.setBuffer(superBounds, offset: 0, index: 2)
    e.setBytes(&cam, length: MemoryLayout<Cam>.stride, index: 3); e.setBuffer(stepsOut, offset: 0, index: 4)
    e.dispatchThreads(MTLSize(width: tex.width, height: tex.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    e.endEncoding()
  }

  func encodePresent(_ cb: MTLCommandBuffer, from tex: MTLTexture, pass: MTLRenderPassDescriptor) {
    let r = cb.makeRenderCommandEncoder(descriptor: pass)!
    r.setRenderPipelineState(pPresent); r.setFragmentTexture(tex, index: 0)
    r.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    r.endEncoding()
  }

  /// Total sand grains in the world (must equal `grains` forever: sand is conserved).
  func census() -> Int {
    memset(censusOut.contents(), 0, censusOut.length)
    let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pCensus); e.setBuffer(vox, offset: 0, index: 0); e.setBuffer(censusOut, offset: 0, index: 1)
    e.dispatchThreads(MTLSize(width: 10 * 1024 * 64, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let p = censusOut.contents().bindMemory(to: UInt32.self, capacity: 1024)
    return (0..<1024).reduce(0) { $0 + Int(p[$1]) }
  }
}

/// A camera that flies: position plus yaw/pitch.
struct FlyCam {
  var pos = SIMD3<Float>(-900, 1400, -900)
  var yaw: Float = 0.785, pitch: Float = -0.22
  var fovY: Float = 1.0
  var forward: SIMD3<Float> { SIMD3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)) }
  func uniforms(aspect: Float, shadows: Bool, stats: Bool = false) -> Cam {
    let f = forward, r = simd_normalize(simd_cross(f, SIMD3<Float>(0, 1, 0))), u = simd_cross(r, f)
    let sun = simd_normalize(SIMD3<Float>(-0.55, 0.70, -0.35))   // behind the default camera: the faces you see are lit
    return Cam(pos: SIMD4(pos, 0), fwd: SIMD4(f, 0), right: SIMD4(r, 0), up: SIMD4(u, 0), sun: SIMD4(sun, 0),
               tanHalf: tan(fovY / 2), aspect: aspect, shadows: shadows ? 1 : 0, stats: stats ? 1 : 0)
  }
}
