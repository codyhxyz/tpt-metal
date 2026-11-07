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
/// State: airborne sand (1 bit per cell, 4 GiB) with a fall speed per column; the pile as a shallow granular flow on
/// 8x8-column tiles (grain count + momentum per tile) and its surface (one height per column).
/// A tick: plan (which tiles run) -> fall -> `substeps` flow steps -> surface -> bounds for the rays.
final class Engine {
  let WX = 4096, WZ = 4096, WORDS = 32
  let substeps = 12
  var TILES: Int { (WX / 8) * (WZ / 8) }
  var SUPERS: Int { (WX / 64) * (WZ / 64) }
  var cells: Int { WX * WZ * WORDS * 64 }

  let device: MTLDevice, queue: MTLCommandQueue
  let vox, HF, FV, air, countdown, awake, active, args, args2, tileBounds, superBounds, stepsOut, censusOut: MTLBuffer
  let V: [MTLBuffer], Q: [MTLBuffer], changed: [MTLBuffer]
  let pPlan, pFlowArgs, pFall, pFlow, pSurface, pSuper, pSeed, pCensus, pRender, pResolve: MTLComputePipelineState
  let pPresent: MTLRenderPipelineState
  var cur = 0
  var tick: UInt32 = 0
  var cube = 1024
  var grains: Int { cube * cube * cube }

  init(cube: Int = 1024) throws {
    self.cube = cube
    let dev = MTLCreateSystemDefaultDevice()!
    device = dev
    queue = dev.makeCommandQueue()!
    let lib = try dev.makeLibrary(source: shaderSource(wx: 4096, wz: 4096, words: 32, substeps: 12), options: nil)
    func k(_ n: String) throws -> MTLComputePipelineState { try dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
    pPlan = try k("plan"); pFlowArgs = try k("flowArgs"); pFall = try k("fall"); pFlow = try k("flow")
    pSurface = try k("surface"); pSuper = try k("superBounds"); pSeed = try k("seed"); pCensus = try k("census"); pRender = try k("render"); pResolve = try k("resolve")
    let rp = MTLRenderPipelineDescriptor()
    rp.vertexFunction = lib.makeFunction(name: "presentV"); rp.fragmentFunction = lib.makeFunction(name: "presentF")
    rp.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
    pPresent = try dev.makeRenderPipelineState(descriptor: rp)

    func buf(_ len: Int, _ opt: MTLResourceOptions = .storageModePrivate) -> MTLBuffer { dev.makeBuffer(length: len, options: opt)! }
    let tiles = (4096 / 8) * (4096 / 8), cols = 4096 * 4096
    vox = buf(cols * 32 * 8)
    HF = buf(cols * 4); FV = buf(cols * 8)
    V = [buf(tiles * 4), buf(tiles * 4)]; Q = [buf(tiles * 8), buf(tiles * 8)]
    changed = [buf(tiles * 4), buf(tiles * 4)]
    air = buf(tiles * 8); countdown = buf(tiles * 4); awake = buf(tiles * 4); active = buf(tiles * 4)
    args = buf(12, .storageModeShared); args2 = buf(12)
    let a = args.contents().bindMemory(to: UInt32.self, capacity: 3); a[0] = 0; a[1] = 1; a[2] = 1
    tileBounds = buf(tiles * 8); superBounds = buf((4096 / 64) * (4096 / 64) * 8)
    stepsOut = buf(4, .storageModeShared); censusOut = buf(1024 * 4, .storageModeShared)
    reset()
  }

  /// Empty world, then the cube of sand hanging in the sky, at rest.
  func reset() {
    let cb = queue.makeCommandBuffer()!
    let b = cb.makeBlitCommandEncoder()!
    for buf in [vox, HF, FV, V[0], V[1], Q[0], Q[1], changed[0], changed[1], air, countdown, awake, tileBounds, superBounds] {
      b.fill(buffer: buf, range: 0..<buf.length, value: 0)
    }
    b.endEncoding()
    let e = cb.makeComputeCommandEncoder()!
    var s = Seed(x0: Int32((WX - cube) / 2), z0: Int32((WZ - cube) / 2), size: Int32(cube),
                 y0: Int32(WORDS * 64 - cube - 64), y1: Int32(WORDS * 64 - 64))
    e.setComputePipelineState(pSeed)
    e.setBuffer(vox, offset: 0, index: 0); e.setBuffer(changed[0], offset: 0, index: 1)
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
    e.dispatchThreads(MTLSize(width: SUPERS, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
  }
  private func perTile(_ e: MTLComputeCommandEncoder) {   // one 64-thread group per active tile (a thread per column)
    e.dispatchThreadgroups(indirectBuffer: args, indirectBufferOffset: 0, threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
  }

  /// One physics tick (1/60 s).
  func encodeTick(_ cb: MTLCommandBuffer) {
    let nxt = 1 - cur
    let bl = cb.makeBlitCommandEncoder()!
    bl.fill(buffer: args, range: 0..<4, value: 0)
    bl.endEncoding()
    var t = Tick(tick: tick, sub: 0)
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pPlan)
    e.setBuffer(changed[cur], offset: 0, index: 0); e.setBuffer(changed[nxt], offset: 0, index: 1)
    e.setBuffer(countdown, offset: 0, index: 2); e.setBuffer(awake, offset: 0, index: 3)
    e.setBuffer(args, offset: 0, index: 4); e.setBuffer(active, offset: 0, index: 5)
    e.dispatchThreads(MTLSize(width: TILES, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.setComputePipelineState(pFlowArgs)
    e.setBuffer(args, offset: 0, index: 0); e.setBuffer(args2, offset: 0, index: 1)
    e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))

    // air -> pile (grains that land are added to their tile's count in V[0])
    e.setComputePipelineState(pFall)
    e.setBuffer(vox, offset: 0, index: 0); e.setBuffer(HF, offset: 0, index: 1); e.setBuffer(FV, offset: 0, index: 2)
    e.setBuffer(V[0], offset: 0, index: 3); e.setBuffer(active, offset: 0, index: 4); e.setBuffer(changed[nxt], offset: 0, index: 5)
    e.setBuffer(air, offset: 0, index: 6)
    perTile(e)

    // the pile flows (an even number of steps: the state ends back in V[0], Q[0])
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

    // its surface, and the rays' bounds
    e.setComputePipelineState(pSurface)
    e.setBuffer(V[0], offset: 0, index: 0); e.setBuffer(HF, offset: 0, index: 1); e.setBuffer(active, offset: 0, index: 2)
    e.setBuffer(air, offset: 0, index: 3); e.setBuffer(tileBounds, offset: 0, index: 4)
    perTile(e)
    encodeSuper(e)
    e.endEncoding()
    cur = nxt; tick &+= 1
  }

  /// Active tiles of the last completed tick (read after its command buffer finished).
  var activeTiles: Int { Int(args.contents().load(as: UInt32.self)) }

  func encodeRender(_ cb: MTLCommandBuffer, to tex: MTLTexture, depth: MTLTexture, cam: inout Cam) {
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pRender)
    e.setTexture(tex, index: 0); e.setTexture(depth, index: 1)
    e.setBuffer(vox, offset: 0, index: 0); e.setBuffer(HF, offset: 0, index: 1); e.setBuffer(tileBounds, offset: 0, index: 2)
    e.setBuffer(superBounds, offset: 0, index: 3); e.setBytes(&cam, length: MemoryLayout<Cam>.stride, index: 4)
    e.setBuffer(stepsOut, offset: 0, index: 5)
    e.dispatchThreads(MTLSize(width: tex.width, height: tex.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    e.endEncoding()
  }

  /// Temporal antialiasing: blend this frame's jittered samples into the reprojected history.
  func encodeResolve(_ cb: MTLCommandBuffer, raw: MTLTexture, depth: MTLTexture, history: MTLTexture, into out: MTLTexture,
                     cam: inout Cam, prev: inout Cam, valid: Bool) {
    var v: UInt32 = valid ? 1 : 0
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pResolve)
    e.setTexture(raw, index: 0); e.setTexture(depth, index: 1); e.setTexture(history, index: 2); e.setTexture(out, index: 3)
    e.setBytes(&cam, length: MemoryLayout<Cam>.stride, index: 0); e.setBytes(&prev, length: MemoryLayout<Cam>.stride, index: 1)
    e.setBytes(&v, length: 4, index: 2)
    e.dispatchThreads(MTLSize(width: raw.width, height: raw.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    e.endEncoding()
  }

  func encodePresent(_ cb: MTLCommandBuffer, from tex: MTLTexture, pass: MTLRenderPassDescriptor) {
    let r = cb.makeRenderCommandEncoder(descriptor: pass)!
    r.setRenderPipelineState(pPresent); r.setFragmentTexture(tex, index: 0)
    r.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    r.endEncoding()
  }

  /// Total sand grains: airborne bits + the pile's counts. Must equal `grains` forever.
  func census() -> Int {
    memset(censusOut.contents(), 0, censusOut.length)
    let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pCensus)
    e.setBuffer(vox, offset: 0, index: 0); e.setBuffer(V[0], offset: 0, index: 1); e.setBuffer(censusOut, offset: 0, index: 2)
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
  var forward: SIMD3<Float> { SIMD3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)) }
  func uniforms(aspect: Float, shadows: Bool, stats: Bool = false) -> Cam {
    let f = forward, r = simd_normalize(simd_cross(f, SIMD3<Float>(0, 1, 0))), u = simd_cross(r, f)
    let sun = simd_normalize(SIMD3<Float>(-0.55, 0.70, -0.35))   // behind the default camera: the faces you see are lit
    return Cam(pos: SIMD4(pos, 0), fwd: SIMD4(f, 0), right: SIMD4(r, 0), up: SIMD4(u, 0), sun: SIMD4(sun, 0),
               tanHalf: tan(fovY / 2), aspect: aspect, shadows: shadows ? 1 : 0, stats: stats ? 1 : 0)
  }
}
