import Metal

/// Mirrors of Shaders/Grain.h.
struct GrainPlan {
  var texels: UInt32 = 0, sites: UInt32 = 0, window: UInt32 = 0, levels: UInt32 = 0
  var tileM: Float = 0, texelM: Float = 0, siteM: Float = 0, d50: Float = 0
  var sigLn: Float = 0, zMax: Float = 0, pad0: Float = 0, pad1: Float = 0
}
struct GrainFrame {
  var org01 = SIMD4<Float>(), org2 = SIMD4<Float>()
  var tileM: Float = 0, texelM: Float = 0, levels: UInt32 = 0, period: UInt32 = 0
}
enum G { static let texA = 3, texB = 4, texH0 = 5, texH1 = 6, bindFrame = 12 }

/// Grain detail (Shaders/Grain.h): the grain tile, baked once on the GPU at launch, and the per-frame values the render
/// reads with it. Swapping in a geometric grain version replaces this class and grainDetail(), nothing else.
final class Grains {
  let plan: GrainPlan
  let a, b, h0, h1: MTLTexture
  /// Lattice units the hex tiling repeats after: the largest power of two whose float32 spacing is still within half
  /// a texel (a texel is 2 / texels lattice units: the lattice edge is half a tile), with a factor of 2 left for the
  /// distance from the camera to what it sees.
  let period: UInt32

  init(world: World, lib: ShaderLib, fc: MTLFunctionConstantValues) throws {
    let dev = world.device
    func k(_ name: String) throws -> MTLComputePipelineState {
      try dev.makeComputePipelineState(function: try lib.makeFunction(name: name, constantValues: fc))
    }
    let pPlan = try k("g_plan"), pHeight = try k("g_height"), pDerive = try k("g_derive")

    // 1. the plan, worked out on the GPU from the same tables the shaders use
    let planBuf = dev.makeBuffer(length: MemoryLayout<GrainPlan>.stride, options: .storageModeShared)!
    world.residency.addAllocation(planBuf); world.residency.commit()
    var cb = world.queue.makeCommandBuffer()!
    var e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pPlan); e.setBuffer(planBuf, offset: 0, index: 0)
    e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    plan = planBuf.contents().load(as: GrainPlan.self)
    let n = Int(plan.texels)
    period = UInt32(1) << UInt32((22 - plan.texels.trailingZeroBitCount))

    // 2. the textures
    func tex(_ f: MTLPixelFormat, mips: Bool) -> MTLTexture {
      let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: n, height: n, mipmapped: mips)
      d.usage = [.shaderRead, .shaderWrite]; d.storageMode = .private
      return dev.makeTexture(descriptor: d)!
    }
    let hu = tex(.rg32Float, mips: false)
    a = tex(.rgba16Float, mips: true); b = tex(.rgba16Float, mips: true)
    h0 = tex(.rgba8Unorm, mips: true); h1 = tex(.rgba8Unorm, mips: true)
    let hmax = dev.makeBuffer(length: 4, options: .storageModePrivate)!
    let rs = world.residency
    for t in [a, b, h0, h1, hu] as [MTLAllocation] { rs.addAllocation(t) }
    rs.addAllocation(hmax); rs.commit()

    // 3. bake: heights, then what shading needs, then the mip chains
    cb = world.queue.makeCommandBuffer()!
    let bl = cb.makeBlitCommandEncoder()!
    bl.fill(buffer: hmax, range: 0..<4, value: 0); bl.endEncoding()
    let grid = MTLSize(width: n, height: n, depth: 1), tg = MTLSize(width: 8, height: 8, depth: 1)
    e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pHeight)
    e.setTexture(hu, index: 0); e.setBuffer(planBuf, offset: 0, index: 0); e.setBuffer(hmax, offset: 0, index: 1)
    e.dispatchThreads(grid, threadsPerThreadgroup: tg)
    e.memoryBarrier(scope: [.textures, .buffers])
    e.setComputePipelineState(pDerive)
    e.setTexture(hu, index: 0); e.setTexture(a, index: 1); e.setTexture(b, index: 2); e.setTexture(h0, index: 3); e.setTexture(h1, index: 4)
    e.dispatchThreads(grid, threadsPerThreadgroup: tg)
    e.endEncoding()
    let mb = cb.makeBlitCommandEncoder()!
    for t in [a, b, h0, h1] { mb.generateMipmaps(for: t) }
    mb.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    rs.removeAllocation(hu); rs.removeAllocation(hmax); rs.removeAllocation(planBuf); rs.commit()
  }

  /// Per frame: the textures and the camera's place in each projection plane's hex lattice (Shaders/Grain.h).
  func bind(_ e: MTLComputeCommandEncoder, camPos: SIMD3<Float>, upscale: Float) {
    var f = GrainFrame()
    f.tileM = plan.tileM; f.texelM = plan.texelM; f.levels = plan.levels; f.period = period
    // lattice coordinates: twice the tile coordinate (a lattice edge is half a tile), skewed to the triangle grid
    let l = 2.0 * World.cellMetres / Double(plan.tileM), per = Double(period), r3 = 3.0.squareRoot()
    let p = SIMD3<Double>(Double(camPos.x), Double(camPos.y), Double(camPos.z))
    func org(_ u: Double, _ v: Double) -> SIMD2<Float> {
      let x = u * l, y = v * l
      var s = x - y / r3, t = 2.0 * y / r3
      s -= (s / per).rounded(.down) * per; t -= (t / per).rounded(.down) * per
      return SIMD2(Float(s), Float(t))
    }
    let o0 = org(p.z, p.y), o1 = org(p.x, p.z), o2 = org(p.x, p.y)
    f.org01 = SIMD4(o0.x, o0.y, o1.x, o1.y); f.org2 = SIMD4(o2.x, o2.y, upscale, 0)
    e.setTexture(a, index: G.texA); e.setTexture(b, index: G.texB); e.setTexture(h0, index: G.texH0); e.setTexture(h1, index: G.texH1)
    e.setBytes(&f, length: MemoryLayout<GrainFrame>.stride, index: G.bindFrame)
  }
}
