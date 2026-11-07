import Metal
import simd
import Foundation
import QuartzCore

/// An edit, as the shaders see it (Shaders/World.metal, `WEdit`).
struct WEdit {
  var c: SIMD4<Int32>, chunkLo: SIMD4<Int32>, brickLo: SIMD4<Int32>
  var r2: Int32, mat: UInt32, mode: UInt32, pad: UInt32 = 0
}

/// A Metal 4 placement-sparse buffer: a huge virtual range whose memory comes from placement heaps, mapped 64 KB
/// at a time only as the pool fills.
final class SparsePool {
  static let tile = 64 << 10
  let buffer: MTLBuffer, elem: Int, heapBytes: Int
  private(set) var tiles = 0
  private var heaps: [MTLHeap] = []
  private var heapUsed = 0

  init(_ dev: MTLDevice, virtualBytes: Int, elem: Int, heapBytes: Int = 64 << 20) {
    let len = (virtualBytes + Self.tile - 1) / Self.tile * Self.tile
    guard let b = dev.makeBuffer(length: len, options: [.storageModePrivate, .hazardTrackingModeTracked], placementSparsePageSize: .size64)
    else { fatalError("no placement-sparse buffer of \(len >> 20) MB (needs Metal 4 sparse buffer support)") }
    buffer = b; self.elem = elem; self.heapBytes = heapBytes
  }
  var capacity: Int { min(tiles * Self.tile, buffer.length) / elem }
  var mappedBytes: Int { tiles * Self.tile }
  var heapCount: Int { heaps.count }

  /// Maps tiles until `n` elements fit (as far as the virtual range allows). True if anything was mapped.
  @discardableResult
  func grow(to n: Int, dev: MTLDevice, q4: MTL4CommandQueue, rs: MTLResidencySet) -> Bool {
    let need = min((n * elem + Self.tile - 1) / Self.tile, buffer.length / Self.tile)
    if need <= tiles { return false }
    let heapTiles = heapBytes / Self.tile
    while tiles < need {
      if heaps.isEmpty || heapUsed == heapTiles {
        let hd = MTLHeapDescriptor()
        hd.type = .placement; hd.storageMode = .private; hd.size = heapBytes
        hd.maxCompatiblePlacementSparsePageSize = .size64
        guard let h = dev.makeHeap(descriptor: hd) else { fatalError("no placement heap of \(heapBytes >> 20) MB") }
        heaps.append(h); heapUsed = 0
        rs.addAllocation(h)
      }
      let k = min(need - tiles, heapTiles - heapUsed)
      q4.updateMappings(buffer: buffer, heap: heaps.last!, operations: [
        MTL4UpdateSparseBufferMappingOperation(mode: .map, bufferRange: NSRange(location: tiles, length: k), heapOffset: heapUsed)])
      tiles += k; heapUsed += k
    }
    return true
  }
}

/// A general voxel world: one byte of material per cell, stored as 64^3 chunks -> 8^3 bricks -> cells, uniform
/// boxes collapsed at every level (see Shaders/World.metal). Generated on the GPU as an island; drawn by rays;
/// edited by spheres whose cost follows the sphere, not the world.
final class World {
  // material ids (Shaders/World.metal)
  static let AIR: UInt32 = 0, ROCK: UInt32 = 1, SAND: UInt32 = 2, WATER: UInt32 = 3, PLANT: UInt32 = 4
  static let SNOW: UInt32 = 5, STONE: UInt32 = 6, BASALT: UInt32 = 7, WALL: UInt32 = 8
  static let names = ["air", "rock", "sand", "water", "plant", "snow", "stone", "basalt", "wall"]
  static let sea = 192

  // counters (Shaders/World.metal, C_*)
  enum C: Int { case tBump = 0, bBump, tFree, bFree, cand, over, tCap, bCap, args, steps = 11, pick = 12 }

  let NW: Int, NH: Int
  let device: MTLDevice, queue: MTLCommandQueue, q4: MTL4CommandQueue
  let residency: MTLResidencySet, mapEvent: MTLSharedEvent
  private var mapValue: UInt64 = 0, waitedValue: UInt64 = 0
  let grid, owner, tableFree, brickFree, ctr: MTLBuffer
  let tables, bricks: SparsePool
  let pHeight, pSurface, pBands, pChunks, pBrickClass, pCandArgs, pBrickFill, pCollapse: MTLComputePipelineState
  let pEditChunks, pFixFree, pEditBricks, pEditCollapse, pEditTables, pRender: MTLComputePipelineState
  let pPresent: MTLRenderPipelineState
  /// The shaders and the world's function constants, for the passes built outside this class (Mesher, SunShadows).
  let library: ShaderLib, constants: MTLFunctionConstantValues
  /// Generation report: (stage, GPU ms).
  private(set) var genStages: [(String, Double)] = []
  private(set) var genWall = 0.0

  // tool
  var material = World.SAND
  var radius: Int32 = 12
  private var lastEdit = 0.0

  var chunks: Int { (NW / 64) * (NH / 64) * (NW / 64) }

  init(size: Int = 16384, height: Int = 1024) throws {
    precondition(size % 64 == 0 && height % 64 == 0 && height <= 65535)
    NW = size; NH = height
    let dev = MTLCreateSystemDefaultDevice()!
    device = dev
    queue = dev.makeCommandQueue()!
    q4 = dev.makeMTL4CommandQueue()!
    mapEvent = dev.makeSharedEvent()!
    residency = try dev.makeResidencySet(descriptor: MTLResidencySetDescriptor())
    queue.addResidencySet(residency)

    let lib = try loadShaderLibrary(dev)
    let fc = MTLFunctionConstantValues()
    var consts: [Int32] = [Int32(NW), Int32(NH)]
    for i in 0..<consts.count { fc.setConstantValue(&consts[i], type: .int, index: i) }
    library = lib; constants = fc
    func fn(_ n: String) throws -> MTLFunction { try lib.makeFunction(name: n, constantValues: fc) }
    func k(_ n: String) throws -> MTLComputePipelineState { try dev.makeComputePipelineState(function: try fn(n)) }
    pHeight = try k("w_height"); pSurface = try k("w_surface"); pBands = try k("w_bands"); pChunks = try k("w_chunks")
    pBrickClass = try k("w_brickClass"); pCandArgs = try k("w_candArgs"); pBrickFill = try k("w_brickFill"); pCollapse = try k("w_collapse")
    pEditChunks = try k("w_editChunks"); pFixFree = try k("w_fixFree"); pEditBricks = try k("w_editBricks")
    pEditCollapse = try k("w_editCollapse"); pEditTables = try k("w_editTables"); pRender = try k("w_render")
    let rp = MTLRenderPipelineDescriptor()
    rp.vertexFunction = try fn("w_presentV"); rp.fragmentFunction = try fn("w_presentF")
    rp.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
    pPresent = try dev.makeRenderPipelineState(descriptor: rp)

    let nChunks = (NW / 64) * (NH / 64) * (NW / 64)
    grid = dev.makeBuffer(length: nChunks * 4, options: .storageModePrivate)!
    owner = dev.makeBuffer(length: nChunks * 4, options: .storageModePrivate)!
    tableFree = dev.makeBuffer(length: nChunks * 4, options: .storageModePrivate)!
    brickFree = dev.makeBuffer(length: (16 << 20) * 4, options: .storageModePrivate)!   // 16 M freed bricks outstanding
    ctr = dev.makeBuffer(length: 32 * 4, options: .storageModeShared)!
    tables = SparsePool(dev, virtualBytes: nChunks * 2048, elem: 2048)
    bricks = SparsePool(dev, virtualBytes: 64 << 30, elem: 512)                      // up to 128 M bricks
    residency.addAllocation(tables.buffer); residency.addAllocation(bricks.buffer)
    for b in [grid, owner, tableFree, brickFree, ctr] { residency.addAllocation(b) }
    residency.commit()
    generate()
  }

  /// A shader function from Shaders/World.metal, specialised for this world's size.
  func function(_ name: String) throws -> MTLFunction { try library.makeFunction(name: name, constantValues: constants) }

  // MARK: counters and memory

  private var counters: UnsafeMutablePointer<Int32> { ctr.contents().bindMemory(to: Int32.self, capacity: 32) }
  func counter(_ c: C) -> Int { Int(counters[c.rawValue]) }
  func setCounter(_ c: C, _ v: Int) { counters[c.rawValue] = Int32(v) }

  /// A command buffer that starts after every sparse mapping made so far.
  func makeCommandBuffer() -> MTLCommandBuffer {
    let cb = queue.makeCommandBuffer()!
    if waitedValue < mapValue { cb.encodeWaitForEvent(mapEvent, value: mapValue); waitedValue = mapValue }
    return cb
  }
  /// Maps pool memory so `t` tables and `b` bricks fit; the next command buffer waits for it.
  private func reserve(tables t: Int, bricks b: Int) {
    let grewT = tables.grow(to: t, dev: device, q4: q4, rs: residency)
    let grewB = bricks.grow(to: b, dev: device, q4: q4, rs: residency)
    if grewT || grewB {
      residency.commit()
      mapValue += 1
      q4.signalEvent(mapEvent, value: mapValue)
    }
    setCounter(.tCap, tables.capacity); setCounter(.bCap, bricks.capacity)
  }
  /// Tables and bricks in use (handed out minus freed).
  var tablesUsed: Int { counter(.tBump) - counter(.tFree) }
  var bricksUsed: Int { counter(.bBump) - counter(.bFree) }
  /// Bytes of GPU memory the world holds.
  var bytes: Int { tables.mappedBytes + bricks.mappedBytes + grid.length + owner.length + tableFree.length + brickFree.length }

  // MARK: generation

  /// A fresh island: every edit undone. The pools keep their mapped memory and are filled again from the start.
  func reset() { generate() }

  private func generate() {
    let t0 = CACurrentMediaTime()
    genStages = []
    memset(ctr.contents(), 0, ctr.length)
    let colBuf = device.makeBuffer(length: NW * NW * 4, options: .storageModeShared)!
    let bcol = device.makeBuffer(length: (NW / 8) * (NW / 8) * 4, options: .storageModePrivate)!
    let batch = 8192
    let cand = device.makeBuffer(length: batch * 512 * 4, options: .storageModePrivate)!
    func run(_ name: String, _ body: (MTLComputeCommandEncoder) -> Void) {
      let cb = makeCommandBuffer(), e = cb.makeComputeCommandEncoder()!
      body(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
      if let err = cb.error { fatalError("\(name): \(err)") }
      genStages.append((name, (cb.gpuEndTime - cb.gpuStartTime) * 1e3))
    }
    let n2 = MTLSize(width: NW, height: NW, depth: 1), tg2 = MTLSize(width: 16, height: 16, depth: 1)
    run("terrain") { e in
      e.setComputePipelineState(pHeight); e.setBuffer(colBuf, offset: 0, index: 0); e.dispatchThreads(n2, threadsPerThreadgroup: tg2)
      e.setComputePipelineState(pSurface); e.dispatchThreads(n2, threadsPerThreadgroup: tg2)
      e.setComputePipelineState(pBands); e.setBuffer(bcol, offset: 0, index: 1)
      e.dispatchThreads(MTLSize(width: NW / 8, height: NW / 8, depth: 1), threadsPerThreadgroup: tg2)
    }
    var mode: UInt32 = 0
    func chunks(_ e: MTLComputeCommandEncoder) {
      e.setComputePipelineState(pChunks)
      e.setBuffer(grid, offset: 0, index: 0); e.setBuffer(bcol, offset: 0, index: 1); e.setBuffer(ctr, offset: 0, index: 2)
      e.setBuffer(owner, offset: 0, index: 3); e.setBytes(&mode, length: 4, index: 4)
      e.dispatchThreads(MTLSize(width: NW / 64, height: NH / 64, depth: NW / 64), threadsPerThreadgroup: MTLSize(width: 8, height: 4, depth: 8))
    }
    run("chunks: count", chunks)
    let nTables = counter(.tBump)
    setCounter(.tBump, 0)
    reserve(tables: nTables + 64, bricks: 1 << 16)
    mode = 1
    run("chunks: assign", chunks)

    var classMs = 0.0, fillMs = 0.0
    var pending: MTLCommandBuffer? = nil
    for start in stride(from: 0, to: nTables, by: batch) {
      var range = SIMD2<UInt32>(UInt32(start), UInt32(min(batch, nTables - start)))
      setCounter(.cand, 0)
      let ca = makeCommandBuffer(), e = ca.makeComputeCommandEncoder()!
      e.setComputePipelineState(pBrickClass)
      e.setBuffer(tables.buffer, offset: 0, index: 0); e.setBuffer(owner, offset: 0, index: 1); e.setBuffer(bcol, offset: 0, index: 2)
      e.setBuffer(cand, offset: 0, index: 3); e.setBuffer(ctr, offset: 0, index: 4); e.setBytes(&range, length: 8, index: 5)
      e.dispatchThreads(MTLSize(width: Int(range.y) * 512, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      e.setComputePipelineState(pCandArgs); e.setBuffer(ctr, offset: 0, index: 0)
      e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
      e.endEncoding(); ca.commit(); ca.waitUntilCompleted()
      classMs += (ca.gpuEndTime - ca.gpuStartTime) * 1e3
      if let p = pending { p.waitUntilCompleted(); fillMs += (p.gpuEndTime - p.gpuStartTime) * 1e3 }
      // every candidate might need a brick: map that many first
      reserve(tables: nTables + 64, bricks: counter(.bBump) + counter(.cand) + (1 << 16))
      let cf = makeCommandBuffer(), f = cf.makeComputeCommandEncoder()!
      f.setComputePipelineState(pBrickFill)
      f.setBuffer(tables.buffer, offset: 0, index: 0); f.setBuffer(bricks.buffer, offset: 0, index: 1); f.setBuffer(owner, offset: 0, index: 2)
      f.setBuffer(cand, offset: 0, index: 3); f.setBuffer(colBuf, offset: 0, index: 4); f.setBuffer(ctr, offset: 0, index: 5)
      f.dispatchThreadgroups(indirectBuffer: ctr, indirectBufferOffset: C.args.rawValue * 4, threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      f.endEncoding(); cf.commit()
      pending = cf
    }
    if let p = pending { p.waitUntilCompleted(); fillMs += (p.gpuEndTime - p.gpuStartTime) * 1e3 }
    genStages.append(("bricks: classify", classMs)); genStages.append(("bricks: fill", fillMs))
    run("collapse") { e in
      var range = SIMD2<UInt32>(0, UInt32(nTables))
      e.setComputePipelineState(pCollapse)
      e.setBuffer(grid, offset: 0, index: 0); e.setBuffer(tables.buffer, offset: 0, index: 1); e.setBuffer(owner, offset: 0, index: 2)
      e.setBuffer(ctr, offset: 0, index: 3); e.setBuffer(tableFree, offset: 0, index: 4); e.setBytes(&range, length: 8, index: 5)
      e.dispatchThreads(MTLSize(width: nTables, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }
    if counter(.over) != 0 { fatalError("brick pool overflowed during generation") }
    genWall = CACurrentMediaTime() - t0
  }

  // MARK: editing

  enum Mode: UInt32 { case dig = 0, place = 1 }

  /// Digs out (or fills with `mat`, where there is air or water) every cell within `r` of `c`.
  func encodeEdit(_ cb: MTLCommandBuffer, at c: SIMD3<Int32>, radius r: Int32, mode: Mode, mat: UInt32 = 0) {
    let lo = c &- r, hi = c &+ r
    let cLo = lo &>> 6, cHi = hi &>> 6, bLo = lo &>> 3, bHi = hi &>> 3
    var E = WEdit(c: SIMD4(c, 0), chunkLo: SIMD4(cLo, 0), brickLo: SIMD4(bLo, 0), r2: r * r, mat: mat, mode: mode.rawValue)
    let cd = cHi &- cLo &+ 1, bd = bHi &- bLo &+ 1
    let chunkGrid = MTLSize(width: Int(cd.x), height: Int(cd.y), depth: Int(cd.z))
    let brickGrid = MTLSize(width: Int(bd.x), height: Int(bd.y), depth: Int(bd.z))
    let one = MTLSize(width: 1, height: 1, depth: 1), tg128 = MTLSize(width: 128, height: 1, depth: 1)
    let e = cb.makeComputeCommandEncoder()!
    e.setBuffer(grid, offset: 0, index: 0); e.setBuffer(tables.buffer, offset: 0, index: 1)
    e.setBuffer(ctr, offset: 0, index: 3); e.setBytes(&E, length: MemoryLayout<WEdit>.stride, index: 5)
    func fix() { e.setComputePipelineState(pFixFree); e.setBuffer(ctr, offset: 0, index: 0); e.dispatchThreads(one, threadsPerThreadgroup: one)
                 e.setBuffer(grid, offset: 0, index: 0) }
    e.setComputePipelineState(pEditChunks)
    e.setBuffer(owner, offset: 0, index: 2); e.setBuffer(tableFree, offset: 0, index: 4)
    e.dispatchThreads(chunkGrid, threadsPerThreadgroup: one)
    fix()
    e.setComputePipelineState(pEditBricks)
    e.setBuffer(bricks.buffer, offset: 0, index: 2); e.setBuffer(brickFree, offset: 0, index: 4)
    e.dispatchThreadgroups(brickGrid, threadsPerThreadgroup: tg128)
    fix()
    e.setComputePipelineState(pEditCollapse)
    e.dispatchThreadgroups(brickGrid, threadsPerThreadgroup: tg128)
    e.setComputePipelineState(pEditTables)
    e.setBuffer(owner, offset: 0, index: 2); e.setBuffer(tableFree, offset: 0, index: 4)
    e.dispatchThreads(chunkGrid, threadsPerThreadgroup: one)
    e.endEncoding()
  }

  /// Keeps enough pool memory mapped for the largest edit (radius 64: at most 17^3 new bricks, 27 new tables).
  func reserveForEdits() { reserve(tables: counter(.tBump) + 64, bricks: counter(.bBump) + 8192) }

  /// What the crosshair was on in the last finished frame: the cell and the face's outward normal.
  var pick: (cell: SIMD3<Int32>, normal: SIMD3<Int32>)? {
    let p = counters + C.pick.rawValue
    let code = Int(p[3])
    if code == 0 { return nil }
    let n = code - 1
    return (SIMD3(p[0], p[1], p[2]), SIMD3(Int32(n % 3 - 1), Int32(n / 3 % 3 - 1), Int32(n / 9 - 1)))
  }

  /// The tool, once per frame: hold dig or place to repeat every 0.1 s at the crosshair. True if it encoded an edit.
  func tool(_ cb: MTLCommandBuffer, dig: Bool, place: Bool) -> Bool {
    guard dig || place, let p = pick else { return false }
    let now = CACurrentMediaTime()
    if now - lastEdit < 0.1 { return false }
    lastEdit = now
    reserveForEdits()
    if dig { encodeEdit(cb, at: p.cell, radius: radius, mode: .dig) }
    else { encodeEdit(cb, at: p.cell &+ p.normal &* radius, radius: radius, mode: .place, mat: material) }
    return true
  }

  // MARK: drawing

  func encodeRender(_ cb: MTLCommandBuffer, to tex: MTLTexture, depth: MTLTexture, motion: MTLTexture, cam: inout Cam, prev: inout Cam) {
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pRender)
    e.setTexture(tex, index: 0); e.setTexture(depth, index: 1); e.setTexture(motion, index: 2)
    e.setBuffer(grid, offset: 0, index: 0); e.setBuffer(tables.buffer, offset: 0, index: 1); e.setBuffer(bricks.buffer, offset: 0, index: 2)
    e.setBytes(&cam, length: MemoryLayout<Cam>.stride, index: 3); e.setBytes(&prev, length: MemoryLayout<Cam>.stride, index: 4)
    e.setBuffer(ctr, offset: 0, index: 5)
    e.dispatchThreads(MTLSize(width: tex.width, height: tex.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    e.endEncoding()
  }

  func encodePresent(_ cb: MTLCommandBuffer, from tex: MTLTexture, pass: MTLRenderPassDescriptor) {
    let r = cb.makeRenderCommandEncoder(descriptor: pass)!
    r.setRenderPipelineState(pPresent); r.setFragmentTexture(tex, index: 0)
    r.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    r.endEncoding()
  }
}

extension FlyCam {
  /// Off the island's south coast, a little above the sea, looking inland.
  init(island w: World) {
    self.init()
    pos = SIMD3(Float(w.NW) * 0.5, Float(World.sea) + 220, Float(w.NW) * 0.13)
    yaw = 0; pitch = -0.12; speed = 150
  }
}
