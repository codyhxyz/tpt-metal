import Metal
import simd
import Foundation
import QuartzCore

/// An edit, as the shaders see it (Shaders/World.metal, `WEdit`).
/// The water passes' parameters (Shaders/World.metal, `WaterParams`): counts and sub-step stamp, then G and its length.
struct WaterParams { var u: SIMD4<UInt32>; var f: SIMD4<Float> }

/// Indirect dispatch arguments for a list, and counters to zero (Shaders/World.metal, `ArgSpec`).
struct ArgSpec { var a: SIMD4<UInt32>; var zero: SIMD4<UInt32> }

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
  enum C: Int {
    case tBump = 0, bBump, tFree, bFree, cand, over, tCap, bCap, args, steps = 11, pick = 12
    case list0 = 20, list1, fresh, storage, chunkList, wake
    case aFresh = 26, aStorage = 29, aChunks = 32, aBricks = 35, aMove = 38, aWake = 41, aClear = 44
    case fBump = 48, fFree, fCap
  }
  // list capacities (W_MAX* in Shaders/World.metal)
  static let maxL: UInt32 = 1 << 20, maxN: UInt32 = 1 << 18, maxS: UInt32 = 1 << 21, maxC: UInt32 = 1 << 16, maxW: UInt32 = 1 << 16
  static let maxF = 1 << 18   // bricks with flowing water at most (W_MAXF)
  /// Cell size in metres, for gravity in the water's flow.
  static let cellMetres = 0.3

  let NW: Int, NH: Int
  let device: MTLDevice, queue: MTLCommandQueue, q4: MTL4CommandQueue
  let residency: MTLResidencySet, mapEvent: MTLSharedEvent
  private var mapValue: UInt64 = 0, waitedValue: UInt64 = 0
  let grid, owner, tableFree, brickFree, ctr: MTLBuffer
  let tables, bricks: SparsePool
  /// One flow slot index per table entry (0: none), the same layout as `tables`.
  let slots: SparsePool
  /// Water per column of a brick with flowing water: its +x/+z edges' flow and units crossing (8 bytes); the
  /// sub-step each slot was last written in; the slots' free stack.
  let flows, stamps, flowFree: MTLBuffer
  private var waterStamp: UInt32 = 0
  let pHeight, pSurface, pBands, pChunks, pBrickClass, pCandArgs, pBrickFill, pCollapse: MTLComputePipelineState
  let pEditChunks, pFixFree, pEditBricks, pRender: MTLComputePipelineState
  // physics (Shaders/World.metal, "Physics")
  let lists: [MTLBuffer], bits: [MTLBuffer]      // the awake list and its bitmap, this step's and the next
  let fresh, storage, chunkList, wake, cbits: MTLBuffer
  /// Per brick of this step's list: did the move pass see water in it (the water pass skips dry bricks).
  let wetFlag: MTLBuffer
  let pArgs, pStorageList, pClearBits, pStorageChunks, pExpandChunks, pExpandBricks, pMove, pWake, pWaterFlow, pWaterApply, pRetire: MTLComputePipelineState
  private var li = 0, stepNo: UInt32 = 0, stepDebt = 0.0, lastFrame = CACurrentMediaTime()
  /// Physics steps per second: the time scale of falling, since a step moves a grain at most one cell. Set by eye
  /// while playing (`,` halves it, `.` doubles it).
  var stepRate = 30.0
  let pPresent: MTLRenderPipelineState
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
    func fn(_ n: String) throws -> MTLFunction { try lib.makeFunction(name: n, constantValues: fc) }
    func k(_ n: String) throws -> MTLComputePipelineState { try dev.makeComputePipelineState(function: try fn(n)) }
    pHeight = try k("w_height"); pSurface = try k("w_surface"); pBands = try k("w_bands"); pChunks = try k("w_chunks")
    pBrickClass = try k("w_brickClass"); pCandArgs = try k("w_candArgs"); pBrickFill = try k("w_brickFill"); pCollapse = try k("w_collapse")
    pEditChunks = try k("w_editChunks"); pFixFree = try k("w_fixFree"); pEditBricks = try k("w_editBricks")
    pRender = try k("w_render")
    pArgs = try k("w_args"); pStorageList = try k("w_storageList"); pClearBits = try k("w_clearBits")
    pStorageChunks = try k("w_storageChunks"); pExpandChunks = try k("w_expandChunks"); pExpandBricks = try k("w_expandBricks")
    pMove = try k("w_move"); pWake = try k("w_wake"); pWaterFlow = try k("w_waterFlow"); pWaterApply = try k("w_waterApply"); pRetire = try k("w_retire")
    let rp = MTLRenderPipelineDescriptor()
    rp.vertexFunction = try fn("w_presentV"); rp.fragmentFunction = try fn("w_presentF")
    rp.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
    pPresent = try dev.makeRenderPipelineState(descriptor: rp)

    let nChunks = (NW / 64) * (NH / 64) * (NW / 64)
    grid = dev.makeBuffer(length: nChunks * 4, options: .storageModePrivate)!
    owner = dev.makeBuffer(length: nChunks * 4, options: .storageModePrivate)!
    tableFree = dev.makeBuffer(length: nChunks * 4, options: .storageModePrivate)!
    brickFree = dev.makeBuffer(length: (16 << 20) * 4, options: .storageModePrivate)!   // 16 M freed bricks outstanding
    ctr = dev.makeBuffer(length: 64 * 4, options: .storageModeShared)!
    tables = SparsePool(dev, virtualBytes: nChunks * 2048, elem: 2048)
    slots = SparsePool(dev, virtualBytes: nChunks * 2048, elem: 2048)
    flows = dev.makeBuffer(length: World.maxF * 64 * 8, options: .storageModePrivate)!
    flowFree = dev.makeBuffer(length: World.maxF * 4, options: .storageModePrivate)!
    stamps = dev.makeBuffer(length: World.maxF * 4, options: .storageModePrivate)!
    bricks = SparsePool(dev, virtualBytes: 64 << 30, elem: 512)                      // up to 128 M bricks
    residency.addAllocation(tables.buffer); residency.addAllocation(bricks.buffer); residency.addAllocation(slots.buffer)
    residency.addAllocation(flows); residency.addAllocation(flowFree); residency.addAllocation(stamps)
    func priv(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: n, options: .storageModePrivate)! }
    lists = [priv(Int(World.maxL) * 4), priv(Int(World.maxL) * 4)]
    let brickBits = (NW / 8) * (NH / 8) * (NW / 8)
    bits = [priv((brickBits + 31) / 32 * 4), priv((brickBits + 31) / 32 * 4)]
    fresh = priv(Int(World.maxN) * 4); storage = priv(Int(World.maxS) * 4)
    chunkList = priv(Int(World.maxC) * 4); wake = priv(Int(World.maxW) * 4)
    cbits = priv((nChunks + 31) / 32 * 4)
    wetFlag = priv(Int(World.maxL))
    for b in [grid, owner, tableFree, brickFree, ctr, fresh, storage, chunkList, wake, cbits, wetFlag] + lists + bits { residency.addAllocation(b) }
    residency.commit()
    generate()
  }

  // MARK: counters and memory

  private var counters: UnsafeMutablePointer<Int32> { ctr.contents().bindMemory(to: Int32.self, capacity: 64) }
  func counter(_ c: C) -> Int { Int(counters[c.rawValue]) }
  func setCounter(_ c: C, _ v: Int) { counters[c.rawValue] = Int32(v) }

  /// A command buffer that starts after every sparse mapping made so far.
  func makeCommandBuffer() -> MTLCommandBuffer {
    let cb = queue.makeCommandBuffer()!
    if waitedValue < mapValue { cb.encodeWaitForEvent(mapEvent, value: mapValue); waitedValue = mapValue }
    return cb
  }
  /// Makes the rest of `cb` wait for every sparse mapping made so far (call between encoders).
  func waitForMappings(_ cb: MTLCommandBuffer) {
    if waitedValue < mapValue { cb.encodeWaitForEvent(mapEvent, value: mapValue); waitedValue = mapValue }
  }
  /// Maps pool memory so `t` tables and `b` bricks fit; the next command buffer waits for it.
  private func reserve(tables t: Int, bricks b: Int) {
    let grewTables = tables.grow(to: t, dev: device, q4: q4, rs: residency)
    let grewSlots = slots.grow(to: t, dev: device, q4: q4, rs: residency)
    let grewT = grewTables || grewSlots
    let grewB = bricks.grow(to: b, dev: device, q4: q4, rs: residency)
    if grewT || grewB {
      residency.commit()
      mapValue += 1
      q4.signalEvent(mapEvent, value: mapValue)
    }
    setCounter(.tCap, tables.capacity); setCounter(.bCap, bricks.capacity); setCounter(.fCap, World.maxF)
  }
  /// Tables and bricks in use (handed out minus freed).
  var tablesUsed: Int { counter(.tBump) - counter(.tFree) }
  var bricksUsed: Int { counter(.bBump) - counter(.bFree) }
  /// Bytes of GPU memory the world holds.
  var bytes: Int { tables.mappedBytes + bricks.mappedBytes + grid.length + owner.length + tableFree.length + brickFree.length }

  // MARK: generation

  /// A fresh island: every edit undone. The pools keep their mapped memory and are filled again from the start.
  func reset() { generate() }   // generate() also clears the physics: counters, bitmaps, step

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
    // physics: nothing awake
    let cb = makeCommandBuffer(), bl = cb.makeBlitCommandEncoder()!
    for b in bits + [cbits] { bl.fill(buffer: b, range: 0..<b.length, value: 0) }
    bl.fill(buffer: slots.buffer, range: 0..<slots.mappedBytes, value: 0)       // no table has a flow slot yet
    bl.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    li = 0; stepNo = 0; stepDebt = 0
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
    e.setBuffer(owner, offset: 0, index: 2); e.setBuffer(tableFree, offset: 0, index: 4); e.setBuffer(slots.buffer, offset: 0, index: 6)
    e.dispatchThreads(chunkGrid, threadsPerThreadgroup: one)
    fix()
    e.setComputePipelineState(pEditBricks)
    e.setBuffer(bricks.buffer, offset: 0, index: 2); e.setBuffer(brickFree, offset: 0, index: 4); e.setBuffer(wake, offset: 0, index: 6)
    e.dispatchThreadgroups(brickGrid, threadsPerThreadgroup: tg128)
    fix()
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

  // MARK: physics

  /// Bricks in the list the next step will run.
  var awakeBricks: Int { counter(li == 0 ? .list0 : .list1) }

  /// One physics step (Shaders/World.metal, "Physics"), in one compute encoder.
  func encodeStep(_ cb: MTLCommandBuffer) {
    let cur = li, nxt = 1 - li
    let curCount = (cur == 0 ? C.list0 : C.list1).rawValue, nextCount = (nxt == 0 ? C.list0 : C.list1).rawValue
    let one = MTLSize(width: 1, height: 1, depth: 1), none = UInt32.max
    let e = cb.makeComputeCommandEncoder()!
    func args(_ count: Int, per: Int, arg: C, cap: UInt32, zero: [Int] = []) {
      var z = SIMD4<UInt32>(repeating: none)
      for (i, v) in zero.enumerated() { z[i] = UInt32(v) }
      var a = ArgSpec(a: SIMD4(UInt32(count), UInt32(per), UInt32(arg.rawValue), cap), zero: z)
      e.setComputePipelineState(pArgs); e.setBuffer(ctr, offset: 0, index: 0)
      e.setBytes(&a, length: MemoryLayout<ArgSpec>.stride, index: 1)
      e.dispatchThreads(one, threadsPerThreadgroup: one)
      e.memoryBarrier(scope: .buffers)
    }
    func run(_ p: MTLComputePipelineState, _ arg: C, _ buffers: [MTLBuffer]) {   // constants set beforehand with setBytes
      e.setComputePipelineState(p)
      for (i, b) in buffers.enumerated() { e.setBuffer(b, offset: 0, index: i) }
      e.dispatchThreadgroups(indirectBuffer: ctr, indirectBufferOffset: arg.rawValue * 4, threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      e.memoryBarrier(scope: .buffers)
    }
    func fixFree() {
      e.setComputePipelineState(pFixFree); e.setBuffer(ctr, offset: 0, index: 0)
      e.dispatchThreads(one, threadsPerThreadgroup: one); e.memoryBarrier(scope: .buffers)
    }
    // 1. storage for the bricks that just woke and their +x/+y/+z neighbours, listed once each through the idle bitmap
    args(C.fresh.rawValue, per: 64, arg: .aFresh, cap: World.maxN)
    run(pStorageList, .aFresh, [fresh, ctr, bits[nxt], storage])
    args(C.storage.rawValue, per: 64, arg: .aStorage, cap: World.maxS)
    var cc = SIMD2<UInt32>(UInt32(C.storage.rawValue), World.maxS)
    e.setBytes(&cc, length: 8, index: 3)
    run(pClearBits, .aStorage, [storage, ctr, bits[nxt]])
    run(pStorageChunks, .aStorage, [storage, ctr, grid, cbits, chunkList])
    args(C.chunkList.rawValue, per: 64, arg: .aChunks, cap: World.maxC)
    run(pExpandChunks, .aChunks, [grid, tables.buffer, owner, ctr, tableFree, chunkList, cbits, slots.buffer])
    fixFree()
    args(C.storage.rawValue, per: 1, arg: .aBricks, cap: World.maxS)
    run(pExpandBricks, .aBricks, [grid, tables.buffer, bricks.buffer, ctr, brickFree, storage])
    fixFree()
    // 2. move (the lists of fresh bricks, storage and chunks are used up; the next list starts empty)
    args(curCount, per: 1, arg: .aMove, cap: World.maxL, zero: [C.fresh.rawValue, C.storage.rawValue, C.chunkList.rawValue, nextCount])
    var mp = SIMD4<UInt32>(UInt32(curCount), stepNo & 1, stepNo, UInt32(nextCount))
    e.setBytes(&mp, length: 16, index: 9); e.setBuffer(wetFlag, offset: 0, index: 10)
    run(pMove, .aMove, [grid, tables.buffer, bricks.buffer, ctr, lists[cur], lists[nxt], fresh, bits[nxt], bits[cur]])
    // 2b. water flows by its head, in as many sub-steps as keep a wave under one column per sub-step (2D explicit
    // stability: sqrt(2) c dt / dx <= 1, c = sqrt(g depth), depth up to the 8 cells a brick holds)
    let dt = 1.0 / stepRate
    let c = (9.8 * 8 * World.cellMetres).squareRoot()
    let subs = max(1, Int((2.0.squareRoot() * c * dt / World.cellMetres).rounded(.up)))
    let ds = dt / Double(subs)
    e.setBuffer(wetFlag, offset: 0, index: 13); e.setBuffer(stamps, offset: 0, index: 14)
    for _ in 0..<subs {
      waterStamp &+= 1
      var wp = WaterParams(u: SIMD4(UInt32(curCount), waterStamp, UInt32(nextCount), 0),
                           f: SIMD4(Float(9.8 * ds * ds / World.cellMetres), Float(ds), 0, 0))
      e.setBytes(&wp, length: MemoryLayout<WaterParams>.stride, index: 12)
      run(pWaterFlow, .aMove, [grid, tables.buffer, bricks.buffer, ctr, lists[cur], lists[nxt], fresh, bits[nxt], bits[cur], slots.buffer, flows, flowFree])
      fixFree()
      e.setBuffer(wetFlag, offset: 0, index: 13); e.setBuffer(stamps, offset: 0, index: 14)
      e.setBytes(&wp, length: MemoryLayout<WaterParams>.stride, index: 12)
      run(pWaterApply, .aMove, [grid, tables.buffer, bricks.buffer, ctr, lists[cur], lists[nxt], fresh, bits[nxt], bits[cur], slots.buffer, flows])
      e.setBuffer(wetFlag, offset: 0, index: 13); e.setBuffer(stamps, offset: 0, index: 14)
    }
    // 3. bricks edits changed
    args(C.wake.rawValue, per: 64, arg: .aWake, cap: World.maxW)
    var nc = UInt32(nextCount)
    e.setBytes(&nc, length: 4, index: 6)
    run(pWake, .aWake, [wake, ctr, bits[nxt], bits[cur], lists[nxt], fresh])
    // 4. this step's bitmap back to clear; bricks that fell asleep give back their flow slots (the edits' list is used up)
    args(curCount, per: 64, arg: .aClear, cap: World.maxL, zero: [C.wake.rawValue])
    var lc = SIMD2<UInt32>(UInt32(curCount), World.maxL)
    e.setBytes(&lc, length: 8, index: 3)
    e.setBuffer(bits[nxt], offset: 0, index: 4); e.setBuffer(grid, offset: 0, index: 5)
    e.setBuffer(slots.buffer, offset: 0, index: 6); e.setBuffer(flowFree, offset: 0, index: 7)
    run(pRetire, .aClear, [lists[cur], ctr, bits[cur]])
    e.endEncoding()
    li = nxt; stepNo &+= 1
  }

  /// Once per frame: the tool, then a physics step when one is due and anything is awake. True if it encoded work.
  func frame(_ cb: MTLCommandBuffer, dig: Bool, place: Bool) -> Bool {
    let now = CACurrentMediaTime(), dt = min(now - lastFrame, 0.1)
    lastFrame = now
    let edited = tool(cb, dig: dig, place: place)
    // counters from the last finished step (a frame late at most: then this step waits a frame)
    if !edited && awakeBricks == 0 && counter(.fresh) == 0 && counter(.wake) == 0 { stepDebt = 0; return false }
    stepDebt = min(stepDebt + dt * stepRate, 1)
    if stepDebt < 1 { return edited }
    stepDebt -= 1
    reserve(tables: counter(.tBump) + 1024, bricks: counter(.bBump) + 8 * counter(.fresh) + (1 << 16))
    waitForMappings(cb)
    encodeStep(cb)
    return true
  }

  /// The tool, once per frame: hold dig or place to repeat every 0.1 s at the crosshair. True if it encoded an edit.
  func tool(_ cb: MTLCommandBuffer, dig: Bool, place: Bool) -> Bool {
    guard dig || place, let p = pick else { return false }
    let now = CACurrentMediaTime()
    if now - lastEdit < 0.1 { return false }
    lastEdit = now
    reserveForEdits()
    waitForMappings(cb)
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
