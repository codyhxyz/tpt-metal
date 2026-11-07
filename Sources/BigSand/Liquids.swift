import Metal
import simd

/// Liquid surfaces as height fields per mesh chunk, refit in place (Shaders/Liquid.h). The liquids piece.
///
/// An mc near the camera that owns a liquid surface gets a slot: S.lVerts vertices in `verts`, 1024 kinds in `kinds`
/// and a BLAS of fixed topology. A slot's BLAS starts as a copy of a flat template grid refit to its heights, and is
/// refit in place whenever they change; a freed slot keeps its BLAS for the next mc, so steady state allocates nothing.
///
/// Each frame, in the render's command buffer before the TLAS (kernels in Shaders/Liquid.h):
///   ops      the CPU's changes to the per-mc bits: mcs scanned for the first time (known), mcs that left the
///            tracked ball (not known, covered or slotted), slots given and taken
///   dirty    liquidDirty of known mcs: coverage off (a slot's mesh is stale now), listed for a scan, bits cleared
///   scan     new mcs (nearest first, a budget a frame) and the dirty ones: does it own a liquid surface? Uniform
///            chunks answer most at once; the rest read their cells. One with none (and no slot) is covered at once.
///   heights  a budget of slots: kinds, vertices, coverage
///   refit    those slots' BLASes (a new slot's: from the template)
/// A frame's findings come back when its command buffer has finished: a dirty mc with a slot queues the slot for
/// heights, one with a surface and no slot queues a slot, an empty slot is freed. Meanwhile the mc isn't covered, so
/// the render walks its voxels (its liquid hits from the TLAS may be stale: the render must ignore them there).
/// The tracked ball: mcs whose centre is within S.radius of the camera's mc come in; beyond 1.15 x that they leave.
final class Liquids {
  /// Slot vertices (S.lVerts packed_float3 per slot) and kinds (1024 bytes per slot), as Shaders/Surface.h lays out.
  /// Replaced by larger copies when the slots outgrow them: bind them afresh each frame.
  private(set) var verts: MTLBuffer, kinds: MTLBuffer
  /// liquidCovered bits, one per mc.
  let covered: MTLBuffer
  /// The liquid instances to put in the TLAS (mask S.maskLiquid, user id = slot).
  private(set) var instances: [(blas: MTLAccelerationStructure, slot: UInt32)] = []
  /// True when instances or any liquid BLAS changed since the surfaces piece last cleared it.
  var changed = false

  // budgets and capacities (a frame record holds one frame's lists)
  private static let capScan = 16384          // mcs scanned a frame: new and dirty
  private static let capOps = 16384
  private static let maxJobs = 256            // slots computed and refit a frame
  private static let newPerFrame = 4096       // new mcs scanned a frame (most answer from their chunk)
  private static let records = 6              // frames in flight at most, plus room
  private static let perHeap = 64             // BLASes per heap
  // The ball holds ~3200 mc columns (~4200 out to where they leave); liquid surfaces are mostly the sea, one mc
  // thick, and inland none: 2048 slots cover a coast; open sea doubles it, once.
  private static let firstCap = 2048, maxCap = 16384
  private static let r2In: Int32 = 32 * 32    // (S.radius / 32)^2 in mcs: comes in
  private static let r2Out: Int32 = 1354      // (1.15 S.radius / 32)^2: leaves
  private static let half: Int32 = 38         // the square of mc columns liquidDirty is watched over (covers 1.15 R)
  private static let vBytes = S.lVerts * 12
  // a frame record's layout (bytes; Shaders/Liquid.h L_C_*)
  private enum Lay {
    static let ctl = 0, scan = 256, res = scan + capScan * 4, full = res + capScan
    static let ops = full + capScan * 4, jobs = ops + capOps * 8, jobRes = jobs + maxJobs * 8, bytes = jobRes + maxJobs * 4
  }
  private static let opClear: UInt32 = 0, opUnslot: UInt32 = 1, opKnow: UInt32 = 2, opSlot: UInt32 = 3

  /// Shaders/Liquid.h LDirtyArgs.
  private struct DirtyArgs { var x0, z0, w, h: Int32; var list, cap: UInt32; var pad0: UInt32 = 0, pad1: UInt32 = 0 }

  private final class Record {
    let buf: MTLBuffer
    var cb: MTLCommandBuffer? = nil
    var epoch = 0
    var jobs: [(slot: Int32, mc: UInt32)] = []
    init(_ b: MTLBuffer) { buf = b }
  }

  private unowned let world: World
  private let dev: MTLDevice
  private let mcx: Int32, mcy: Int32
  /// Per mc, as `covered`: scanned and tracked; has a slot.
  private let known, hasSlot: MTLBuffer
  private let pOps, pDirty, pArgs, pQuick, pFull, pHeights: MTLComputePipelineState
  private let index, templateVerts: MTLBuffer
  private let template: MTLAccelerationStructure
  private let blasSize: Int, heapBytes: Int
  private let scratch: MTLBuffer, refitScratch: Int
  private var heaps: [MTLHeap] = []

  // slots
  private var capacity: Int
  private var blas: [MTLAccelerationStructure] = []
  private var descs: [MTLPrimitiveAccelerationStructureDescriptor] = []
  private var ready: [Bool] = []          // copied from the template yet
  private var slotMc: [UInt32] = []       // ~0: free
  private var instAt: [Int] = []
  private var queued: [Bool] = []         // in dirtyQ
  private var jobStamp: [Int] = []        // frame of its last job
  private var recStamp: [Int] = []        // record harvested when it last had a job there
  private var freeSlots: [Int32] = []
  private var mcSlot: [UInt32: Int32] = [:]

  // tracking
  private var knownCols: [UInt32]         // per mc column: the y levels known
  private var knownList: [Int32] = []     // columns with any
  private var camMc = SIMD3<Int32>(Int32.min, 0, 0)
  private var cursor = 0
  private let offsets: [SIMD3<Int8>]      // mc offsets within the radius, nearest first

  // work
  private var dirtyQ: [Int32] = [], dirtyHead = 0
  private var allocQ: [UInt32] = [], allocHead = 0, allocSet = Set<UInt32>()
  private var opsA: [SIMD2<UInt32>] = [], opsB: [SIMD2<UInt32>] = []   // clears first, then sets (Liquid.h)
  private var recs: [Record] = [], flight: [Int] = []
  private var frameNo = 0, epoch = 0, harvestNo = 0
  private var retired: [(MTLBuffer, Int)] = []
  private var residencyDirty = false

  init(world: World, lib: ShaderLib, fc: MTLFunctionConstantValues) throws {
    precondition(world.NH / S.mc == 32, "one bit word per mc column (liquidDirty, liquidCovered)")
    self.world = world
    let dev = world.device
    self.dev = dev
    let d = S.dims(nw: world.NW, nh: world.NH)
    mcx = Int32(d.x); mcy = Int32(d.y)
    let bitBytes = (d.count + 31) / 32 * 4
    func priv(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: n, options: .storageModePrivate)! }
    covered = priv(bitBytes); known = priv(bitBytes); hasSlot = priv(bitBytes)
    capacity = Liquids.firstCap
    verts = priv(capacity * Liquids.vBytes); kinds = priv(capacity * 1024)
    func k(_ n: String) throws -> MTLComputePipelineState {
      let p = try dev.makeComputePipelineState(function: try lib.makeFunction(name: n, constantValues: fc))
      precondition(p.maxTotalThreadsPerThreadgroup >= 256, "\(n) needs 256 threads a group")
      return p
    }
    pOps = try k("l_ops"); pDirty = try k("l_dirty"); pArgs = try k("l_args")
    pQuick = try k("l_scanQuick"); pFull = try k("l_scanFull"); pHeights = try k("l_heights")

    // the shared index buffer (Surface.h's triangle order) and the flat grid every slot's BLAS is first built from
    var idx = [UInt16](), grid = [Float]()
    for z in 0..<32 { for x in 0..<32 {
      let v = UInt16(z * 33 + x)
      idx += [v, v + 33, v + 34, v, v + 34, v + 1]
    } }
    for z in 0...32 { for x in 0...32 { grid += [Float(x), 0, Float(z)] } }
    index = dev.makeBuffer(bytes: idx, length: idx.count * 2, options: .storageModeShared)!
    templateVerts = dev.makeBuffer(bytes: grid, length: grid.count * 4, options: .storageModeShared)!
    let tdesc = Liquids.descriptor(templateVerts, 0, index)
    let sizes = dev.accelerationStructureSizes(descriptor: tdesc)
    template = dev.makeAccelerationStructure(size: sizes.accelerationStructureSize)!
    // slot BLASes: as large as the template (a refit's destination must be), from tracked heaps so the TLAS build
    // that follows in the same command buffer waits for their refits
    let sa = dev.heapAccelerationStructureSizeAndAlign(size: sizes.accelerationStructureSize)
    blasSize = sa.size
    heapBytes = Liquids.perHeap * ((sa.size + sa.align - 1) / sa.align * sa.align) + sa.align
    refitScratch = (sizes.refitScratchBufferSize + 255) / 256 * 256
    scratch = priv(max(Liquids.maxJobs * refitScratch, 256))
    let buildScratch = priv(max(sizes.buildScratchBufferSize, 256))
    for _ in 0..<Liquids.records { recs.append(Record(dev.makeBuffer(length: Lay.bytes, options: .storageModeShared)!)) }
    knownCols = [UInt32](repeating: 0, count: d.x * d.x)

    var offs: [(Int32, SIMD3<Int8>)] = []
    for dz in -32...32 { for dy in -32...32 { for dx in -32...32 {
      let r2 = Int32(dx * dx + dy * dy + dz * dz)
      if r2 <= Liquids.r2In { offs.append((r2, SIMD3(Int8(dx), Int8(dy), Int8(dz)))) }
    } } }
    offsets = offs.sorted { $0.0 < $1.0 }.map { $0.1 }

    let rs = world.residency
    for b in [verts, kinds, covered, known, hasSlot, index, templateVerts, scratch, buildScratch] { rs.addAllocation(b) }
    rs.addAllocation(template)
    for r in recs { rs.addAllocation(r.buf) }
    rs.commit()
    let cb = world.makeCommandBuffer()
    let bl = cb.makeBlitCommandEncoder()!
    for b in [covered, known, hasSlot] { bl.fill(buffer: b, range: 0..<b.length, value: 0) }
    bl.endEncoding()
    let a = cb.makeAccelerationStructureCommandEncoder()!
    a.build(accelerationStructure: template, descriptor: tdesc, scratchBuffer: buildScratch, scratchBufferOffset: 0)
    a.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    rs.removeAllocation(buildScratch)
    rs.commit()
  }

  private static func descriptor(_ vb: MTLBuffer, _ off: Int, _ index: MTLBuffer) -> MTLPrimitiveAccelerationStructureDescriptor {
    let g = MTLAccelerationStructureTriangleGeometryDescriptor()
    g.vertexBuffer = vb; g.vertexBufferOffset = off; g.vertexStride = 12; g.vertexFormat = .float3
    g.indexBuffer = index; g.indexType = .uint16; g.triangleCount = S.lTris; g.opaque = true
    let d = MTLPrimitiveAccelerationStructureDescriptor()
    d.geometryDescriptors = [g]; d.usage = .refit
    return d
  }

  // MARK: per frame

  /// Once per frame, before the TLAS is built: liquid surfaces near `camera`, from the liquidDirty bits.
  func update(_ cb: MTLCommandBuffer, camera: SIMD3<Float>, liquidDirty: MTLBuffer) {
    frameNo += 1
    harvest()
    releaseRetired()
    let cm = SIMD3<Int32>((simd_clamp(camera, SIMD3(repeating: -1e6), SIMD3(repeating: 1e6)) / Float(S.mc)).rounded(.down))
    guard let ri = recs.indices.first(where: { recs[$0].cb == nil }) else {
      // every record in flight: only keep the coverage honest; the dirt waits for the next frame
      if camMc.x != Int32.min { encodeDirty(cb, liquidDirty, nil) }
      commitResidency()
      return
    }
    let r = recs[ri], p = r.buf.contents()
    if cm != camMc { evict(around: cm); camMc = cm; cursor = 0 }

    // new mcs, nearest first
    let scanList = (p + Lay.scan).bindMemory(to: UInt32.self, capacity: Liquids.capScan)
    var nNew = 0
    let budget = min(Liquids.newPerFrame, Liquids.capOps - Liquids.maxJobs - opsA.count - opsB.count)
    while cursor < offsets.count && nNew < budget {
      let m = camMc &+ SIMD3<Int32>(truncatingIfNeeded: offsets[cursor])
      cursor += 1
      if m.x < 0 || m.y < 0 || m.z < 0 || m.x >= mcx || m.y >= mcy || m.z >= mcx { continue }
      let col = Int(m.z * mcx + m.x), bit = UInt32(1) << UInt32(m.y)
      if knownCols[col] & bit != 0 { continue }
      if knownCols[col] == 0 { knownList.append(Int32(col)) }
      knownCols[col] |= bit
      opsB.append(SIMD2(UInt32(col) | Liquids.opKnow << 28, bit))
      scanList[nNew] = UInt32(col) * UInt32(mcy) + UInt32(m.y); nNew += 1
    }

    // jobs: stale slots first, then new ones
    var jobs: [(slot: Int32, mc: UInt32)] = []
    jobs.reserveCapacity(Liquids.maxJobs)
    while jobs.count < Liquids.maxJobs && dirtyHead < dirtyQ.count {
      let s = Int(dirtyQ[dirtyHead]); dirtyHead += 1; queued[s] = false
      if slotMc[s] == UInt32.max || jobStamp[s] == frameNo { continue }
      jobStamp[s] = frameNo; jobs.append((Int32(s), slotMc[s]))
    }
    if dirtyHead == dirtyQ.count { dirtyQ.removeAll(keepingCapacity: true); dirtyHead = 0 }
    while jobs.count < Liquids.maxJobs && allocHead < allocQ.count {
      let mc = allocQ[allocHead]
      if !isKnown(mc) || mcSlot[mc] != nil { allocHead += 1; allocSet.remove(mc); continue }
      guard let s = takeSlot(cb) else { break }
      allocHead += 1; allocSet.remove(mc)
      slotMc[s] = mc; mcSlot[mc] = Int32(s)
      instAt[s] = instances.count; instances.append((blas[s], UInt32(s))); changed = true
      opsB.append(SIMD2(mc >> 5 | Liquids.opSlot << 28, 1 << (mc & 31)))
      jobStamp[s] = frameNo; jobs.append((Int32(s), mc))
    }
    if allocHead == allocQ.count { allocQ.removeAll(keepingCapacity: true); allocHead = 0 }

    // the record: control words, ops, jobs
    let ctl = (p + Lay.ctl).bindMemory(to: UInt32.self, capacity: 16)
    ctl[0] = UInt32(nNew); ctl[1] = 0
    let ops = (p + Lay.ops).bindMemory(to: SIMD2<UInt32>.self, capacity: Liquids.capOps)
    let nA = opsA.count, nB = opsB.count
    for (i, o) in opsA.enumerated() { ops[i] = o }
    for (i, o) in opsB.enumerated() { ops[nA + i] = o }
    opsA.removeAll(keepingCapacity: true); opsB.removeAll(keepingCapacity: true)
    let jb = (p + Lay.jobs).bindMemory(to: SIMD2<UInt32>.self, capacity: Liquids.maxJobs)
    for (i, j) in jobs.enumerated() { jb[i] = SIMD2(UInt32(j.slot), j.mc) }

    let e = cb.makeComputeCommandEncoder()!
    let one = MTLSize(width: 1, height: 1, depth: 1), tg64 = MTLSize(width: 64, height: 1, depth: 1)
    let tg256 = MTLSize(width: 256, height: 1, depth: 1)
    func runOps(_ first: Int, _ n: Int) {
      if n == 0 { return }
      var range = SIMD2<UInt32>(UInt32(first), UInt32(n))
      e.setComputePipelineState(pOps)
      e.setBuffer(r.buf, offset: Lay.ops, index: 0); e.setBytes(&range, length: 8, index: 1)
      e.setBuffer(known, offset: 0, index: 2); e.setBuffer(covered, offset: 0, index: 3)
      e.setBuffer(hasSlot, offset: 0, index: 4); e.setBuffer(liquidDirty, offset: 0, index: 5)
      e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: tg64)
      e.memoryBarrier(scope: .buffers)
    }
    func args(_ count: Int, _ arg: Int, per: Int) {
      var a = SIMD4<UInt32>(UInt32(count), UInt32(Liquids.capScan), UInt32(arg), UInt32(per))
      e.setComputePipelineState(pArgs)
      e.setBuffer(r.buf, offset: Lay.ctl, index: 0); e.setBytes(&a, length: 16, index: 1)
      e.dispatchThreads(one, threadsPerThreadgroup: one)
      e.memoryBarrier(scope: .buffers)
    }
    runOps(0, nA)
    runOps(nA, nB)
    encodeDirty(cb, liquidDirty, (e, r.buf))
    args(0, 4, per: 64)                                     // L_C_SCAN -> L_C_AQ
    var cap = UInt32(Liquids.capScan)
    e.setComputePipelineState(pQuick)
    e.setBuffer(world.grid, offset: 0, index: 0); e.setBuffer(r.buf, offset: Lay.ctl, index: 1)
    e.setBuffer(r.buf, offset: Lay.scan, index: 2); e.setBuffer(r.buf, offset: Lay.res, index: 3)
    e.setBuffer(r.buf, offset: Lay.full, index: 4); e.setBuffer(hasSlot, offset: 0, index: 5)
    e.setBuffer(covered, offset: 0, index: 6); e.setBytes(&cap, length: 4, index: 7)
    e.dispatchThreadgroups(indirectBuffer: r.buf, indirectBufferOffset: Lay.ctl + 4 * 4, threadsPerThreadgroup: tg64)
    e.memoryBarrier(scope: .buffers)
    args(1, 8, per: 1)                                      // L_C_FULL -> L_C_AF
    e.setComputePipelineState(pFull)
    e.setBuffer(world.grid, offset: 0, index: 0); e.setBuffer(world.tables.buffer, offset: 0, index: 1)
    e.setBuffer(world.bricks.buffer, offset: 0, index: 2); e.setBuffer(r.buf, offset: Lay.ctl, index: 3)
    e.setBuffer(r.buf, offset: Lay.scan, index: 4); e.setBuffer(r.buf, offset: Lay.res, index: 5)
    e.setBuffer(r.buf, offset: Lay.full, index: 6); e.setBuffer(hasSlot, offset: 0, index: 7)
    e.setBuffer(covered, offset: 0, index: 8)
    e.dispatchThreadgroups(indirectBuffer: r.buf, indirectBufferOffset: Lay.ctl + 8 * 4, threadsPerThreadgroup: tg256)
    e.memoryBarrier(scope: .buffers)
    if !jobs.isEmpty {
      e.setComputePipelineState(pHeights)
      e.setBuffer(world.grid, offset: 0, index: 0); e.setBuffer(world.tables.buffer, offset: 0, index: 1)
      e.setBuffer(world.bricks.buffer, offset: 0, index: 2); e.setBuffer(r.buf, offset: Lay.jobs, index: 3)
      e.setBuffer(r.buf, offset: Lay.jobRes, index: 4); e.setBuffer(verts, offset: 0, index: 5)
      e.setBuffer(kinds, offset: 0, index: 6); e.setBuffer(covered, offset: 0, index: 7)
      e.dispatchThreadgroups(MTLSize(width: jobs.count, height: 1, depth: 1), threadsPerThreadgroup: tg256)
    }
    e.endEncoding()

    if !jobs.isEmpty {
      let a = cb.makeAccelerationStructureCommandEncoder()!
      for (j, job) in jobs.enumerated() {
        let s = Int(job.slot)
        a.refit(sourceAccelerationStructure: ready[s] ? blas[s] : template, descriptor: descs[s],
                destinationAccelerationStructure: blas[s], scratchBuffer: scratch, scratchBufferOffset: j * refitScratch)
        ready[s] = true
      }
      a.endEncoding()
      changed = true
    }
    r.cb = cb; r.epoch = epoch; r.jobs = jobs
    flight.append(ri)
    commitResidency()
  }

  /// The dirty pass, into `rec`'s encoder and scan list, or (nil) on its own, only taking coverage off.
  private func encodeDirty(_ cb: MTLCommandBuffer, _ dirty: MTLBuffer, _ rec: (MTLComputeCommandEncoder, MTLBuffer)?) {
    let x0 = max(camMc.x - Liquids.half, 0), x1 = min(camMc.x + Liquids.half, mcx - 1)
    let z0 = max(camMc.z - Liquids.half, 0), z1 = min(camMc.z + Liquids.half, mcx - 1)
    if x0 > x1 || z0 > z1 { return }
    var a = DirtyArgs(x0: x0, z0: z0, w: x1 - x0 + 1, h: z1 - z0 + 1, list: rec == nil ? 0 : 1, cap: UInt32(Liquids.capScan))
    let e = rec?.0 ?? cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pDirty)
    e.setBuffer(dirty, offset: 0, index: 0); e.setBuffer(known, offset: 0, index: 1); e.setBuffer(covered, offset: 0, index: 2)
    e.setBuffer(rec?.1 ?? covered, offset: Lay.ctl, index: 3); e.setBuffer(rec?.1 ?? covered, offset: rec == nil ? 0 : Lay.scan, index: 4)
    e.setBytes(&a, length: MemoryLayout<DirtyArgs>.stride, index: 5); e.setBuffer(hasSlot, offset: 0, index: 6)
    e.dispatchThreads(MTLSize(width: Int(a.w * a.h), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    if rec == nil { e.endEncoding() } else { e.memoryBarrier(scope: .buffers) }
  }

  /// What finished frames found, oldest first.
  private func harvest() {
    while let ri = flight.first, let cb = recs[ri].cb, cb.status == .completed || cb.status == .error {
      flight.removeFirst()
      let r = recs[ri]
      r.cb = nil
      guard r.epoch == epoch, cb.status == .completed else { continue }
      harvestNo += 1
      let p = r.buf.contents()
      let jobRes = (p + Lay.jobRes).bindMemory(to: UInt32.self, capacity: Liquids.maxJobs)
      for (k, j) in r.jobs.enumerated() where slotMc[Int(j.slot)] == j.mc {
        recStamp[Int(j.slot)] = harvestNo
        if jobRes[k] == 0 {                                   // owns no surface now
          release(Int(j.slot))
          opsA.append(SIMD2(j.mc >> 5 | Liquids.opUnslot << 28, 1 << (j.mc & 31)))
        }
      }
      let ctl = (p + Lay.ctl).bindMemory(to: UInt32.self, capacity: 16)
      let scan = (p + Lay.scan).bindMemory(to: UInt32.self, capacity: Liquids.capScan)
      let res = (p + Lay.res).bindMemory(to: UInt8.self, capacity: Liquids.capScan)
      for i in 0..<min(Int(ctl[0]), Liquids.capScan) {
        let mc = scan[i]
        if !isKnown(mc) { continue }
        if let s = mcSlot[mc] {
          // its heights in this frame were computed after its dirt: already current
          if recStamp[Int(s)] != harvestNo && !queued[Int(s)] { queued[Int(s)] = true; dirtyQ.append(s) }
        } else if res[i] != 0 && !allocSet.contains(mc) { allocSet.insert(mc); allocQ.append(mc) }
      }
    }
  }

  private func isKnown(_ mc: UInt32) -> Bool { knownCols[Int(mc >> 5)] >> (mc & 31) & 1 != 0 }

  /// Mcs beyond 1.15 R of `c` leave: not known, covered or slotted.
  private func evict(around c: SIMD3<Int32>) {
    var keep: [Int32] = []
    keep.reserveCapacity(knownList.count)
    for col in knownList {
      let x = col % mcx, z = col / mcx, h2 = (x - c.x) * (x - c.x) + (z - c.z) * (z - c.z)
      var bits = knownCols[Int(col)], out: UInt32 = 0, rest = bits
      while rest != 0 {
        let y = Int32(rest.trailingZeroBitCount); rest &= rest - 1
        if h2 + (y - c.y) * (y - c.y) > Liquids.r2Out { out |= 1 << UInt32(y) }
      }
      if out != 0 {
        opsA.append(SIMD2(UInt32(col) | Liquids.opClear << 28, out))
        bits &= ~out; knownCols[Int(col)] = bits
        while out != 0 {
          let y = UInt32(out.trailingZeroBitCount); out &= out - 1
          if let s = mcSlot[UInt32(col) * UInt32(mcy) + y] { release(Int(s)) }
        }
      }
      if bits != 0 { keep.append(col) }
    }
    knownList = keep
  }

  /// A slot back to the free list (its BLAS kept for the next mc); out of the TLAS.
  private func release(_ s: Int) {
    mcSlot[slotMc[s]] = nil
    slotMc[s] = UInt32.max
    let i = instAt[s], last = instances.count - 1
    if i != last { instances[i] = instances[last]; instAt[Int(instances[i].slot)] = i }
    instances.removeLast(); instAt[s] = -1
    freeSlots.append(Int32(s))
    changed = true
  }

  /// A free slot, a new one, or (growing the buffers) a new one past the old capacity; nil at the limit.
  private func takeSlot(_ cb: MTLCommandBuffer) -> Int? {
    if let s = freeSlots.popLast() { return Int(s) }
    let s = blas.count
    if s == capacity {
      if capacity == Liquids.maxCap { return nil }
      grow(cb)
    }
    if s / Liquids.perHeap == heaps.count {
      let hd = MTLHeapDescriptor()
      hd.storageMode = .private; hd.hazardTrackingMode = .tracked; hd.size = heapBytes
      let h = dev.makeHeap(descriptor: hd)!
      heaps.append(h); world.residency.addAllocation(h); residencyDirty = true
    }
    guard let a = heaps[s / Liquids.perHeap].makeAccelerationStructure(size: blasSize) else { return nil }
    blas.append(a)
    descs.append(Liquids.descriptor(verts, s * Liquids.vBytes, index))
    ready.append(false); slotMc.append(UInt32.max); instAt.append(-1); queued.append(false)
    jobStamp.append(0); recStamp.append(0)
    return s
  }

  /// Twice the slots: larger vertex and kind buffers, the old contents copied in this command buffer.
  private func grow(_ cb: MTLCommandBuffer) {
    let cap2 = capacity * 2
    let v2 = dev.makeBuffer(length: cap2 * Liquids.vBytes, options: .storageModePrivate)!
    let k2 = dev.makeBuffer(length: cap2 * 1024, options: .storageModePrivate)!
    let bl = cb.makeBlitCommandEncoder()!
    bl.copy(from: verts, sourceOffset: 0, to: v2, destinationOffset: 0, size: verts.length)
    bl.copy(from: kinds, sourceOffset: 0, to: k2, destinationOffset: 0, size: kinds.length)
    bl.endEncoding()
    world.residency.addAllocation(v2); world.residency.addAllocation(k2)
    retired.append((verts, frameNo + Liquids.records + 2)); retired.append((kinds, frameNo + Liquids.records + 2))
    verts = v2; kinds = k2; capacity = cap2; residencyDirty = true
    for d in descs { (d.geometryDescriptors![0] as! MTLAccelerationStructureTriangleGeometryDescriptor).vertexBuffer = v2 }
  }

  /// Old buffers leave the residency set once no frame in flight can use them.
  private func releaseRetired() {
    while let (b, f) = retired.first, f <= frameNo {
      world.residency.removeAllocation(b); retired.removeFirst(); residencyDirty = true
    }
  }

  private func commitResidency() {
    if residencyDirty { world.residency.commit(); residencyDirty = false }
  }

  /// The world was generated afresh: drop every slot.
  func reset() {
    epoch += 1                                                 // what frames in flight found is moot
    for s in 0..<slotMc.count where slotMc[s] != UInt32.max { release(s) }
    for s in 0..<queued.count { queued[s] = false }
    mcSlot.removeAll(); dirtyQ.removeAll(); dirtyHead = 0
    allocQ.removeAll(); allocHead = 0; allocSet.removeAll()
    opsA.removeAll(); opsB.removeAll()
    for c in knownList { knownCols[Int(c)] = 0 }
    knownList.removeAll()
    camMc = SIMD3(Int32.min, 0, 0); cursor = 0
    let cb = world.makeCommandBuffer(), bl = cb.makeBlitCommandEncoder()!
    for b in [covered, known, hasSlot] { bl.fill(buffer: b, range: 0..<b.length, value: 0) }
    bl.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    changed = true
  }
}
