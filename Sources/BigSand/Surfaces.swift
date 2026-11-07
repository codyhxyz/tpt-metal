import Metal
import Synchronization

/// Which mesh chunks have smooth surfaces, their BLASes and the TLAS (Shaders/Surface.h). The surfaces piece.
///
/// Three distances, centre to centre from the camera's mc, in mcs (so each set changes only when the camera crosses
/// an mc boundary, and is walked as differences of column spans):
///   reach  S.radius (32 mcs): mcs are meshed.
///   show   S.radius + 64 cells (34 mcs): a surface is in the TLAS and its solidCovered bit is set. The camera is at
///          most 28 cells from its mc's centre, so every shown mc's centre is within S.radius + 92 of it: the render
///          consults coverage out to S.radius + 96 and would draw a mesh beyond that twice (mesh and voxels).
///   keep   1.15 S.radius: BLASes stay cached for re-entry; beyond it they are dropped.
///
/// A frame (update, in its own command buffer, committed just before the render's):
///   liquids  Liquids.update (its own encoders)
///   build    AS encoder: BLASes from finished mesher runs' records; compacted copies of BLASes built earlier
///   jobs     compute: this frame's coverage ops and TLAS instance patches, the dirty scan round the camera, the
///            pre-filter and the mesher on the next jobs
///   tlas     AS encoder: this frame's BLASes' compacted sizes; the TLAS, if its instances or any BLAS changed
/// What the GPU writes for the CPU (scan lists, mesh records, compacted sizes) is read only once the frame that wrote
/// it has completed (the render buffer's completion handler counts frames): a frame or two late, never waited for.
///
/// Hazards. The BLASes live in untracked heaps, as does the scratch: one fence chains every AS encoder to the one
/// before it (this frame's and the last), so builds, copies, compacted sizes and the TLAS see finished BLASes and
/// the scratch is never shared between encoders at once. The render reads the TLAS (tracked) after this frame's tlas
/// encoder, hence after its BLASes. The TLAS is ping-ponged: a rebuild never writes the one the last frame bound,
/// and Metal's tracking holds the rare rebuild of the other one until any older frame reading it is done. Instances
/// and coverage change only through GPU copies from per-frame buffers, in the same command buffer as the TLAS that
/// matches them, so an older frame in flight sees old coverage with the old TLAS. A BLAS that is replaced or dropped
/// (and the staging a build read) is freed only once every frame that could still read it has completed.
final class Surfaces {
  unowned let world: World
  /// One bit per mc (mcIdx), set by physics and edits where cells changed: solids / liquids need meshing again.
  let solidDirty, liquidDirty: MTLBuffer
  /// One bit per mc: its solid surface is in the TLAS, or it has none (Surface.h).
  let covered: MTLBuffer
  let liquids: Liquids
  private let mesher: Mesher
  private let dev: MTLDevice
  private let pScan, pFilter, pCover, pPatch: MTLComputePipelineState
  private let mx: Int, my: Int   // mcs along x/z and along y

  // distances, mc units squared
  private let reach2: Int, show2: Int, keep2: Int, nb: Int

  // budgets
  private static let maxCand = 4096          // jobs handed to the pre-filter per run
  private static let maxJobs = 512           // of those, meshed per run (the rest go back in the queue)
  private static let vtxCap = 3 << 18, triCap = 1 << 20   // staging per run (37 MB)
  private static let maxBuilds = 512         // BLAS builds per frame
  private static let maxCompact = 512        // compacting copies per frame
  private static let maxPatch = 32768, maxOps = 32768, scanCap = 8192
  private static let instCap = 1 << 17       // TLAS instances (solid + liquid)
  private static let blasScratch = 64 << 20, heapBytes = 256 << 20
  private static let instStride = MemoryLayout<MTLIndirectAccelerationStructureInstanceDescriptor>.stride   // 72

  // per-mc state
  private static let kQ: UInt8 = 1    // in the job queue
  private static let kF: UInt8 = 2    // in a mesher run, or its BLAS waits to be built
  private static let kS: UInt8 = 4    // surface known: its BLAS is in `cache`, or it has none
  private static let kD: UInt8 = 8    // dirtied while F: mesh again when the result lands
  private static let kV: UInt8 = 16   // shown: covered, and its BLAS (if any) is a TLAS instance
  private static let kP: UInt8 = 32   // a coverage op is pending this frame
  private var st: [UInt8]
  private struct Blas { var a: MTLAccelerationStructure; var inst: Int32 }   // inst: its TLAS instance, -1 if not shown
  private var cache: [UInt32: Blas] = [:]
  private var buckets: [[UInt32]]   // the job queue: by distance in mcs when pushed, nearest first
  private var lowest: Int
  private var ref: SIMD3<Int>? = nil   // the camera's mc the sets were walked for

  // mesher runs: a staging set each, recycled once its builds have completed
  private enum RunState { case free, meshing, building }
  private final class Run {
    let staging: MeshStaging
    let jobs, meshJobs, empties, cnt, out: MTLBuffer
    var state = RunState.free, frame: UInt64 = 0, refs = 0, lastUse: UInt64 = 0
    init(_ dev: MTLDevice) {
      staging = MeshStaging(dev, vtxCap: Surfaces.vtxCap, triCap: Surfaces.triCap)
      func shared(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: n, options: .storageModeShared)! }
      jobs = shared(Surfaces.maxCand * 4); meshJobs = shared(Surfaces.maxCand * 4); empties = shared(Surfaces.maxCand * 4)
      cnt = shared(16); out = shared(Surfaces.maxJobs * MemoryLayout<MeshOut>.stride)
    }
  }
  private let runs: [Run]
  private struct Build { var mc: UInt32, run: Int, vtxOffset: UInt32, triOffset: UInt32, triCount: UInt32 }
  private var builds = Fifo<Build>()

  // per-frame upload/readback buffers, a ring of 4: reused only once the frame that used them has completed
  private final class Slot {
    let ops, patchIdx, patchWords, scan, scanCount, sizes: MTLBuffer
    var frame: UInt64 = 0
    init(_ dev: MTLDevice) {
      func shared(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: n, options: .storageModeShared)! }
      ops = shared(Surfaces.maxOps * 4); patchIdx = shared(Surfaces.maxPatch * 4)
      patchWords = shared(Surfaces.maxPatch * Surfaces.instStride)
      scan = shared(Surfaces.scanCap * 4); scanCount = shared(16); sizes = shared(Surfaces.maxBuilds * 4)
    }
  }
  private let slots: [Slot]
  private var frame: UInt64 = 0, harvested: UInt64 = 0
  private let done = Atomic<UInt64>(0)   // the last frame whose command buffer completed
  private var lastCB: MTLCommandBuffer? = nil

  // instances: a CPU mirror (solids [0, nS), liquids [nS, nS + nL)), patched into `inst` on the GPU
  private let inst: MTLBuffer
  private let mirror: UnsafeMutablePointer<MTLIndirectAccelerationStructureInstanceDescriptor>
  private var instMC: [UInt32]
  private var nS = 0, nL = 0
  private var patchStamp: [UInt64]
  private var patches: [Int32] = [], ops: [UInt32] = []

  // acceleration structures
  private var heaps: [MTLHeap] = []
  private let scratch: MTLBuffer
  private let fence: MTLFence
  private var fenceArmed = false
  private var tlas: [MTLAccelerationStructure?] = [nil, nil], tlasCap = [0, 0], cur = 0
  private var live = false, liveL = 0   // the bound TLAS has instances; of them, liquids
  private var grave: [(UInt64, MTLAccelerationStructure)] = [], graveHead = 0
  private var tlasGrave: [(UInt64, MTLAccelerationStructure)] = []
  private struct Compact { var mc: UInt32, a: MTLAccelerationStructure, frame: UInt64, k: Int, size: Int }
  private var compacts: [Compact] = [], cHead = 0, cRes = 0   // [cHead, cRes): compacted size known
  private var newBuilt: [MTLAccelerationStructure] = []
  private var residencyDirty = false

  private var pendingVis = Fifo<UInt32>(), pendingDrop = Fifo<UInt32>(), pendingDirty = Fifo<UInt32>()

  init(world: World, lib: ShaderLib, fc: MTLFunctionConstantValues) throws {
    self.world = world
    let dev = world.device, dims = S.dims(nw: world.NW, nh: world.NH), n = dims.count
    self.dev = dev; mx = dims.x; my = dims.y
    func priv(_ n: Int) -> MTLBuffer { dev.makeBuffer(length: n, options: .storageModePrivate)! }
    solidDirty = priv((n + 31) / 32 * 4); liquidDirty = priv((n + 31) / 32 * 4); covered = priv((n + 31) / 32 * 4)
    liquids = try Liquids(world: world, lib: lib, fc: fc)
    mesher = try Mesher(world: world, lib: lib, fc: fc)
    func k(_ name: String) throws -> MTLComputePipelineState {
      try dev.makeComputePipelineState(function: try lib.makeFunction(name: name, constantValues: fc))
    }
    pScan = try k("s_scan"); pFilter = try k("s_prefilter"); pCover = try k("s_cover"); pPatch = try k("s_patch")

    let r = Int(S.radius) / S.mc, s = (Int(S.radius) + 64) / S.mc, kf = Double(r) * 1.15
    reach2 = r * r; show2 = s * s; keep2 = Int(kf * kf)
    nb = Surfaces.isqrt(keep2) + 1
    st = [UInt8](repeating: 0, count: n)
    buckets = [[UInt32]](repeating: [], count: nb); lowest = nb
    runs = (0..<3).map { _ in Run(dev) }
    slots = (0..<4).map { _ in Slot(dev) }

    inst = priv(Surfaces.instCap * Surfaces.instStride)
    mirror = .allocate(capacity: Surfaces.instCap)
    instMC = [UInt32](repeating: 0, count: Surfaces.instCap)
    patchStamp = [UInt64](repeating: 0, count: Surfaces.instCap)
    patches.reserveCapacity(Surfaces.maxPatch); ops.reserveCapacity(Surfaces.maxOps)
    cache.reserveCapacity(1 << 16)
    // the scratch serves a frame's BLAS builds and then its TLAS build (at offset 0)
    let td = Surfaces.tlasDescriptor(inst, count: Surfaces.instCap)
    let tScratch = dev.accelerationStructureSizes(descriptor: td).buildScratchBufferSize
    scratch = dev.makeBuffer(length: max(Surfaces.blasScratch, tScratch), options: [.storageModePrivate, .hazardTrackingModeUntracked])!
    fence = dev.makeFence()!

    let rs = world.residency
    for b in [solidDirty, liquidDirty, covered, inst, scratch] { rs.addAllocation(b) }
    for u in runs {
      for b in [u.staging.verts, u.staging.indices, u.staging.prims, u.staging.counters, u.jobs, u.meshJobs, u.empties, u.cnt, u.out] { rs.addAllocation(b) }
    }
    for sl in slots { for b in [sl.ops, sl.patchIdx, sl.patchWords, sl.scan, sl.scanCount, sl.sizes] { rs.addAllocation(b) } }
    rs.commit()
    clearBits()
  }

  deinit { mirror.deallocate() }

  // MARK: frame

  /// Once per frame, in its own command buffer committed just before the render's (so its GPU time shows apart):
  /// mesh, build, and keep the TLAS current. Frames count as done when `render`, which reads what this builds, completes.
  func update(_ cb: MTLCommandBuffer, render: MTLCommandBuffer, camera: SIMD3<Float>) {
    frame += 1
    let f = frame
    render.addCompletedHandler { [weak self] _ in _ = self?.done.max(f, ordering: .releasing) }
    lastCB = render
    let done = self.done.load(ordering: .acquiring)
    releaseGrave(done)
    liquids.update(cb, camera: camera, liquidDirty: liquidDirty)
    let slot = slots[Int(f % 4)]
    if slot.frame > done { return }   // the GPU is 4 frames behind: nothing new this frame
    harvest(done)
    slot.frame = f

    let c = SIMD3<Int>(Int((camera.x / Float(S.mc)).rounded(.down)), Int((camera.y / Float(S.mc)).rounded(.down)),
                       Int((camera.z / Float(S.mc)).rounded(.down)))
    if ref != c { walk(from: ref, to: c) }

    patches.removeAll(keepingCapacity: true); ops.removeAll(keepingCapacity: true)
    var tlasDirty = syncLiquids()
    drainEvents()
    harvestRuns(done)
    encodeBuilds(cb)
    let run = nextRun(done)
    encodeJobs(cb, slot: slot, run: run)
    tlasDirty = tlasDirty || !patches.isEmpty
    if !newBuilt.isEmpty || tlasDirty {
      let e = cb.makeAccelerationStructureCommandEncoder()!
      if fenceArmed { e.waitForFence(fence) }
      for (k, a) in newBuilt.enumerated() { e.writeCompactedSize(accelerationStructure: a, buffer: slot.sizes, offset: k * 4) }
      if tlasDirty { buildTLAS(e, frame: f) }
      e.updateFence(fence); e.endEncoding(); fenceArmed = true
    }
    if residencyDirty { world.residency.commit(); residencyDirty = false }
  }

  /// The render's bindings S.bindTLAS ... S.bindInfo.
  func bind(_ e: MTLComputeCommandEncoder) {
    var info = SurfInfo()
    e.setBuffer(covered, offset: 0, index: S.bindSCover)
    e.setBuffer(liquids.covered, offset: 0, index: S.bindLCover)
    e.setBuffer(liquids.verts, offset: 0, index: S.bindLVerts)
    e.setBuffer(liquids.kinds, offset: 0, index: S.bindLKinds)
    if live, let t = tlas[cur] {
      e.setAccelerationStructure(t, bufferIndex: S.bindTLAS)
      info.enabled = 1
      info.liquids = UInt32(liveL)   // liquid instances in the bound TLAS (0: the render skips its liquid query)
    }
    e.setBytes(&info, length: MemoryLayout<SurfInfo>.stride, index: S.bindInfo)
  }

  /// The world was generated afresh: drop every surface.
  func reset() {
    // every frame so far has completed (buffers on one queue complete in order): nothing in flight reads a BLAS
    if let c = lastCB, c.status.rawValue >= MTLCommandBufferStatus.committed.rawValue {
      c.waitUntilCompleted(); _ = done.max(frame, ordering: .releasing)
    }
    grave.removeAll(keepingCapacity: true); graveHead = 0
    for (_, t) in tlasGrave { world.residency.removeAllocation(t) }
    tlasGrave.removeAll()
    cache.removeAll(keepingCapacity: true)
    nS = 0; nL = 0; live = false; liveL = 0
    st.withUnsafeMutableBufferPointer { _ = memset($0.baseAddress!, 0, $0.count) }
    for i in 0..<nb { buckets[i].removeAll(keepingCapacity: true) }
    lowest = nb; ref = nil
    pendingVis.clear(); pendingDrop.clear(); pendingDirty.clear(); builds.clear()
    compacts.removeAll(keepingCapacity: true); cHead = 0; cRes = 0
    for u in runs { u.state = .free; u.refs = 0 }
    harvested = frame
    liquids.reset()
    clearBits()
  }

  // MARK: the camera's sets

  private static func isqrt(_ v: Int) -> Int {   // floor(sqrt(v)), exactly as s_scan's sIsqrt
    var h = Int(Double(v).squareRoot())
    while h * h > v { h -= 1 }
    while (h + 1) * (h + 1) <= v { h += 1 }
    return h
  }
  private func d2(_ m: UInt32) -> Int {
    guard let r = ref else { return .max }
    let i = Int(m), y = i % my, c = i / my, x = c % mx, z = c / mx
    let dx = x - r.x, dy = y - r.y, dz = z - r.z
    return dx * dx + dy * dy + dz * dz
  }
  /// Column (dx, dz from c)'s mcs within `lim` of c: y in lo...hi (empty if lo > hi).
  private func span(_ dxz2: Int, _ lim: Int, _ cy: Int) -> (Int, Int) {
    let r = lim - dxz2
    if r < 0 { return (1, 0) }
    let h = Surfaces.isqrt(r)
    return (max(cy - h, 0), min(cy + h, my - 1))
  }
  /// Every mc within `lim` of n and not within it of o (all of n's if there is no o), a column span at a time.
  private func diff(_ o: SIMD3<Int>?, _ n: SIMD3<Int>, _ lim: Int, _ body: (UInt32) -> Void) {
    let h = Surfaces.isqrt(lim)
    for dz in -h...h {
      let z = n.z + dz
      if z < 0 || z >= mx { continue }
      for dx in -h...h {
        let x = n.x + dx
        if x < 0 || x >= mx { continue }
        let (lo, hi) = span(dx * dx + dz * dz, lim, n.y)
        if lo > hi { continue }
        let base = (z * mx + x) * my
        var olo = hi + 1, ohi = hi   // nothing of the span was within o's
        if let o {
          let ox = x - o.x, oz = z - o.z
          let s = span(ox * ox + oz * oz, lim, o.y)
          if s.0 <= s.1 { olo = s.0; ohi = s.1 }
        }
        if lo <= min(hi, olo - 1) { for y in lo...min(hi, olo - 1) { body(UInt32(base + y)) } }
        if max(lo, ohi + 1) <= hi && olo <= hi { for y in max(lo, ohi + 1)...hi { body(UInt32(base + y)) } }
      }
    }
  }
  /// The camera crossed into mc n: queue what came within reach, show / hide what crossed show, drop what left keep.
  private func walk(from o: SIMD3<Int>?, to n: SIMD3<Int>) {
    ref = n   // push measures from the new centre
    diff(o, n, reach2) { m in if st[Int(m)] & (Self.kQ | Self.kF | Self.kS) == 0 { push(m) } }
    if let o {
      diff(o, n, show2) { m in if st[Int(m)] & Self.kS != 0 { pendingVis.push(m) } }
      diff(n, o, show2) { m in if st[Int(m)] & Self.kV != 0 { pendingVis.push(m) } }
      diff(n, o, keep2) { m in if st[Int(m)] & Self.kS != 0 { pendingDrop.push(m) } }
    }
  }

  private func push(_ m: UInt32) {
    let i = Int(m)
    if st[i] & Self.kQ != 0 { return }
    st[i] |= Self.kQ
    let b = min(Surfaces.isqrt(d2(m)), nb - 1)
    buckets[b].append(m)
    lowest = min(lowest, b)
  }

  // MARK: events (each checks room for its patches and ops first; what doesn't fit waits a frame)

  private func room(_ p: Int, _ o: Int) -> Bool { patches.count + p <= Self.maxPatch && ops.count + o <= Self.maxOps }

  private func op(_ m: UInt32) {
    let i = Int(m)
    if st[i] & Self.kP != 0 { return }
    st[i] |= Self.kP
    ops.append(m)
  }
  private func patch(_ i: Int) {
    if patchStamp[i] == frame { return }
    patchStamp[i] = frame
    patches.append(Int32(i))
  }
  private static func desc(_ a: MTLAccelerationStructure, mask: UInt32, user: UInt32) -> MTLIndirectAccelerationStructureInstanceDescriptor {
    var d = MTLIndirectAccelerationStructureInstanceDescriptor()
    d.transformationMatrix = MTLPackedFloat4x3(columns: (MTLPackedFloat3Make(1, 0, 0), MTLPackedFloat3Make(0, 1, 0),
                                                         MTLPackedFloat3Make(0, 0, 1), MTLPackedFloat3Make(0, 0, 0)))
    d.options = []; d.mask = mask; d.intersectionFunctionTableOffset = 0; d.userID = user
    d.accelerationStructureID = a.gpuResourceID
    return d
  }
  /// A solid instance at nS (the first liquid moves to the end). Nil if the TLAS is full.
  private func addInst(_ m: UInt32, _ a: MTLAccelerationStructure) -> Int32? {
    if nS + nL >= Self.instCap { return nil }
    if nL > 0 { mirror[nS + nL] = mirror[nS]; patch(nS + nL) }
    mirror[nS] = Self.desc(a, mask: S.maskSolid, user: m); instMC[nS] = m; patch(nS)
    nS += 1
    return Int32(nS - 1)
  }
  /// The last solid fills the hole; the last liquid fills the last solid's place. The caller fixes its own entry.
  private func removeInst(_ i: Int) {
    let last = nS - 1
    if i != last { mirror[i] = mirror[last]; instMC[i] = instMC[last]; cache[instMC[i]]?.inst = Int32(i); patch(i) }
    if nL > 0 { mirror[last] = mirror[last + nL]; patch(last) }
    nS -= 1
  }
  private func setInst(_ i: Int, _ a: MTLAccelerationStructure) { mirror[i].accelerationStructureID = a.gpuResourceID; patch(i) }

  /// Shown iff the surface is known and within show. Up to 2 patches and an op.
  private func syncVis(_ m: UInt32) {
    let i = Int(m), s = st[i]
    let want = s & Self.kS != 0 && d2(m) <= show2
    if want == (s & Self.kV != 0) { return }
    if want {
      if let b = cache[m] {
        guard let k = addInst(m, b.a) else { return }   // TLAS full: the voxels draw it
        cache[m]!.inst = k
      }
      st[i] |= Self.kV
    } else {
      if let b = cache[m], b.inst >= 0 { removeInst(Int(b.inst)); cache[m]!.inst = -1 }
      st[i] &= ~Self.kV
    }
    op(m)
  }
  /// A mesher result for m (nil: no surface), within keep. Up to 4 patches and an op.
  private func apply(_ m: UInt32, _ a: MTLAccelerationStructure?) {
    let i = Int(m), s = st[i]
    var k: Int32 = -1
    if let old = cache[m] { k = old.inst; bury(old.a) }
    if let a {
      if k >= 0 { setInst(Int(k), a) }
      else if s & Self.kV != 0 { k = addInst(m, a) ?? -1; if k < 0 { st[i] &= ~Self.kV } }
      cache[m] = Blas(a: a, inst: k)
    } else {
      if k >= 0 { removeInst(Int(k)) }
      cache[m] = nil
    }
    st[i] = (st[i] & ~(Self.kF | Self.kD)) | Self.kS
    syncVis(m)
    op(m)   // always: the pre-filter may have set the bit on the GPU
    if s & Self.kD != 0 { push(m) }
  }
  /// m left keep: hide it and drop its BLAS. Up to 2 patches and an op.
  private func drop(_ m: UInt32) {
    let i = Int(m)
    if st[i] & Self.kV != 0 {
      if let b = cache[m], b.inst >= 0 { removeInst(Int(b.inst)) }
      st[i] &= ~Self.kV
      op(m)
    }
    if let b = cache.removeValue(forKey: m) { bury(b.a) }
    st[i] &= ~Self.kS
  }
  /// A result for m that is no longer wanted (m left keep while it was meshed).
  private func discard(_ m: UInt32) {
    st[Int(m)] &= ~(Self.kF | Self.kD)
    if st[Int(m)] & Self.kS != 0 { drop(m) }
    op(m)   // the pre-filter may have set the bit
  }

  private func drainEvents() {
    while let m = pendingVis.first, room(2, 1) { pendingVis.pop(); syncVis(m) }
    while let m = pendingDrop.first, room(2, 1) {
      pendingDrop.pop()
      if st[Int(m)] & Self.kS != 0 && d2(m) > keep2 { drop(m) }
    }
    while let m = pendingDirty.first, room(2, 1) {
      pendingDirty.pop()
      let s = st[Int(m)]
      if s & Self.kF != 0 { st[Int(m)] |= Self.kD }
      else if s & Self.kQ != 0 { continue }
      else if s & Self.kS != 0 { if d2(m) <= show2 { push(m) } else { drop(m) } }   // a stale cached BLAS is dropped
      else if d2(m) <= reach2 { push(m) }
    }
  }

  /// Liquid instances follow the solids; rewritten (where they differ) when the liquids piece says they changed.
  private func syncLiquids() -> Bool {
    if !liquids.changed { return false }
    let L = liquids.instances, n = min(L.count, Self.instCap - nS)
    if !room(n, 0) { return false }
    for k in 0..<n {
      let j = nS + k, id = L[k].blas.gpuResourceID
      if k < nL && mirror[j].accelerationStructureID._impl == id._impl && mirror[j].userID == L[k].slot { continue }
      mirror[j] = Self.desc(L[k].blas, mask: S.maskLiquid, user: L[k].slot); patch(j)
    }
    nL = n
    liquids.changed = false
    return true
  }

  // MARK: results

  /// Scan lists and compacted sizes of the frames that completed since the last look.
  private func harvest(_ done: UInt64) {
    let last = min(done, frame - 1)
    if harvested < last {
      for g in (harvested + 1)...last {
        let sl = slots[Int(g % 4)]
        if sl.frame != g { continue }
        let n = min(Int(sl.scanCount.contents().load(as: UInt32.self)), Self.scanCap)
        let p = sl.scan.contents().bindMemory(to: UInt32.self, capacity: Self.scanCap)
        for k in 0..<n { pendingDirty.push(p[k]) }
      }
      harvested = last
    }
    while cRes < compacts.count && compacts[cRes].frame <= done {
      let c = compacts[cRes], sl = slots[Int(c.frame % 4)]
      if sl.frame == c.frame { compacts[cRes].size = Int(sl.sizes.contents().load(fromByteOffset: c.k * 4, as: UInt32.self)) }
      cRes += 1
    }
  }

  /// Finished mesher runs: empties applied now, retries back in the queue, surfaces queued for building.
  private func harvestRuns(_ done: UInt64) {
    for (ri, u) in runs.enumerated() {
      if u.state == .building && u.refs == 0 && u.lastUse <= done { u.state = .free }
      if u.state != .meshing || u.frame > done { continue }
      let c = u.cnt.contents().bindMemory(to: UInt32.self, capacity: 4)
      let nMesh = min(Int(c[0]), Self.maxCand), nEmpty = min(Int(c[1]), Self.maxCand), nOut = min(nMesh, Self.maxJobs)
      if !room(4 * (nEmpty + nOut), nEmpty + nOut) { continue }
      let empties = u.empties.contents().bindMemory(to: UInt32.self, capacity: Self.maxCand)
      for k in 0..<nEmpty { let m = empties[k]; if d2(m) <= keep2 { apply(m, nil) } else { discard(m) } }
      let out = u.out.contents().bindMemory(to: MeshOut.self, capacity: Self.maxJobs)
      for j in 0..<nOut {
        let r = out[j], m = r.mc
        if r.triCount == S.retry { st[Int(m)] &= ~(Self.kF | Self.kD); push(m) }
        else if r.triCount == 0 { if d2(m) <= keep2 { apply(m, nil) } else { discard(m) } }
        else { builds.push(Build(mc: m, run: ri, vtxOffset: r.vtxOffset, triOffset: r.triOffset, triCount: r.triCount)); u.refs += 1 }
      }
      let jobs = u.meshJobs.contents().bindMemory(to: UInt32.self, capacity: Self.maxCand)
      for j in nOut..<max(nOut, nMesh) { let m = jobs[j]; st[Int(m)] &= ~(Self.kF | Self.kD); push(m) }   // past maxJobs: not meshed
      u.state = u.refs > 0 ? .building : .free
    }
  }

  // MARK: acceleration structures

  private func alloc(_ size: Int) -> MTLAccelerationStructure? {
    let sa = dev.heapAccelerationStructureSizeAndAlign(size: size)
    for h in heaps.reversed() where h.maxAvailableSize(alignment: sa.align) >= sa.size {
      if let a = h.makeAccelerationStructure(size: sa.size) { return a }
    }
    let hd = MTLHeapDescriptor()
    hd.type = .automatic; hd.storageMode = .private; hd.hazardTrackingMode = .untracked
    hd.size = max(Self.heapBytes, sa.size + sa.align)
    guard let h = dev.makeHeap(descriptor: hd) else { return nil }
    heaps.append(h); world.residency.addAllocation(h); residencyDirty = true
    return h.makeAccelerationStructure(size: sa.size)
  }
  /// Freed once every frame up to this one has completed.
  private func bury(_ a: MTLAccelerationStructure) { grave.append((frame, a)) }
  private func releaseGrave(_ done: UInt64) {
    while graveHead < grave.count && grave[graveHead].0 <= done { graveHead += 1 }
    if graveHead == grave.count { grave.removeAll(keepingCapacity: true); graveHead = 0 }
    else if graveHead >= 1024 && graveHead * 2 >= grave.count { grave.removeFirst(graveHead); graveHead = 0 }
    while let t = tlasGrave.first, t.0 <= done { world.residency.removeAllocation(t.1); tlasGrave.removeFirst(); residencyDirty = true }
  }

  /// BLASes for queued records, then compacting copies of BLASes whose compacted size is known.
  private func encodeBuilds(_ cb: MTLCommandBuffer) {
    newBuilt.removeAll(keepingCapacity: true)
    var e: MTLAccelerationStructureCommandEncoder? = nil
    func enc() -> MTLAccelerationStructureCommandEncoder {
      if let e { return e }
      let n = cb.makeAccelerationStructureCommandEncoder()!
      if fenceArmed { n.waitForFence(fence) }
      e = n
      return n
    }
    var used = 0
    while let b = builds.first, newBuilt.count < Self.maxBuilds, room(4, 1) {
      let u = runs[b.run]
      if d2(b.mc) > keep2 {
        builds.pop(); u.refs -= 1; discard(b.mc); continue
      }
      let g = MTLAccelerationStructureTriangleGeometryDescriptor()
      g.vertexBuffer = u.staging.verts; g.vertexBufferOffset = Int(b.vtxOffset) * 12; g.vertexStride = 12; g.vertexFormat = .float3
      g.indexBuffer = u.staging.indices; g.indexBufferOffset = Int(b.triOffset) * 12; g.indexType = .uint32
      g.triangleCount = Int(b.triCount)
      g.primitiveDataBuffer = u.staging.prims; g.primitiveDataBufferOffset = Int(b.triOffset) * 16
      g.primitiveDataStride = 16; g.primitiveDataElementSize = 16
      g.opaque = true
      let d = MTLPrimitiveAccelerationStructureDescriptor()
      d.geometryDescriptors = [g]
      let sz = dev.accelerationStructureSizes(descriptor: d)
      let off = (used + 255) & ~255
      if off + sz.buildScratchBufferSize > scratch.length { break }
      guard let a = alloc(sz.accelerationStructureSize) else { break }
      enc().build(accelerationStructure: a, descriptor: d, scratchBuffer: scratch, scratchBufferOffset: off)
      used = off + sz.buildScratchBufferSize
      builds.pop(); u.refs -= 1; u.lastUse = frame
      apply(b.mc, a)
      compacts.append(Compact(mc: b.mc, a: a, frame: frame, k: newBuilt.count, size: 0))
      newBuilt.append(a)
    }
    var n = 0
    while cHead < cRes, n < Self.maxCompact, room(1, 0) {
      let c = compacts[cHead]
      guard let b = cache[c.mc], b.a === c.a, c.size > 0, c.size < c.a.size else { cHead += 1; continue }
      guard let dst = alloc(c.size) else { break }
      enc().copyAndCompact(sourceAccelerationStructure: c.a, destinationAccelerationStructure: dst)
      bury(c.a)
      cache[c.mc]!.a = dst
      if b.inst >= 0 { setInst(Int(b.inst), dst) }
      cHead += 1; n += 1
    }
    if cHead == compacts.count { compacts.removeAll(keepingCapacity: true); cHead = 0; cRes = 0 }
    else if cHead >= 4096 && cHead * 2 >= compacts.count { compacts.removeFirst(cHead); cRes -= cHead; cHead = 0 }
    guard let e else { return }
    e.updateFence(fence); e.endEncoding(); fenceArmed = true
  }

  private static func tlasDescriptor(_ inst: MTLBuffer, count: Int) -> MTLInstanceAccelerationStructureDescriptor {
    let d = MTLInstanceAccelerationStructureDescriptor()
    d.instanceDescriptorType = .indirect
    d.instanceDescriptorBuffer = inst; d.instanceDescriptorBufferOffset = 0; d.instanceDescriptorStride = instStride
    d.instanceCount = count
    return d
  }
  /// Into the TLAS the last frame didn't bind (grown to a power of two if the instances outgrew it).
  private func buildTLAS(_ e: MTLAccelerationStructureCommandEncoder, frame f: UInt64) {
    let n = nS + nL
    if n == 0 { live = false; liveL = 0; return }   // nothing to trace: the render draws voxels only
    let next = 1 - cur
    if tlasCap[next] < n {
      var cap = 1024
      while cap < n { cap *= 2 }
      let size = dev.accelerationStructureSizes(descriptor: Self.tlasDescriptor(inst, count: cap)).accelerationStructureSize
      let t = dev.makeAccelerationStructure(size: size)!
      if let old = tlas[next] { tlasGrave.append((f, old)) }
      tlas[next] = t; tlasCap[next] = cap
      world.residency.addAllocation(t); residencyDirty = true
    }
    e.build(accelerationStructure: tlas[next]!, descriptor: Self.tlasDescriptor(inst, count: n), scratchBuffer: scratch, scratchBufferOffset: 0)
    cur = next; live = true; liveL = nL
  }

  // MARK: jobs

  /// A free run with the nearest queued mcs as its jobs, or nil.
  private func nextRun(_ done: UInt64) -> Run? {
    guard let u = runs.first(where: { $0.state == .free }), lowest < nb else { return nil }
    let jobs = u.jobs.contents().bindMemory(to: UInt32.self, capacity: Self.maxCand)
    var n = 0
    while n < Self.maxCand {
      while lowest < nb && buckets[lowest].isEmpty { lowest += 1 }
      if lowest == nb { break }
      let m = buckets[lowest].removeLast(), i = Int(m), s = st[i]
      if s & Self.kQ == 0 { continue }
      let d = d2(m)
      if s & Self.kS != 0 {
        if d > show2 { st[i] &= ~Self.kQ; pendingDirty.push(m); continue }   // dirty and cached out of sight: dropped
      } else if d > reach2 { st[i] &= ~Self.kQ; continue }
      let b = min(Surfaces.isqrt(d), nb - 1)
      if b > lowest { buckets[b].append(m); continue }   // the camera moved away from it since it was queued
      st[i] = (s & ~Self.kQ) | Self.kF
      jobs[n] = m; n += 1
    }
    if n == 0 { return nil }
    u.state = .meshing; u.frame = frame; u.refs = 0
    memset(u.cnt.contents(), 0, 16)
    let sc = u.staging.counters.contents().bindMemory(to: UInt32.self, capacity: 4)
    sc[S.scVtx] = 0; sc[S.scTri] = 0; sc[S.scVCap] = UInt32(Self.vtxCap); sc[S.scTCap] = UInt32(Self.triCap)
    jobCount = n
    return u
  }
  private var jobCount = 0

  /// Coverage ops, instance patches, the dirty scan, and the pre-filter and mesher for a new run.
  private func encodeJobs(_ cb: MTLCommandBuffer, slot: Slot, run: Run?) {
    let e = cb.makeComputeCommandEncoder()!
    let tg = MTLSize(width: 64, height: 1, depth: 1)
    if !ops.isEmpty {
      let p = slot.ops.contents().bindMemory(to: UInt32.self, capacity: Self.maxOps)
      for (k, m) in ops.enumerated() {
        let i = Int(m)
        p[k] = st[i] & Self.kV != 0 ? m : m | 0x8000_0000
        st[i] &= ~Self.kP
      }
      var n = UInt32(ops.count)
      e.setComputePipelineState(pCover)
      e.setBuffer(covered, offset: 0, index: 0); e.setBuffer(slot.ops, offset: 0, index: 1); e.setBytes(&n, length: 4, index: 2)
      e.dispatchThreads(MTLSize(width: ops.count, height: 1, depth: 1), threadsPerThreadgroup: tg)
    }
    if !patches.isEmpty {
      let idx = slot.patchIdx.contents().bindMemory(to: UInt32.self, capacity: Self.maxPatch)
      let words = slot.patchWords.contents().bindMemory(to: MTLIndirectAccelerationStructureInstanceDescriptor.self, capacity: Self.maxPatch)
      for (k, i) in patches.enumerated() { idx[k] = UInt32(i); words[k] = mirror[Int(i)] }
      var n = UInt32(patches.count)
      e.setComputePipelineState(pPatch)
      e.setBuffer(inst, offset: 0, index: 0); e.setBuffer(slot.patchIdx, offset: 0, index: 1)
      e.setBuffer(slot.patchWords, offset: 0, index: 2); e.setBytes(&n, length: 4, index: 3)
      e.dispatchThreads(MTLSize(width: patches.count * Self.instStride / 4, height: 1, depth: 1), threadsPerThreadgroup: tg)
    }
    e.memoryBarrier(scope: .buffers)
    // the dirty scan round the camera's mc, out to keep
    if let r = ref {
      memset(slot.scanCount.contents(), 0, 16)
      let h = Surfaces.isqrt(keep2)
      var p = SScan(ref: SIMD4(Int32(r.x), Int32(r.y), Int32(r.z), 0), keep2: Int32(keep2), reach: Int32(h), cap: Int32(Self.scanCap), pad: 0)
      e.setComputePipelineState(pScan)
      e.setBuffer(solidDirty, offset: 0, index: 0); e.setBuffer(slot.scan, offset: 0, index: 1)
      e.setBuffer(slot.scanCount, offset: 0, index: 2); e.setBytes(&p, length: MemoryLayout<SScan>.stride, index: 3)
      e.dispatchThreads(MTLSize(width: 2 * h + 1, height: 2 * h + 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    }
    if let u = run {
      e.setComputePipelineState(pFilter)
      e.setBuffer(world.grid, offset: 0, index: 0); e.setBuffer(world.tables.buffer, offset: 0, index: 1)
      e.setBuffer(u.jobs, offset: 0, index: 2); e.setBuffer(u.meshJobs, offset: 0, index: 3); e.setBuffer(u.empties, offset: 0, index: 4)
      e.setBuffer(u.cnt, offset: 0, index: 5); e.setBuffer(covered, offset: 0, index: 6)
      e.dispatchThreadgroups(MTLSize(width: jobCount, height: 1, depth: 1), threadsPerThreadgroup: tg)
      e.memoryBarrier(scope: .buffers)
      mesher.encode(e, jobs: u.meshJobs, count: u.cnt, countOffset: 0, maxJobs: Self.maxJobs, staging: u.staging, out: u.out)
    }
    e.endEncoding()
  }

  /// Dirty bits and coverage all clear (private memory starts undefined; a fresh world starts unmeshed).
  private func clearBits() {
    let cb = world.makeCommandBuffer(), bl = cb.makeBlitCommandEncoder()!
    for b in [solidDirty, liquidDirty, covered] { bl.fill(buffer: b, range: 0..<b.length, value: 0) }
    bl.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  }
}

/// Shaders/SurfaceJobs.h SScan.
private struct SScan { var ref: SIMD4<Int32>; var keep2, reach, cap, pad: Int32 }

/// A queue that pops from the front without moving the rest (trimmed now and then).
private struct Fifo<T> {
  private var a: [T] = []
  private var h = 0
  var first: T? { h < a.count ? a[h] : nil }
  mutating func push(_ v: T) { a.append(v) }
  mutating func pop() {
    h += 1
    if h == a.count { a.removeAll(keepingCapacity: true); h = 0 }
    else if h >= 4096 && h * 2 >= a.count { a.removeFirst(h); h = 0 }
  }
  mutating func clear() { a.removeAll(keepingCapacity: true); h = 0 }
}
