import Metal
import Foundation
import Mesher
import simd

/// One region's quads in the pool, as the vertex shaders read it (Shaders/World.metal, `RegionMesh`).
struct RegionMesh {
  /// xyz: the world cell where the region's interior starts; w: cells per voxel.
  var origin = SIMD4<Int32>()
  /// Solid quads: offset, count; water quads: offset, count (in quads, into `Mesher.quads`).
  var range = SIMD4<UInt32>()
  /// Where each face's solid quads start, relative to range.x (faces +y, -y, +x, -x, +z, -z); 6 used.
  var faceStart = SIMD8<UInt32>()
}

/// Turns regions of the world into quads for the rasterizer, with binary greedy meshing (Sources/Mesher).
///
/// The GPU gathers each region's cells, padded by one cell, into staging memory the CPU shares (no copy on unified
/// memory); a thread per core meshes them. Each region gets two meshes: solid faces with baked corner ambient
/// occlusion, and water faces (water against air only) for the water pass. Quads are 8 bytes, in one pool that the
/// vertex shaders pull from (`quadCorner` in Shaders/World.metal).
///
/// Frames: call `beginFrame()` once per frame, with at most 2 frames in flight. It returns that frame's copy of the
/// region table. Pool space and table entries a re-mesh replaces stay untouched until 3 frames later, so frames
/// already on the GPU keep drawing the old quads.
final class Mesher {
  struct Request {
    /// Caller's key for the region (stays the same when it is meshed again).
    var id: Int
    /// The world cell where the region's interior starts.
    var origin: SIMD3<Int32>
    /// Interior cells per side, at most 62.
    var size: Int
  }

  static let slotBytes = 393_216, masksOffset = 262_144, maskBytes = 32_768
  static let framesInFlight = 3

  let world: World
  /// The quad pool: 8 bytes per quad, written by the CPU, read by the vertex shaders.
  let quads: MTLBuffer
  let maxRegions: Int
  private let batch: Int
  private let staging, requests: MTLBuffer
  private let pGather: MTLComputePipelineState
  private let contexts: [OpaquePointer]
  private let lock = NSLock()
  private var pool: QuadAllocator
  private var slotOf: [Int: Int32] = [:]
  private var freeSlots: [Int32]
  private var table: [RegionMesh]
  private let tables: [MTLBuffer]
  private var dirty: [Set<Int32>]
  private var retired: [[Range<Int>]]
  private var retiredSlots: [[Int32]]
  private var frame = 0

  /// `quadCapacity` quads in the pool (8 bytes each), up to `maxRegions` regions, `batch` regions gathered at once.
  init(world: World, quadCapacity: Int = 32 << 20, maxRegions: Int = 1 << 16, batch: Int = 128) throws {
    self.world = world; self.maxRegions = maxRegions; self.batch = batch
    let dev = world.device
    quads = dev.makeBuffer(length: quadCapacity * 8, options: [.storageModeShared, .cpuCacheModeWriteCombined])!
    staging = dev.makeBuffer(length: batch * Self.slotBytes, options: .storageModeShared)!
    requests = dev.makeBuffer(length: batch * 32, options: .storageModeShared)!
    pGather = try dev.makeComputePipelineState(function: try world.function("w_gather"))
    contexts = (0..<ProcessInfo.processInfo.activeProcessorCount).map { _ in bgm_create() }
    pool = QuadAllocator(capacity: quadCapacity)
    freeSlots = (0..<Int32(maxRegions)).reversed()
    table = Array(repeating: RegionMesh(), count: maxRegions)
    tables = (0..<Self.framesInFlight).map { _ in
      let b = dev.makeBuffer(length: maxRegions * MemoryLayout<RegionMesh>.stride, options: .storageModeShared)!
      memset(b.contents(), 0, b.length)
      return b
    }
    dirty = Array(repeating: [], count: Self.framesInFlight)
    retired = Array(repeating: [], count: Self.framesInFlight)
    retiredSlots = Array(repeating: [], count: Self.framesInFlight)
  }

  deinit { for c in contexts { bgm_destroy(c) } }

  /// The table slot of region `id` (the baseInstance its draws use), if it is meshed.
  func slot(of id: Int) -> Int? { lock.lock(); defer { lock.unlock() }; return slotOf[id].map(Int.init) }

  /// Meshes (or re-meshes) these regions. Waits for the GPU gather, then meshes on all cores.
  func mesh(_ reqs: [Request]) {
    for start in stride(from: 0, to: reqs.count, by: batch) {
      let part = reqs[start..<min(start + batch, reqs.count)]
      let r = requests.contents().bindMemory(to: Int32.self, capacity: batch * 8)
      for (i, q) in part.enumerated() {
        precondition(q.size >= 1 && q.size <= Int(BGM_MAX), "a region is 1 to 62 cells on a side")
        r[i * 8 + 0] = q.origin.x; r[i * 8 + 1] = q.origin.y; r[i * 8 + 2] = q.origin.z; r[i * 8 + 3] = 1
        r[i * 8 + 4] = Int32(q.size); r[i * 8 + 5] = Int32(i)
      }
      let cb = world.makeCommandBuffer(), e = cb.makeComputeCommandEncoder()!
      e.setComputePipelineState(pGather)
      e.setBuffer(world.grid, offset: 0, index: 0); e.setBuffer(world.tables.buffer, offset: 0, index: 1)
      e.setBuffer(world.bricks.buffer, offset: 0, index: 2); e.setBuffer(requests, offset: 0, index: 3)
      e.setBuffer(staging, offset: 0, index: 4)
      e.dispatchThreads(MTLSize(width: 64, height: 64, depth: part.count), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
      e.endEncoding(); cb.commit(); cb.waitUntilCompleted()

      let items = Array(part), workers = contexts.count
      DispatchQueue.concurrentPerform(iterations: workers) { w in
        let ctx = contexts[w]
        for i in stride(from: w, to: items.count, by: workers) { meshOne(items[i], slot: i, ctx: ctx) }
      }
    }
  }

  /// Drops region `id`'s quads.
  func remove(id: Int) {
    lock.lock(); defer { lock.unlock() }
    guard let s = slotOf.removeValue(forKey: id) else { return }
    retire(table[Int(s)])
    table[Int(s)] = RegionMesh()
    markDirty(s)
    // The slot itself is reused only after the frames that may still draw it are done.
    retiredSlots[frame % Self.framesInFlight].append(s)
  }

  /// Once per frame, before encoding draws: the region table for this frame (bind it where the vertex shaders read
  /// `RegionMesh`), and the pool space retired 3 frames ago goes back to the pool.
  func beginFrame() -> MTLBuffer {
    lock.lock(); defer { lock.unlock() }
    frame += 1
    let i = frame % Self.framesInFlight
    for r in retired[i] { pool.release(r) }
    retired[i].removeAll(keepingCapacity: true)
    freeSlots.append(contentsOf: retiredSlots[i])
    retiredSlots[i].removeAll(keepingCapacity: true)
    let dst = tables[i].contents().bindMemory(to: RegionMesh.self, capacity: maxRegions)
    for s in dirty[i] { dst[Int(s)] = table[Int(s)] }
    dirty[i].removeAll(keepingCapacity: true)
    return tables[i]
  }

  /// Quads in use, out of the pool's capacity.
  var quadsUsed: Int { lock.lock(); defer { lock.unlock() }; return pool.used }

  // MARK: -

  private func meshOne(_ q: Request, slot: Int, ctx: OpaquePointer) {
    let base = staging.contents() + slot * Self.slotBytes
    let voxels = base.assumingMemoryBound(to: UInt8.self)
    let m = (base + Self.masksOffset).assumingMemoryBound(to: UInt64.self)
    let maskWords = Self.maskBytes / 8
    var solidStart = [Int32](repeating: 0, count: 6), waterStart = [Int32](repeating: 0, count: 6)
    // The context's quads are overwritten by its next call: copy the solid mesh out before meshing the water.
    let ns = Int(bgm_mesh(ctx, voxels, m, m + maskWords, 1, &solidStart))
    let so = place(ns, ctx)
    let nw = Int(bgm_mesh(ctx, voxels, m + 2 * maskWords, m + 3 * maskWords, 0, &waterStart))
    let wo = place(nw, ctx)
    lock.lock(); defer { lock.unlock() }

    var mesh = RegionMesh()
    mesh.origin = SIMD4(q.origin, 1)
    mesh.range = SIMD4(UInt32(so), UInt32(ns), UInt32(wo), UInt32(nw))
    for f in 0..<6 { mesh.faceStart[f] = UInt32(solidStart[f]) }
    let s: Int32
    if let old = slotOf[q.id] { s = old; retire(table[Int(s)]) }
    else { guard let f = freeSlots.popLast() else { fatalError("more than \(maxRegions) regions") }; s = f; slotOf[q.id] = s }
    table[Int(s)] = mesh
    markDirty(s)
  }

  /// Copies the context's last `n` quads into newly allocated pool space; returns where.
  private func place(_ n: Int, _ ctx: OpaquePointer) -> Int {
    if n == 0 { return 0 }
    lock.lock()
    let at = pool.allocate(n)
    lock.unlock()
    guard let at else { fatalError("quad pool full: \(n) more quads don't fit") }
    memcpy(quads.contents() + at * 8, bgm_quads(ctx), n * 8)
    return at
  }

  private func retire(_ m: RegionMesh) {   // with the lock held
    let i = frame % Self.framesInFlight
    if m.range.y > 0 { retired[i].append(Int(m.range.x) ..< Int(m.range.x + m.range.y)) }
    if m.range.w > 0 { retired[i].append(Int(m.range.z) ..< Int(m.range.z + m.range.w)) }
  }
  private func markDirty(_ s: Int32) { for i in 0..<Self.framesInFlight { dirty[i].insert(s) } }
}

/// First-fit allocation of quad ranges in the pool, with neighbouring free ranges merged on release.
struct QuadAllocator {
  private var free: [Range<Int>]   // sorted, disjoint, never adjacent
  let capacity: Int
  private(set) var used = 0
  init(capacity: Int) { self.capacity = capacity; free = [0..<capacity] }

  mutating func allocate(_ n: Int) -> Int? {
    for i in free.indices where free[i].count >= n {
      let s = free[i].lowerBound
      if free[i].count == n { free.remove(at: i) } else { free[i] = (s + n)..<free[i].upperBound }
      used += n
      return s
    }
    return nil
  }

  mutating func release(_ r: Range<Int>) {
    guard !r.isEmpty else { return }
    used -= r.count
    var lo = 0, hi = free.count
    while lo < hi { let mid = (lo + hi) / 2; if free[mid].lowerBound < r.lowerBound { lo = mid + 1 } else { hi = mid } }
    var merged = r, at = lo
    if at > 0 && free[at - 1].upperBound == merged.lowerBound { merged = free[at - 1].lowerBound..<merged.upperBound; free.remove(at: at - 1); at -= 1 }
    if at < free.count && free[at].lowerBound == merged.upperBound { merged = merged.lowerBound..<free[at].upperBound; free.remove(at: at) }
    free.insert(merged, at: at)
  }
}
