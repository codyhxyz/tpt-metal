import Metal

// Swift mirrors of Shaders/Surface.h: the CONTRACT between the four pieces that build smooth surfaces (mesher,
// surfaces, liquids, render). Read-only for every piece; change it only by agreement.

/// Per-triangle data in a solid BLAS (primitive data, 16 bytes): octahedral normals at vertices 0, 1, 2; material.
struct SurfPrim { var n0: UInt32, n1: UInt32, n2: UInt32, mat: UInt32 }
/// One mesher job's result (32 bytes). `triCount == S.retry`: staging was full, mesh it again later.
struct MeshOut { var mc, triCount, vtxOffset, vtxCount, triOffset, pad0, pad1, pad2: UInt32 }
/// The render's switch for the surface bindings.
struct SurfInfo { var enabled: UInt32 = 0, pad0: UInt32 = 0, pad1: UInt32 = 0, pad2: UInt32 = 0 }

enum S {
  static let mc = 32                       // mesh chunk side, cells
  static let radius: Float = 1024          // mcs whose centre is within this distance of the camera keep surfaces
  static let retry = UInt32.max
  static let maskSolid: UInt32 = 1, maskLiquid: UInt32 = 2
  static let scVtx = 0, scTri = 1, scVCap = 2, scTCap = 3   // staging counters
  static let bindTLAS = 6, bindSCover = 7, bindLCover = 8, bindLVerts = 9, bindLKinds = 10, bindInfo = 11
  static let lSide = 32, lVerts = 33 * 33, lTris = 2 * 32 * 32
  /// mcs along x/z and along y, and their count, for a world of nw x nh x nw cells.
  static func dims(nw: Int, nh: Int) -> (x: Int, y: Int, count: Int) { (nw / mc, nh / mc, (nw / mc) * (nw / mc) * (nh / mc)) }
  /// Shaders/Surface.h mcIdx.
  static func idx(_ m: SIMD3<Int32>, nw: Int, nh: Int) -> Int { (Int(m.z) * (nw / mc) + Int(m.x)) * (nh / mc) + Int(m.y) }
}

/// Staging for one mesher run (owned and recycled by the surfaces piece; filled by the mesher).
final class MeshStaging {
  let verts, indices, prims: MTLBuffer   // packed_float3 per vertex, 3 x UInt32 per triangle, SurfPrim per triangle
  let counters: MTLBuffer                // 4 x UInt32: S.scVtx, S.scTri handed out; S.scVCap, S.scTCap capacities
  let vtxCap: Int, triCap: Int
  init(_ dev: MTLDevice, vtxCap: Int, triCap: Int) {
    self.vtxCap = vtxCap; self.triCap = triCap
    verts = dev.makeBuffer(length: vtxCap * 12, options: .storageModePrivate)!
    indices = dev.makeBuffer(length: triCap * 12, options: .storageModePrivate)!
    prims = dev.makeBuffer(length: triCap * 16, options: .storageModePrivate)!
    counters = dev.makeBuffer(length: 16, options: .storageModeShared)!
  }
}
