// Solid surfaces by surface nets: the mesher piece (Shaders/Mesh.h). Contract: Shaders/Surface.h.
import Foundation
import Metal

/// Solid surfaces of mesh chunks by surface nets (Shaders/Mesh.h): one threadgroup per job, sized by an indirect
/// dispatch whose arguments the GPU computes from the job count, so the CPU never waits on it.
final class Mesher {
  /// Threads per group (SN_T in Shaders/Mesh.h).
  static let threads = 512
  unowned let world: World
  let pArgs, pMesh: MTLComputePipelineState
  /// The mesh dispatch's indirect arguments.
  let args: MTLBuffer
  /// One word per staging vertex: its dual cube while meshing, then its octahedral normal for the triangles.
  private var scratch: MTLBuffer?

  init(world: World, lib: ShaderLib, fc: MTLFunctionConstantValues) throws {
    self.world = world
    let dev = world.device
    pArgs = try dev.makeComputePipelineState(function: try lib.makeFunction(name: "sn_args", constantValues: fc))
    pMesh = try dev.makeComputePipelineState(function: try lib.makeFunction(name: "sn_mesh", constantValues: fc))
    guard pMesh.maxTotalThreadsPerThreadgroup >= Mesher.threads else {
      throw NSError(domain: "BigSand", code: 3, userInfo: [NSLocalizedDescriptionKey:
        "sn_mesh runs at most \(pMesh.maxTotalThreadsPerThreadgroup) threads per group; it needs \(Mesher.threads)"])
    }
    args = dev.makeBuffer(length: 16, options: .storageModePrivate)!
    world.residency.addAllocation(args)
  }

  /// Meshes the solid surfaces of the mcs listed in `jobs` (UInt32 mcIdx each; their count is the UInt32 at byte
  /// `countOffset` of `count`, capped at `maxJobs`) into `staging`, and writes `out[j]` (MeshOut) for job j.
  /// Uses only `e`; leaves a memory barrier after its last dispatch.
  func encode(_ e: MTLComputeCommandEncoder, jobs: MTLBuffer, count: MTLBuffer, countOffset: Int, maxJobs: Int,
              staging: MeshStaging, out: MTLBuffer) {
    if (scratch?.length ?? 0) < staging.vtxCap * 4 {   // grows with the largest staging seen; bound below, so resident
      scratch = world.device.makeBuffer(length: max(staging.vtxCap, 1) * 4, options: .storageModePrivate)!
    }
    let one = MTLSize(width: 1, height: 1, depth: 1)
    var p = SIMD2<UInt32>(UInt32(countOffset / 4), UInt32(maxJobs))
    e.setComputePipelineState(pArgs)
    e.setBuffer(count, offset: 0, index: 0); e.setBuffer(args, offset: 0, index: 1); e.setBytes(&p, length: 8, index: 2)
    e.dispatchThreads(one, threadsPerThreadgroup: one)
    e.memoryBarrier(scope: .buffers)
    e.setComputePipelineState(pMesh)
    for (i, b) in [world.grid, world.tables.buffer, world.bricks.buffer, jobs, out,
                   staging.verts, staging.indices, staging.prims, staging.counters, scratch!].enumerated() {
      e.setBuffer(b, offset: 0, index: i)
    }
    e.dispatchThreadgroups(indirectBuffer: args, indirectBufferOffset: 0,
                           threadsPerThreadgroup: MTLSize(width: Mesher.threads, height: 1, depth: 1))
    e.memoryBarrier(scope: .buffers)
  }
}
