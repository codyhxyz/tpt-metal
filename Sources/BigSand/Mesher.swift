import Metal

/// Solid surfaces of mesh chunks by surface nets (Shaders/Mesh.h). The mesher piece. (Stub until it lands.)
final class Mesher {
  init(world: World, lib: ShaderLib, fc: MTLFunctionConstantValues) throws {}
  /// Meshes the solid surfaces of the mcs listed in `jobs` (UInt32 mcIdx each; their count is the UInt32 at byte
  /// `countOffset` of `count`, capped at `maxJobs`) into `staging`, and writes `out[j]` (MeshOut) for job j.
  /// Uses only `e`; leaves a memory barrier after its last dispatch.
  func encode(_ e: MTLComputeCommandEncoder, jobs: MTLBuffer, count: MTLBuffer, countOffset: Int, maxJobs: Int,
              staging: MeshStaging, out: MTLBuffer) {}
}
