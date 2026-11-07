import Metal

/// Liquid surfaces as height fields per mesh chunk, refit in place (Shaders/Liquid.h). The liquids piece. (Stub.)
final class Liquids {
  /// Slot vertices (S.lVerts packed_float3 per slot) and kinds (1024 bytes per slot), as Shaders/Surface.h lays out.
  let verts: MTLBuffer, kinds: MTLBuffer
  /// liquidCovered bits, one per mc.
  let covered: MTLBuffer
  /// The liquid instances to put in the TLAS (mask S.maskLiquid, user id = slot).
  private(set) var instances: [(blas: MTLAccelerationStructure, slot: UInt32)] = []
  /// True when instances or any liquid BLAS changed since the surfaces piece last cleared it.
  var changed = false
  init(world: World, lib: ShaderLib, fc: MTLFunctionConstantValues) throws {
    let dev = world.device, n = S.dims(nw: world.NW, nh: world.NH).count
    verts = dev.makeBuffer(length: S.lVerts * 12, options: .storageModePrivate)!
    kinds = dev.makeBuffer(length: 1024, options: .storageModePrivate)!
    covered = dev.makeBuffer(length: (n + 31) / 32 * 4, options: .storageModePrivate)!
    for b in [verts, kinds, covered] { world.residency.addAllocation(b) }
  }
  /// Once per frame, before the TLAS is built: liquid surfaces near `camera`, from the liquidDirty bits.
  func update(_ cb: MTLCommandBuffer, camera: SIMD3<Float>, liquidDirty: MTLBuffer) {}
  /// The world was generated afresh: drop every slot.
  func reset() {}
}
