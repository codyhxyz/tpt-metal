import Metal

/// Which mesh chunks have smooth surfaces, their BLASes and the TLAS (Shaders/Surface.h). The surfaces piece. (Stub.)
final class Surfaces {
  unowned let world: World
  /// One bit per mc (mcIdx), set by physics and edits where cells changed: solids / liquids need meshing again.
  let solidDirty, liquidDirty: MTLBuffer
  init(world: World, lib: ShaderLib, fc: MTLFunctionConstantValues) throws {
    self.world = world
    let dev = world.device, n = S.dims(nw: world.NW, nh: world.NH).count
    solidDirty = dev.makeBuffer(length: (n + 31) / 32 * 4, options: .storageModePrivate)!
    liquidDirty = dev.makeBuffer(length: (n + 31) / 32 * 4, options: .storageModePrivate)!
    for b in [solidDirty, liquidDirty] { world.residency.addAllocation(b) }
  }
  /// Once per frame, in the render's command buffer before its encoder: mesh, build, and keep the TLAS current.
  func update(_ cb: MTLCommandBuffer, camera: SIMD3<Float>) {}
  /// The render's bindings S.bindTLAS ... S.bindInfo.
  func bind(_ e: MTLComputeCommandEncoder) {
    var info = SurfInfo()
    e.setBytes(&info, length: MemoryLayout<SurfInfo>.stride, index: S.bindInfo)
  }
  /// The world was generated afresh: drop every surface.
  func reset() {}
}
