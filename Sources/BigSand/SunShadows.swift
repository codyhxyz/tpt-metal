import Metal
import simd

/// What the shadow pass's vertex shader reads (Shaders/World.metal, `ShadowPass`).
struct ShadowPass { var viewProj = simd_float4x4(); var eye = SIMD4<Int32>() }

/// What a fragment shader needs to look up sun shadows (Shaders/World.metal, `ShadowData`, `sunShadow`).
struct ShadowData {
  var viewProj: (simd_float4x4, simd_float4x4, simd_float4x4, simd_float4x4) = (.init(), .init(), .init(), .init())
  var splits = SIMD4<Float>()
  var texel = SIMD4<Float>()
  var normalOffset = SIMD4<Float>()
}

/// Cascaded shadow maps for the sun: 4 depth maps, each covering a slice of the view out to `distance` cells.
///
/// Slices split by the practical scheme (a blend of logarithmic and even splits). Each cascade is an orthographic
/// box around its slice's bounding sphere; the sphere's radius doesn't change as the camera turns, and its centre
/// snaps to whole shadow-map texels in world space, so shadow edges don't crawl as the camera moves.
/// Positions are relative to an integer `eye` cell, the same one the main pass uses (`quadCorner`).
final class SunShadows {
  static let cascades = 4
  let size: Int
  /// Depth32Float, one slice per cascade.
  let map: MTLTexture
  /// Shadows reach this far from the camera, in cells.
  var distance: Float = 3072
  /// 0: even splits, 1: logarithmic.
  var lambda: Float = 0.8
  /// How far behind each cascade (toward the sun) casters still count: the world's height covers every mountain.
  var casterReach: Float
  private(set) var data = ShadowData()
  private var passes = [ShadowPass](repeating: ShadowPass(), count: SunShadows.cascades)
  private let pipeline: MTLRenderPipelineState, depthState: MTLDepthStencilState

  init(world: World, size: Int = 2048) throws {
    self.size = size
    let dev = world.device
    let td = MTLTextureDescriptor()
    td.textureType = .type2DArray; td.pixelFormat = .depth32Float
    td.width = size; td.height = size; td.arrayLength = Self.cascades
    td.usage = [.renderTarget, .shaderRead]; td.storageMode = .private
    map = dev.makeTexture(descriptor: td)!
    let rp = MTLRenderPipelineDescriptor()
    rp.vertexFunction = try world.function("w_shadowV")
    rp.depthAttachmentPixelFormat = .depth32Float
    pipeline = try dev.makeRenderPipelineState(descriptor: rp)
    let dd = MTLDepthStencilDescriptor()
    dd.depthCompareFunction = .less; dd.isDepthWriteEnabled = true
    depthState = dev.makeDepthStencilState(descriptor: dd)!
    casterReach = Float(world.NH) * 1.5
  }

  /// Fits the cascades to the camera. `cam` comes from FlyCam.uniforms (pos, fwd, right, up, sun, tanHalf, aspect);
  /// `eye` is the integer cell positions are measured from.
  func update(cam: Cam, eye: SIMD3<Int32>) {
    let n: Float = 1, f = distance, N = Float(Self.cascades)
    var splits = [Float](repeating: 0, count: Self.cascades + 1)
    splits[0] = n
    for i in 1...Self.cascades {
      let t = Float(i) / N
      splits[i] = lambda * n * pow(f / n, t) + (1 - lambda) * (n + (f - n) * t)
    }
    let pos = SIMD3<Float>(cam.pos.x, cam.pos.y, cam.pos.z), fwd = SIMD3<Float>(cam.fwd.x, cam.fwd.y, cam.fwd.z)
    let L = simd_normalize(SIMD3<Float>(cam.sun.x, cam.sun.y, cam.sun.z))
    // light space: z points from the sun toward the ground
    let lz = -L, lx = simd_normalize(simd_cross(SIMD3<Float>(0, 1, 0), lz)), ly = simd_cross(lz, lx)
    let t2 = cam.tanHalf * cam.tanHalf * (1 + cam.aspect * cam.aspect)   // squared slope of the frustum's corner rays
    let eyeF = SIMD3<Float>(Float(eye.x), Float(eye.y), Float(eye.z))
    let texel = 1 / Float(size)
    for i in 0..<Self.cascades {
      let d0 = splits[i], d1 = splits[i + 1]
      // the smallest sphere around the slice's 8 corners: its centre on the view axis
      let zc = min((d0 + d1) * (1 + t2) / 2, d1)
      var r = ((zc - d0) * (zc - d0) + d0 * d0 * t2).squareRoot()
      r = max(r, ((d1 - zc) * (d1 - zc) + d1 * d1 * t2).squareRoot()).rounded(.up)
      let centre = pos + fwd * zc
      // snap the centre to whole texels, measured in world space so it holds still while the camera moves
      let world = 2 * r / Float(size)
      let sx = (simd_dot(centre, lx) / world).rounded(.down) * world
      let sy = (simd_dot(centre, ly) / world).rounded(.down) * world
      let o = lx * sx + ly * sy + lz * simd_dot(centre, lz) - eyeF     // the box's centre, eye-relative
      let view = simd_float4x4(rows: [
        SIMD4(lx, -simd_dot(lx, o)), SIMD4(ly, -simd_dot(ly, o)), SIMD4(lz, -simd_dot(lz, o)), SIMD4(0, 0, 0, 1)])
      let zn = -r - casterReach, zf = r
      let proj = simd_float4x4(rows: [
        SIMD4(1 / r, 0, 0, 0), SIMD4(0, 1 / r, 0, 0), SIMD4(0, 0, 1 / (zf - zn), -zn / (zf - zn)), SIMD4(0, 0, 0, 1)])
      let vp = proj * view
      passes[i] = ShadowPass(viewProj: vp, eye: SIMD4(eye, 0))
      switch i { case 0: data.viewProj.0 = vp; case 1: data.viewProj.1 = vp; case 2: data.viewProj.2 = vp; default: data.viewProj.3 = vp }
      data.normalOffset[i] = 1.5 * world
    }
    data.splits = SIMD4(splits[1], splits[2], splits[3], splits[4])
    data.texel = SIMD4(texel, 0, 0, 0)
  }

  /// The cascade's matrix (eye-relative position to clip space), for culling what `draw` submits.
  func viewProj(_ cascade: Int) -> simd_float4x4 { passes[cascade].viewProj }

  /// Renders the 4 cascades. `draw` binds the quad pool at vertex buffer 0 and this frame's region table at 1, then
  /// draws each region's solid quads: vertexStart = range.x * 6, vertexCount = range.y * 6, baseInstance = its slot.
  /// The pass state and buffer 2 (`ShadowPass`) are already set when it's called.
  func encode(_ cb: MTLCommandBuffer, draw: (MTLRenderCommandEncoder, Int) -> Void) {
    for i in 0..<Self.cascades {
      let rpd = MTLRenderPassDescriptor()
      rpd.depthAttachment.texture = map; rpd.depthAttachment.slice = i
      rpd.depthAttachment.loadAction = .clear; rpd.depthAttachment.clearDepth = 1; rpd.depthAttachment.storeAction = .store
      let e = cb.makeRenderCommandEncoder(descriptor: rpd)!
      e.setRenderPipelineState(pipeline); e.setDepthStencilState(depthState)
      e.setCullMode(.none)
      e.setDepthBias(1, slopeScale: 1.5, clamp: 0)
      var p = passes[i]
      e.setVertexBytes(&p, length: MemoryLayout<ShadowPass>.stride, index: 2)
      draw(e, i)
      e.endEncoding()
    }
  }
}
