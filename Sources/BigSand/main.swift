import AppKit
import MetalKit
import ImageIO
import UniformTypeIdentifiers
import MetalFX

// BigSand: a cube of sand falls and piles up.
//   swift run -c release BigSand                 window: WASD/QE fly, drag to look, Shift fast, R drop again, Space pause, F shadows
//   swift run -c release BigSand --shots [N]     headless: N ticks, a few saved frames, grain census


// MARK: - headless benchmark

func savePNG(_ tex: MTLTexture, _ engine: Engine, _ path: String) {
  let w = tex.width, h = tex.height
  let buf = engine.device.makeBuffer(length: w * h * 8, options: .storageModeShared)!
  let cb = engine.queue.makeCommandBuffer()!, b = cb.makeBlitCommandEncoder()!
  b.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(), sourceSize: MTLSize(width: w, height: h, depth: 1),
         to: buf, destinationOffset: 0, destinationBytesPerRow: w * 8, destinationBytesPerImage: w * h * 8)
  b.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  let src = buf.contents().bindMemory(to: Float16.self, capacity: w * h * 4)
  var px = [UInt8](repeating: 255, count: w * h * 4)
  for i in 0..<(w * h) { for c in 0..<3 {
    let v = max(0, min(1, Float(src[i * 4 + c])))
    px[i * 4 + c] = UInt8(255 * (v <= 0.0031308 ? v * 12.92 : 1.055 * pow(v, 1 / 2.4) - 0.055))
  } }
  let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
  let dst = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
  CGImageDestinationAddImage(dst, ctx.makeImage()!, nil); CGImageDestinationFinalize(dst)
}

func median(_ a: [Double]) -> Double { a.isEmpty ? 0 : a.sorted()[a.count / 2] }

/// Headless look: run `ticks` ticks, save a few frames, check that no grain was created or lost.
func shots(ticks: Int, outDir: String) throws {
  let e = try Engine()
  let before = e.census()
  let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1512, height: 982, mipmapped: false)
  td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .private
  let tex = e.device.makeTexture(descriptor: td)!
  td.pixelFormat = .r32Float
  let depth = e.device.makeTexture(descriptor: td)!
  td.pixelFormat = .rg16Float
  let motion = e.device.makeTexture(descriptor: td)!
  var cam = FlyCam().uniforms(aspect: 1512.0 / 982.0, shadows: true)
  let at: Set<Int> = [60, 90, 110, 140, 200, 300, ticks - 1]
  for i in 0..<ticks {
    let cb = e.queue.makeCommandBuffer()!
    e.encodeTick(cb)
    if at.contains(i) { var same = cam; e.encodeRender(cb, to: tex, depth: depth, motion: motion, cam: &cam, prev: &same) }
    cb.commit(); cb.waitUntilCompleted()
    if at.contains(i) { savePNG(tex, e, "\(outDir)/tick-\(i).png") }
  }
  let after = e.census()
  print("grains: \(before) before, \(after) after -> \(before == after ? "conserved" : "CHANGED BY \(after - before)")")
}

// MARK: - window

final class View: MTKView {
  var keys = Set<UInt16>(), shift = false
  var drag: ((CGFloat, CGFloat) -> Void)?
  override var acceptsFirstResponder: Bool { true }
  override func keyDown(with ev: NSEvent) { keys.insert(ev.keyCode) }
  override func keyUp(with ev: NSEvent) { keys.remove(ev.keyCode) }
  override func flagsChanged(with ev: NSEvent) { shift = ev.modifierFlags.contains(.shift) }
  override func mouseDragged(with ev: NSEvent) { drag?(ev.deltaX, ev.deltaY) }
}

/// Each displayed pair of refreshes: physics ticks once, the rays are traced at a third of the window's resolution
/// (MetalFX's largest supported upscale, if smaller), MetalFX's temporal scaler turns them into an antialiased frame at
/// full resolution, and MetalFX's frame interpolator builds the frame halfway between it and the previous one.
/// The display shows the in-between frame, then the real one: 120 Hz on ProMotion from 60 rendered frames.
final class Renderer: NSObject, MTKViewDelegate {
  let e: Engine; var cam = FlyCam(); var tex: MTLTexture?
  var depth: MTLTexture?, motion: MTLTexture?, scaler: MTLFXTemporalScaler?
  var outputs: [MTLTexture] = [], oi = 0            // scaler outputs: this frame's and the previous one's
  var interp: MTLFXFrameInterpolator?, between: MTLTexture?
  var pending: MTLTexture?                           // the real frame, shown on the refresh after its in-between frame
  var interpolate = false                            // only on a display that refreshes at 120 Hz or more
  var prev = Cam(), valid = false, frame = 0
  var running = true, shadows = true
  var last = CACurrentMediaTime(), lastTitle = CACurrentMediaTime()
  var simMs: [Double] = [], drawMs: [Double] = [], act: [Int] = []
  let lock = NSLock()
  weak var window: NSWindow?
  init(_ e: Engine) { self.e = e }

  func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { scaler = nil }

  private func make(_ f: MTLPixelFormat, _ w: Int, _ h: Int, _ use: MTLTextureUsage) -> MTLTexture {
    let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: w, height: h, mipmapped: false)
    td.usage = use; td.storageMode = .private
    return e.device.makeTexture(descriptor: td)!
  }

  /// Scaler, interpolator and their textures for a window of W x H pixels.
  private func build(_ view: MTKView, _ W: Int, _ H: Int) {
    let scale = max(1.0 / 3.0, Double(MTLFXTemporalScalerDescriptor.supportedInputContentMinScale(device: e.device)))
    let w = max(Int(Double(W) * scale), 1), h = max(Int(Double(H) * scale), 1)
    let d = MTLFXTemporalScalerDescriptor()
    d.inputWidth = w; d.inputHeight = h; d.outputWidth = W; d.outputHeight = H
    d.colorTextureFormat = .rgba16Float; d.depthTextureFormat = .r32Float
    d.motionTextureFormat = .rg16Float; d.outputTextureFormat = .rgba16Float
    guard let s = d.makeTemporalScaler(device: e.device) else { fatalError("MetalFX refused this temporal scaler setup") }
    interpolate = (view.window?.screen?.maximumFramesPerSecond ?? 60) >= 120 && MTLFXFrameInterpolatorDescriptor.supportsDevice(e.device)
    var ip: MTLFXFrameInterpolator? = nil
    if interpolate {
      let fd = MTLFXFrameInterpolatorDescriptor()
      fd.colorTextureFormat = .rgba16Float; fd.outputTextureFormat = .rgba16Float
      fd.depthTextureFormat = .r32Float; fd.motionTextureFormat = .rg16Float
      fd.inputWidth = w; fd.inputHeight = h; fd.outputWidth = W; fd.outputHeight = H
      fd.scaler = s
      ip = fd.makeFrameInterpolator(device: e.device)
      interpolate = ip != nil
    }
    var colorUse = s.colorTextureUsage.union(.shaderWrite), depthUse = s.depthTextureUsage.union(.shaderWrite)
    var motionUse = s.motionTextureUsage.union(.shaderWrite), outUse = s.outputTextureUsage.union(.shaderRead)
    if let ip { colorUse.formUnion(.shaderRead); depthUse.formUnion(ip.depthTextureUsage); motionUse.formUnion(ip.motionTextureUsage); outUse.formUnion(ip.colorTextureUsage) }
    tex = make(.rgba16Float, w, h, colorUse)
    depth = make(.r32Float, w, h, depthUse)
    motion = make(.rg16Float, w, h, motionUse)
    outputs = [make(.rgba16Float, W, H, outUse), make(.rgba16Float, W, H, outUse)]
    between = ip.map { make(.rgba16Float, W, H, $0.outputTextureUsage.union(.shaderRead)) }
    scaler = s; interp = ip; pending = nil; valid = false
    view.preferredFramesPerSecond = interpolate ? 120 : 60
  }

  func draw(in view: MTKView) {
    guard let v = view as? View, let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable else { return }
    // The refresh after an in-between frame shows the real frame rendered with it.
    if let p = pending {
      let cr = e.queue.makeCommandBuffer()!
      e.encodePresent(cr, from: p, pass: pass); cr.present(drawable); cr.commit()
      pending = nil
      return
    }
    let now = CACurrentMediaTime(), dt = Float(min(now - last, 0.1)); last = now
    // fly
    let speed: Float = (v.shift ? 2400 : 600) * dt
    let f = cam.forward, r = simd_normalize(simd_cross(f, SIMD3<Float>(0, 1, 0)))
    for k in v.keys { switch k {
      case 13: cam.pos += f * speed; case 1: cam.pos -= f * speed
      case 2: cam.pos += r * speed; case 0: cam.pos -= r * speed
      case 14: cam.pos.y += speed; case 12: cam.pos.y -= speed
      default: break } }
    if v.keys.remove(15) != nil { e.reset(); valid = false }  // R: drop it again (and start MetalFX's history over)
    if v.keys.remove(49) != nil { running.toggle() }          // Space
    if v.keys.remove(3) != nil { shadows.toggle() }           // F

    let W = max(Int(view.drawableSize.width), 2), H = max(Int(view.drawableSize.height), 2)
    if scaler == nil || outputs[0].width != W || outputs[0].height != H { build(view, W, H) }
    let s = scaler!, w = tex!.width, h = tex!.height
    if running {
      let cs = e.queue.makeCommandBuffer()!
      e.encodeTick(cs)
      cs.addCompletedHandler { [weak self] c in
        guard let self else { return }
        let a = self.e.activeTiles
        self.lock.lock(); self.simMs.append((c.gpuEndTime - c.gpuStartTime) * 1e3); self.act.append(a); self.lock.unlock()
      }
      cs.commit()
    }
    var u = cam.uniforms(aspect: Float(w) / Float(h), shadows: shadows)
    // a new sub-pixel sample point every frame: the Halton (2, 3) sequence, 16 points
    func halton(_ i: Int, _ b: Int) -> Float { var f: Float = 1, r: Float = 0, n = i; while n > 0 { f /= Float(b); r += f * Float(n % b); n /= b }; return r }
    frame += 1
    u.jitter = SIMD2(halton(frame % 16 + 1, 2) - 0.5, halton(frame % 16 + 1, 3) - 0.5)
    if !valid { prev = u }
    let cr = e.queue.makeCommandBuffer()!
    e.encodeRender(cr, to: tex!, depth: depth!, motion: motion!, cam: &u, prev: &prev)
    let out = outputs[oi]
    s.colorTexture = tex; s.depthTexture = depth; s.motionTexture = motion; s.outputTexture = out
    s.inputContentWidth = w; s.inputContentHeight = h
    s.jitterOffsetX = u.jitter.x; s.jitterOffsetY = u.jitter.y
    s.motionVectorScaleX = 1; s.motionVectorScaleY = 1
    s.isDepthReversed = true
    s.reset = !valid
    s.encode(commandBuffer: cr)
    if let ip = interp, valid {
      ip.colorTexture = out; ip.prevColorTexture = outputs[1 - oi]
      ip.depthTexture = depth; ip.motionTexture = motion; ip.outputTexture = between
      ip.motionVectorScaleX = 1; ip.motionVectorScaleY = 1
      ip.jitterOffsetX = u.jitter.x; ip.jitterOffsetY = u.jitter.y
      ip.deltaTime = 1.0 / 60.0
      ip.nearPlane = 1; ip.farPlane = 1e6                     // the rays' depth: reversed, 1 cell / view distance
      ip.fieldOfView = cam.fovY * 180 / .pi; ip.aspectRatio = Float(W) / Float(H)
      ip.isDepthReversed = true
      ip.shouldResetHistory = false
      ip.encode(commandBuffer: cr)
      e.encodePresent(cr, from: between!, pass: pass)
      pending = out
    } else {
      e.encodePresent(cr, from: out, pass: pass)
    }
    cr.present(drawable)
    cr.addCompletedHandler { [weak self] cb in
      guard let self else { return }
      self.lock.lock(); self.drawMs.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3); self.lock.unlock()
    }
    cr.commit()
    oi = 1 - oi; prev = u; valid = true

    if now - lastTitle > 0.5 {
      lastTitle = now
      lock.lock(); let sm = median(simMs), d = median(drawMs), a = Int(median(act.map(Double.init)))
      simMs.removeAll(); drawMs.removeAll(); act.removeAll(); lock.unlock()
      window?.title = String(format: "BigSand · %.2f billion grains · physics %.2f ms · draw %.2f ms · %d tiles awake · %@%@",
                             Double(e.grains) / 1e9, sm, d, a, interpolate ? "120 Hz (interpolated)" : "60 Hz", running ? "" : " · paused")
    }
  }
}

let args = CommandLine.arguments
if let i = args.firstIndex(of: "--shots") {
  let n = i + 1 < args.count ? Int(args[i + 1]) ?? 400 : 400
  let out = FileManager.default.currentDirectoryPath + "/shots"
  try? FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
  try shots(ticks: n, outDir: out)
  exit(0)
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let engine = try Engine()
let win = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1512, height: 982), styleMask: [.titled, .closable, .resizable, .miniaturizable],
                   backing: .buffered, defer: false)
let view = View(frame: win.contentView!.bounds, device: engine.device)
view.colorPixelFormat = .bgra8Unorm_srgb
view.preferredFramesPerSecond = 120
let renderer = Renderer(engine)
renderer.window = win
view.delegate = renderer
view.drag = { dx, dy in renderer.cam.yaw -= Float(dx) * 0.004; renderer.cam.pitch = max(-1.5, min(1.5, renderer.cam.pitch - Float(dy) * 0.004)) }
win.contentView = view
win.collectionBehavior = [.fullScreenPrimary]      // full screen (Ctrl-Cmd-F): where macOS Game Mode turns on
win.center(); win.makeKeyAndOrderFront(nil); win.makeFirstResponder(view)
app.activate(ignoringOtherApps: true)
app.run()
