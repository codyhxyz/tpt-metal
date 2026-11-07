import AppKit
import MetalKit
import ImageIO
import UniformTypeIdentifiers

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
  var cam = FlyCam().uniforms(aspect: 1512.0 / 982.0, shadows: true)
  let at: Set<Int> = [60, 90, 110, 140, 200, 300, ticks - 1]
  for i in 0..<ticks {
    let cb = e.queue.makeCommandBuffer()!
    e.encodeTick(cb)
    if at.contains(i) { e.encodeRender(cb, to: tex, depth: depth, cam: &cam) }
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

final class Renderer: NSObject, MTKViewDelegate {
  let e: Engine; var cam = FlyCam(); var tex: MTLTexture?
  var depth: MTLTexture?, history: [MTLTexture] = [], hcur = 0, prev = Cam(), valid = false, frame = 0
  var running = true, shadows = true
  var last = CACurrentMediaTime(), lastTitle = CACurrentMediaTime()
  var simMs: [Double] = [], drawMs: [Double] = [], act: [Int] = []
  let lock = NSLock()
  weak var window: NSWindow?
  init(_ e: Engine) { self.e = e }

  func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { tex = nil }

  func draw(in view: MTKView) {
    guard let v = view as? View, let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable else { return }
    let now = CACurrentMediaTime(), dt = Float(min(now - last, 0.1)); last = now
    // fly
    let speed: Float = (v.shift ? 2400 : 600) * dt
    let f = cam.forward, r = simd_normalize(simd_cross(f, SIMD3<Float>(0, 1, 0)))
    for k in v.keys { switch k {
      case 13: cam.pos += f * speed; case 1: cam.pos -= f * speed
      case 2: cam.pos += r * speed; case 0: cam.pos -= r * speed
      case 14: cam.pos.y += speed; case 12: cam.pos.y -= speed
      default: break } }
    if v.keys.remove(15) != nil { e.reset() }                 // R
    if v.keys.remove(49) != nil { running.toggle() }          // Space
    if v.keys.remove(3) != nil { shadows.toggle() }           // F

    let w = Int(view.drawableSize.width) / 2, h = Int(view.drawableSize.height) / 2
    if tex == nil || tex!.width != w || tex!.height != h {
      let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: max(w, 1), height: max(h, 1), mipmapped: false)
      td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .private
      tex = e.device.makeTexture(descriptor: td)
      history = [e.device.makeTexture(descriptor: td)!, e.device.makeTexture(descriptor: td)!]
      td.pixelFormat = .r32Float
      depth = e.device.makeTexture(descriptor: td)
      valid = false
    }
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
    var u = cam.uniforms(aspect: Float(w) / Float(max(h, 1)), shadows: shadows)
    // a new sub-pixel sample point every frame: the Halton (2, 3) sequence, 16 points
    func halton(_ i: Int, _ b: Int) -> Float { var f: Float = 1, r: Float = 0, n = i; while n > 0 { f /= Float(b); r += f * Float(n % b); n /= b }; return r }
    frame += 1
    u.jitter = SIMD2(halton(frame % 16 + 1, 2) - 0.5, halton(frame % 16 + 1, 3) - 0.5)
    let cr = e.queue.makeCommandBuffer()!
    e.encodeRender(cr, to: tex!, depth: depth!, cam: &u)
    e.encodeResolve(cr, raw: tex!, depth: depth!, history: history[hcur], into: history[1 - hcur], cam: &u, prev: &prev, valid: valid)
    hcur = 1 - hcur; prev = u; valid = true
    e.encodePresent(cr, from: history[hcur], pass: pass)
    cr.present(drawable)
    cr.addCompletedHandler { [weak self] cb in
      guard let self else { return }
      self.lock.lock(); self.drawMs.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3); self.lock.unlock()
    }
    cr.commit()

    if now - lastTitle > 0.5 {
      lastTitle = now
      lock.lock(); let s = median(simMs), d = median(drawMs), a = Int(median(act.map(Double.init)))
      simMs.removeAll(); drawMs.removeAll(); act.removeAll(); lock.unlock()
      window?.title = String(format: "BigSand · %.2f billion grains · physics %.2f ms · draw %.2f ms · %d tiles awake%@",
                             Double(e.grains) / 1e9, s, d, a, running ? "" : " · paused")
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
view.preferredFramesPerSecond = 60
let renderer = Renderer(engine)
renderer.window = win
view.delegate = renderer
view.drag = { dx, dy in renderer.cam.yaw -= Float(dx) * 0.004; renderer.cam.pitch = max(-1.5, min(1.5, renderer.cam.pitch - Float(dy) * 0.004)) }
win.contentView = view
win.center(); win.makeKeyAndOrderFront(nil); win.makeFirstResponder(view)
app.activate(ignoringOtherApps: true)
app.run()
