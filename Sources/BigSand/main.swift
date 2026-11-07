import AppKit
import MetalKit
import ImageIO
import UniformTypeIdentifiers

// BigSand: a cube of sand falls and piles up.
//   swift run -c release BigSand                 window: WASD/QE fly, drag to look, Shift fast, R drop again, Space pause, F shadows
//   swift run -c release BigSand --bench [N]     headless: N ticks, timings vs the model, grain census, screenshots

let GBps = 120e9   // measured copy bandwidth (FACTS.md)

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

func bench(ticks: Int, outDir: String) throws {
  let t0 = Date()
  let e = try Engine()
  print(String(format: "world %d x %d x %d = %.1f G cells (%.1f GiB of cells); cube %d^3 = %.2f G grains; %d flow rounds/tick; setup %.2f s",
               e.WX, e.WORDS * 64, e.WZ, Double(e.cells) / 1e9, Double(e.cells / 8) / 1073741824.0,
               e.cube, Double(e.grains) / 1e9, e.rounds, Date().timeIntervalSince(t0)))
  let before = e.census()
  let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1512, height: 982, mipmapped: false)
  td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .private
  let tex = e.device.makeTexture(descriptor: td)!
  var cam = FlyCam().uniforms(aspect: 1512.0 / 982.0, shadows: true)
  struct Row { var tick: Int; var fall: Double; var topple: Double; var draw: Double; var active: Int }
  var rows: [Row] = []
  let shots: Set<Int> = [100, 175, 220, 300, 450, ticks - 1]
  for i in 0..<ticks {
    let ca = e.queue.makeCommandBuffer()!, cb = e.queue.makeCommandBuffer()!
    e.encodeTick(ca, cb); ca.commit(); cb.commit()
    let cr = e.queue.makeCommandBuffer()!; e.encodeRender(cr, to: tex, cam: &cam); cr.commit()
    cr.waitUntilCompleted()
    rows.append(Row(tick: i, fall: (ca.gpuEndTime - ca.gpuStartTime) * 1e3, topple: (cb.gpuEndTime - cb.gpuStartTime) * 1e3,
                    draw: (cr.gpuEndTime - cr.gpuStartTime) * 1e3, active: e.activeTiles))
    if shots.contains(i) { savePNG(tex, e, "\(outDir)/tick-\(i).png") }
    if Date().timeIntervalSince(t0) > 16 { print("(stopped at tick \(i): time cap)"); break }
  }
  let after = e.census()
  print("ticks      awake tiles | fall ms (model<=) | flow ms (model, GB/s)   | draw ms")
  for start in stride(from: 0, to: rows.count, by: 30) {
    let r = Array(rows[start..<min(start + 30, rows.count)])
    let act = Int(median(r.map { Double($0.active) }))
    let fa = median(r.map { $0.fall }), tp = median(r.map { $0.topple })
    print(String(format: "%4d-%-4d %9d  | %6.2f (%6.2f)    | %6.2f (%5.2f, %5.1f)    | %5.2f", r.first!.tick, r.last!.tick, act,
                 fa, e.fallBytes(act) / GBps * 1e3, tp, e.toppleBytes(act) / GBps * 1e3, e.toppleBytes(act) / (tp / 1e3) / 1e9,
                 median(r.map { $0.draw })))
  }
  print("grains: \(before) before, \(after) after -> \(before == after ? "conserved" : "LOST \(before - after)")")
  print(String(format: "total %.1f s; screenshots in %@", Date().timeIntervalSince(t0), outDir))
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
    }
    if running {
      let ca = e.queue.makeCommandBuffer()!, cb = e.queue.makeCommandBuffer()!
      e.encodeTick(ca, cb)
      cb.addCompletedHandler { [weak self] c in
        guard let self else { return }
        let a = self.e.activeTiles
        let ms = (ca.gpuEndTime - ca.gpuStartTime + c.gpuEndTime - c.gpuStartTime) * 1e3
        self.lock.lock(); self.simMs.append(ms); self.act.append(a); self.lock.unlock()
      }
      ca.commit(); cb.commit()
    }
    var u = cam.uniforms(aspect: Float(w) / Float(max(h, 1)), shadows: shadows)
    let cr = e.queue.makeCommandBuffer()!
    e.encodeRender(cr, to: tex!, cam: &u)
    e.encodePresent(cr, from: tex!, pass: pass)
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
      let model = (e.fallBytes(a) + e.toppleBytes(a)) / GBps * 1e3
      window?.title = String(format: "BigSand · %.2f G grains · physics %.2f ms (model <= %.2f) · draw %.2f ms · %d tiles awake%@",
                             Double(e.grains) / 1e9, s, model, d, a, running ? "" : " · paused")
    }
  }
}

let args = CommandLine.arguments
if let i = args.firstIndex(of: "--bench") {
  let n = i + 1 < args.count ? Int(args[i + 1]) ?? 400 : 400
  let out = FileManager.default.currentDirectoryPath + "/bench-out"
  try? FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
  try bench(ticks: n, outDir: out)
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
