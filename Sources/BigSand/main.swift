import AppKit
import MetalKit
import MetalFX

// BigSand: a general voxel world (an island, every cell one byte of material) you fly over, dig and build in.
//   /Applications/BigSand.app (scripts/make-launcher.sh) rebuilds what changed and runs it.
//   click to grab the mouse (Esc lets go), WASD/QE fly,
//     Shift fast, scroll for fly speed, Space pause
//       left mouse digs, right mouse places, 1-8 or Tab / Shift-Tab pick the material, [ ] the tool's radius
//   --world S: an island world S cells across (multiple of 64; default 16384), 1024 tall


func median(_ a: [Double]) -> Double { a.isEmpty ? 0 : a.sorted()[a.count / 2] }

// MARK: - window

extension World {
  func startCam() -> FlyCam { FlyCam(island: self) }
  func encodeFrame(_ cb: MTLCommandBuffer, dig: Bool, place: Bool) -> Bool { frame(cb, dig: dig, place: place) }
  func key(_ code: UInt16, shift: Bool) -> Bool {
    // 1-8: sand, water, stone, rock, plant, snow, basalt, wall; Tab / Shift-Tab: the next / previous material
    let mats: [UInt16: UInt32] = [18: World.SAND, 19: World.WATER, 20: World.STONE, 21: World.ROCK, 23: World.PLANT,
                                  22: World.SNOW, 26: World.BASALT, 28: World.WALL]
    if let m = mats[code] { material = m; return true }
    if code == 48 {                                                    // Tab: every material but air, in turn
      let n = UInt32(World.names.count) - 1
      material = (material - 1 + (shift ? n - 1 : 1)) % n + 1; return true
    }
    if code == 33 { radius = max(2, radius * 4 / 5); return true }     // [
    if code == 30 { radius = min(64, radius * 5 / 4 + 1); return true } // ]
    return false
  }
  func title() -> String { "\(World.names[Int(material)]) r\(radius)" }
}

final class View: MTKView {
  var keys = Set<UInt16>(), shift = false
  var look: ((CGFloat, CGFloat) -> Void)?
  var scroll: ((CGFloat) -> Void)?
  var left = false, right = false, captured = false
  override var acceptsFirstResponder: Bool { true }
  override func keyDown(with ev: NSEvent) {
    if ev.modifierFlags.contains(.command) { return }   // Cmd shortcuts belong to the menu, not to flying
    if ev.keyCode == 53 { release() } else { keys.insert(ev.keyCode) }
  }
  override func keyUp(with ev: NSEvent) { keys.remove(ev.keyCode) }
  override func flagsChanged(with ev: NSEvent) { shift = ev.modifierFlags.contains(.shift) }
  override func mouseDown(with ev: NSEvent) { if captured { left = true } else { capture() } }
  override func mouseUp(with ev: NSEvent) { left = false }
  override func rightMouseDown(with ev: NSEvent) { if captured { right = true } }
  override func rightMouseUp(with ev: NSEvent) { right = false }
  override func mouseMoved(with ev: NSEvent) { if captured { look?(ev.deltaX, ev.deltaY) } }
  override func mouseDragged(with ev: NSEvent) { look?(ev.deltaX, ev.deltaY) }
  override func rightMouseDragged(with ev: NSEvent) { if captured { look?(ev.deltaX, ev.deltaY) } }
  override func scrollWheel(with ev: NSEvent) { scroll?(ev.scrollingDeltaY) }
  /// Mouse-look: the cursor hides and stays put; moves turn the camera.
  /// The hidden cursor parks at the window's centre, so clicks land here even when it started over another app.
  func capture() {
    if captured { return }
    if let w = window, let top = NSScreen.screens.first?.frame.maxY {
      CGWarpMouseCursorPosition(CGPoint(x: w.frame.midX, y: top - w.frame.midY))
    }
    CGAssociateMouseAndMouseCursorPosition(0); NSCursor.hide(); captured = true
  }
  func release() { if captured { CGAssociateMouseAndMouseCursorPosition(1); NSCursor.unhide() }; captured = false; left = false; right = false }
}

/// Each refresh: the scene's frame work (physics or the tool, when there is any), the rays traced at a third of the
/// window's resolution (MetalFX's largest supported upscale, if smaller), and MetalFX's temporal scaler turning them
/// into an antialiased frame at full resolution. Every displayed frame is a real one, at up to 120 Hz.
final class Renderer: NSObject, MTKViewDelegate {
  let e: World; var cam = FlyCam(); var tex: MTLTexture?
  var depth: MTLTexture?, motion: MTLTexture?, scaler: MTLFXTemporalScaler?, output: MTLTexture?
  var prev = Cam(), valid = false, frame = 0
  var running = true
  var last = CACurrentMediaTime(), lastTitle = CACurrentMediaTime()
  var simMs: [Double] = [], drawMs: [Double] = [], cpuMs: [Double] = [], frameMs: [Double] = []
  let lock = NSLock()
  weak var window: NSWindow?
  var reloadPending = false
  init(_ e: World) { self.e = e; cam = e.startCam() }

  /// Cmd-R: start the scene over at the next frame.
  @objc func reload(_ sender: Any?) { reloadPending = true }

  func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) { scaler = nil }

  private func make(_ f: MTLPixelFormat, _ w: Int, _ h: Int, _ use: MTLTextureUsage) -> MTLTexture {
    let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: w, height: h, mipmapped: false)
    td.usage = use; td.storageMode = .private
    return e.device.makeTexture(descriptor: td)!
  }
  /// The scaler and its textures for a window of W x H pixels.
  private func build(_ W: Int, _ H: Int) {
    // MetalFX's scale is how much it enlarges (1 to supportedInputContentMaxScale, 3 on an M5): trace at 1/scale.
    let scale = 1.0 / min(3.0, Double(MTLFXTemporalScalerDescriptor.supportedInputContentMaxScale(device: e.device)))
    let w = max(Int(Double(W) * scale), 1), h = max(Int(Double(H) * scale), 1)
    let d = MTLFXTemporalScalerDescriptor()
    d.inputWidth = w; d.inputHeight = h; d.outputWidth = W; d.outputHeight = H
    d.colorTextureFormat = .rgba16Float; d.depthTextureFormat = .r32Float
    d.motionTextureFormat = .rg16Float; d.outputTextureFormat = .rgba16Float
    guard let s = d.makeTemporalScaler(device: e.device) else { fatalError("MetalFX refused this temporal scaler setup") }
    tex = make(.rgba16Float, w, h, s.colorTextureUsage.union(.shaderWrite))
    depth = make(.r32Float, w, h, s.depthTextureUsage.union(.shaderWrite))
    motion = make(.rg16Float, w, h, s.motionTextureUsage.union(.shaderWrite))
    output = make(.rgba16Float, W, H, s.outputTextureUsage.union(.shaderRead))
    scaler = s; valid = false
  }

  func draw(in view: MTKView) {
    guard let v = view as? View, let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable else { return }
    let now = CACurrentMediaTime(), dt = Float(min(now - last, 0.1))
    frameMs.append((now - last) * 1e3); last = now
    // fly
    let speed: Float = cam.speed * (v.shift ? 6 : 1) * dt
    let f = cam.forward, r = simd_normalize(simd_cross(f, SIMD3<Float>(0, 1, 0)))
    for k in v.keys { switch k {
      case 13: cam.pos += f * speed; case 1: cam.pos -= f * speed
      case 2: cam.pos += r * speed; case 0: cam.pos -= r * speed
      case 14: cam.pos.y += speed; case 12: cam.pos.y -= speed
      default: break } }
    if reloadPending { reloadPending = false; e.reset(); valid = false }
    for k in v.keys where e.key(k, shift: v.shift) { v.keys.remove(k) }   // the world's keys
    if v.keys.remove(49) != nil { running.toggle() }          // Space

    let W = max(Int(view.drawableSize.width), 2), H = max(Int(view.drawableSize.height), 2)
    if scaler == nil || output!.width != W || output!.height != H { build(W, H) }
    let s = scaler!, w = tex!.width, h = tex!.height
    if running {
      let cs = e.makeCommandBuffer()
      if e.encodeFrame(cs, dig: v.left, place: v.right) {
        cs.addCompletedHandler { [weak self] c in
          guard let self else { return }
          self.lock.lock(); self.simMs.append((c.gpuEndTime - c.gpuStartTime) * 1e3); self.lock.unlock()
        }
        cs.commit()
      }
    }
    var u = cam.uniforms(aspect: Float(w) / Float(h))
    // a new sub-pixel sample point every frame: the Halton (2, 3) sequence, 16 points
    func halton(_ i: Int, _ b: Int) -> Float { var f: Float = 1, r: Float = 0, n = i; while n > 0 { f /= Float(b); r += f * Float(n % b); n /= b }; return r }
    frame += 1
    u.jitter = SIMD2(halton(frame % 16 + 1, 2) - 0.5, halton(frame % 16 + 1, 3) - 0.5)
    if !valid { prev = u }
    let cr = e.makeCommandBuffer()
    e.encodeRender(cr, to: tex!, depth: depth!, motion: motion!, cam: &u, prev: &prev)
    s.colorTexture = tex; s.depthTexture = depth; s.motionTexture = motion; s.outputTexture = output
    s.inputContentWidth = w; s.inputContentHeight = h
    s.jitterOffsetX = u.jitter.x; s.jitterOffsetY = u.jitter.y
    s.motionVectorScaleX = 1; s.motionVectorScaleY = 1
    s.isDepthReversed = true
    s.reset = !valid
    s.encode(commandBuffer: cr)
    e.encodePresent(cr, from: output!, pass: pass)
    cr.present(drawable)
    cr.addCompletedHandler { [weak self] cb in
      guard let self else { return }
      self.lock.lock(); self.drawMs.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3); self.lock.unlock()
    }
    cr.commit()
    prev = u; valid = true
    cpuMs.append((CACurrentMediaTime() - now) * 1e3)

    if now - lastTitle > 0.5 {
      lastTitle = now
      lock.lock(); let sm = median(simMs), d = median(drawMs)
      simMs.removeAll(); drawMs.removeAll(); lock.unlock()
      let cpu = median(cpuMs), fr = median(frameMs), worst = frameMs.max() ?? 0
      cpuMs.removeAll(); frameMs.removeAll()
      // ms: GPU physics, rays; CPU; frame interval (worst)
      window?.title = String(format: "phys %.1f · rays %.1f · cpu %.1f · frame %.1f (%.0f) ms · ", sm, d, cpu, fr, worst)
        + e.title() + (running ? "" : " · paused")
    }
  }
}

let args = CommandLine.arguments
func intArg(_ name: String) -> Int? { args.firstIndex(of: name).flatMap { $0 + 1 < args.count ? Int(args[$0 + 1]) : nil } }
let worldArg = intArg("--world") ?? 16384

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let engine = try World(size: worldArg)
let win = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1512, height: 982), styleMask: [.titled, .closable, .resizable, .miniaturizable],
                   backing: .buffered, defer: false)
let view = View(frame: win.contentView!.bounds, device: engine.device)
view.colorPixelFormat = .bgra8Unorm_srgb
view.preferredFramesPerSecond = 120
let renderer = Renderer(engine)
renderer.window = win
view.delegate = renderer
view.look = { dx, dy in renderer.cam.yaw -= Float(dx) * 0.003; renderer.cam.pitch = max(-1.5, min(1.5, renderer.cam.pitch - Float(dy) * 0.003)) }
view.scroll = { dy in renderer.cam.speed = max(4, min(20000, renderer.cam.speed * Float(pow(1.02, Double(dy))))) }
win.acceptsMouseMovedEvents = true
// The menu: Cmd-R starts over, Cmd-Q quits.
let mainMenu = NSMenu(), appItem = NSMenuItem(), appMenu = NSMenu()
let reloadItem = NSMenuItem(title: "Reload", action: #selector(Renderer.reload(_:)), keyEquivalent: "r")
reloadItem.target = renderer
appMenu.addItem(reloadItem)
appMenu.addItem(.separator())
appMenu.addItem(NSMenuItem(title: "Quit BigSand", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
appItem.submenu = appMenu
mainMenu.addItem(appItem)
app.mainMenu = mainMenu
win.contentView = view
win.collectionBehavior = [.fullScreenPrimary]      // full screen (Ctrl-Cmd-F): where macOS Game Mode turns on
win.center(); win.makeKeyAndOrderFront(nil); win.makeFirstResponder(view)
// Grab the mouse as soon as the app is actually frontmost (activation lands after run() starts), and let go
// whenever it isn't, so Cmd-Tab away never leaves the cursor hidden and frozen.
var grabbedAtLaunch = false
NotificationCenter.default.addObserver(forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main) { _ in
  if !grabbedAtLaunch { grabbedAtLaunch = true; view.capture() }
}
NotificationCenter.default.addObserver(forName: NSApplication.didResignActiveNotification, object: nil, queue: .main) { _ in
  view.release()
}
app.activate(ignoringOtherApps: true)
app.run()
