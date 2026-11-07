import AppKit
import MetalKit
import ImageIO
import UniformTypeIdentifiers
import MetalFX

// BigSand: a general voxel world (an island, every cell one byte of material) you fly over, dig and build in; or,
// with --cube, a cube of sand that falls and piles up.
//   swift run -c release BigSand                 window: click to grab the mouse (Esc lets go), WASD/QE fly,
//                                                 Shift fast, scroll for fly speed, F shadows
//       island: left mouse digs, right mouse places, 1-8 pick the material, [ ] the tool's radius
//       cube:   R drop again, Space pause
//   swift run -c release BigSand --cube S --shots [N]   headless cube benchmark
//   --world S: an island world S cells across (multiple of 64; default 16384), 1024 tall
//   --cube S:  the falling cube instead, S cells on a side (multiple of 64; 1024 is 1.07 G grains, 4096 is 68.7 G)


// MARK: - headless benchmark

func savePNG(_ tex: MTLTexture, _ device: MTLDevice, _ queue: MTLCommandQueue, _ path: String) {
  let w = tex.width, h = tex.height
  let buf = device.makeBuffer(length: w * h * 8, options: .storageModeShared)!
  let cb = queue.makeCommandBuffer()!, b = cb.makeBlitCommandEncoder()!
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

/// Headless run: `ticks` ticks, a few saved frames, GPU time per pass, and a check that no grain was created or lost.
/// Each pass runs in its own command buffer so the GPU's start and end times of that buffer time it.
func shots(ticks: Int, outDir: String, cube: Int) throws {
  let e = try Engine(cube: cube)
  print(String(format: "cube %d (%.2f G grains) in a %d x %d x %d world, %.2f GB of world state, %d flow steps per tick",
               cube, Double(e.grains) / 1e9, e.WX, e.HY, e.WZ, Double(e.bytes) / 1e9, e.substeps))
  let before = e.census()
  let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1512, height: 982, mipmapped: false)
  td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .private
  let tex = e.device.makeTexture(descriptor: td)!
  td.pixelFormat = .r32Float
  let depth = e.device.makeTexture(descriptor: td)!
  td.pixelFormat = .rg16Float
  let motion = e.device.makeTexture(descriptor: td)!
  var cam = FlyCam(world: e).uniforms(aspect: 1512.0 / 982.0, shadows: true)
  let at: Set<Int> = [60, 90, 110, 140, 200, 300, ticks - 1]
  let passes = ["plan", "fall", "flow", "surface", "sun", "bounds"]
  var ms: [String: [Double]] = [:], draw: [Double] = [], tickMs: [Double] = []
  print("tick    plan   fall   flow  surf.    sun bounds    | whole tick  (GPU ms)  tiles awake")
  for i in 0..<ticks {
    // Even ticks: each pass in its own command buffer, to time the passes. Odd ticks: the whole tick in one command
    // buffer, as the window runs it, to time the tick (separate buffers add gaps and let the GPU's clock drop between them).
    var cbs: [(String, MTLCommandBuffer)] = []
    if i % 2 == 0 {
      e.encodeTick { name in
        if let c = cbs.last, c.0 == name { return c.1 }
        let c = e.queue.makeCommandBuffer()!; cbs.append((name, c)); return c
      }
    } else {
      let c = e.queue.makeCommandBuffer()!; e.encodeTick(c); cbs.append(("tick", c))
    }
    for (_, c) in cbs { c.commit() }
    for (_, c) in cbs { c.waitUntilCompleted() }
    var row: [Double] = []
    for (name, c) in cbs { let t = (c.gpuEndTime - c.gpuStartTime) * 1e3; ms[name, default: []].append(t); row.append(t) }
    let total = cbs.map { $0.1.gpuEndTime }.max()! - cbs.map { $0.1.gpuStartTime }.min()!
    if i % 2 == 1 { tickMs.append(total * 1e3) }
    if at.contains(i) {
      let cr = e.queue.makeCommandBuffer()!
      var same = cam; e.encodeRender(cr, to: tex, depth: depth, motion: motion, cam: &cam, prev: &same)
      cr.commit(); cr.waitUntilCompleted()
      draw.append((cr.gpuEndTime - cr.gpuStartTime) * 1e3)
      savePNG(tex, e.device, e.queue, "\(outDir)/tick-\(i).png")
    }
    if i % 2 == 0 && (at.contains(i) || i % 50 == 0) {
      print(String(format: "%4d ", i) + row.map { String(format: " %6.2f", $0) }.joined() + String(format: "    |            %8d", e.activeTiles))
    } else if i % 2 == 1 && (at.contains(i) || i % 50 == 1) {
      print(String(format: "%4d  (one command buffer)                         | %6.2f      %8d", i, total * 1e3, e.activeTiles))
    }
  }
  for p in passes { let a = ms[p] ?? []; print(String(format: "%-8@ median %6.2f ms, max %6.2f ms", p as NSString, median(a), a.max() ?? 0)) }
  print(String(format: "tick     median %6.2f ms, max %6.2f ms (whole ticks, one command buffer each)", median(tickMs), tickMs.max() ?? 0))
  print(String(format: "draw     median %6.2f ms at 1512x982", median(draw)))
  let after = e.census()
  print("grains: \(before) before, \(after) after -> \(before == after ? "conserved" : "CHANGED BY \(after - before)")")
}

// MARK: - window

/// What the window shows: the island world or the falling cube.
protocol Scene: AnyObject {
  var device: MTLDevice { get }
  func makeCommandBuffer() -> MTLCommandBuffer
  func startCam() -> FlyCam
  /// Once per rendered frame, before the rays: physics, or the tool. False if it encoded nothing (the buffer is then dropped).
  func encodeFrame(_ cb: MTLCommandBuffer, dig: Bool, place: Bool) -> Bool
  func encodeRender(_ cb: MTLCommandBuffer, to tex: MTLTexture, depth: MTLTexture, motion: MTLTexture, cam: inout Cam, prev: inout Cam)
  func encodePresent(_ cb: MTLCommandBuffer, from tex: MTLTexture, pass: MTLRenderPassDescriptor)
  func key(_ code: UInt16) -> Bool   // true if it handled the key
  func title(frameMs: Double, drawMs: Double) -> String
  /// Back to the start: the cube hangs in the sky again; the island is generated afresh.
  func reset()
}

extension Engine: Scene {
  func makeCommandBuffer() -> MTLCommandBuffer { queue.makeCommandBuffer()! }
  func startCam() -> FlyCam { FlyCam(world: self) }
  func encodeFrame(_ cb: MTLCommandBuffer, dig: Bool, place: Bool) -> Bool { encodeTick(cb); return true }
  func key(_ code: UInt16) -> Bool { if code == 15 { reset(); return true }; return false }   // R: drop it again
  func title(frameMs: Double, drawMs: Double) -> String {
    String(format: "BigSand · %.2f billion grains · physics %.2f ms · draw %.2f ms · %d tiles awake", Double(grains) / 1e9, frameMs, drawMs, activeTiles)
  }
}

extension World: Scene {
  func startCam() -> FlyCam { FlyCam(island: self) }
  func encodeFrame(_ cb: MTLCommandBuffer, dig: Bool, place: Bool) -> Bool { tool(cb, dig: dig, place: place) }
  func key(_ code: UInt16) -> Bool {
    // 1-8: sand, water, stone, rock, plant, snow, basalt, wall
    let mats: [UInt16: UInt32] = [18: World.SAND, 19: World.WATER, 20: World.STONE, 21: World.ROCK, 23: World.PLANT,
                                  22: World.SNOW, 26: World.BASALT, 28: World.WALL]
    if let m = mats[code] { material = m; return true }
    if code == 33 { radius = max(2, radius * 4 / 5); return true }     // [
    if code == 30 { radius = min(64, radius * 5 / 4 + 1); return true } // ]
    return false
  }
  func title(frameMs: Double, drawMs: Double) -> String {
    String(format: "Island %d x %d x %d · %.2f GB · %d bricks · draw %.2f ms · tool: %@, radius %d", NW, NH, NW, Double(bytes) / 1e9,
           bricksUsed, drawMs, World.names[Int(material)] as NSString, radius)
  }
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
  func capture() { CGAssociateMouseAndMouseCursorPosition(0); NSCursor.hide(); captured = true }
  func release() { if captured { CGAssociateMouseAndMouseCursorPosition(1); NSCursor.unhide() }; captured = false; left = false; right = false }
}

/// Each refresh: the scene's frame work (physics or the tool, when there is any), the rays traced at a third of the
/// window's resolution (MetalFX's largest supported upscale, if smaller), and MetalFX's temporal scaler turning them
/// into an antialiased frame at full resolution. Every displayed frame is a real one, at up to 120 Hz.
final class Renderer: NSObject, MTKViewDelegate {
  let e: Scene; var cam = FlyCam(); var tex: MTLTexture?
  var depth: MTLTexture?, motion: MTLTexture?, scaler: MTLFXTemporalScaler?, output: MTLTexture?
  var prev = Cam(), valid = false, frame = 0
  var running = true, shadows = true
  var last = CACurrentMediaTime(), lastTitle = CACurrentMediaTime()
  var simMs: [Double] = [], drawMs: [Double] = []
  let lock = NSLock()
  weak var window: NSWindow?
  var reloadPending = false
  init(_ e: Scene) { self.e = e; cam = e.startCam() }

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
    let now = CACurrentMediaTime(), dt = Float(min(now - last, 0.1)); last = now
    // fly
    let speed: Float = cam.speed * (v.shift ? 6 : 1) * dt
    let f = cam.forward, r = simd_normalize(simd_cross(f, SIMD3<Float>(0, 1, 0)))
    for k in v.keys { switch k {
      case 13: cam.pos += f * speed; case 1: cam.pos -= f * speed
      case 2: cam.pos += r * speed; case 0: cam.pos -= r * speed
      case 14: cam.pos.y += speed; case 12: cam.pos.y -= speed
      default: break } }
    if reloadPending { reloadPending = false; e.reset(); valid = false }
    for k in v.keys where e.key(k) { v.keys.remove(k); if k == 15 { valid = false } }   // the scene's keys (R: start MetalFX's history over too)
    if v.keys.remove(49) != nil { running.toggle() }          // Space
    if v.keys.remove(3) != nil { shadows.toggle() }           // F

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
    var u = cam.uniforms(aspect: Float(w) / Float(h), shadows: shadows)
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

    if now - lastTitle > 0.5 {
      lastTitle = now
      lock.lock(); let sm = median(simMs), d = median(drawMs)
      simMs.removeAll(); drawMs.removeAll(); lock.unlock()
      window?.title = e.title(frameMs: sm, drawMs: d) + " · rays \(w) x \(h) -> \(W) x \(H)" + (running ? "" : " · paused")
    }
  }
}

let args = CommandLine.arguments
func intArg(_ name: String) -> Int? { args.firstIndex(of: name).flatMap { $0 + 1 < args.count ? Int(args[$0 + 1]) : nil } }
let cubeMode = args.contains("--cube")
let cubeArg = intArg("--cube") ?? 1024
let worldArg = intArg("--world") ?? 16384
if let i = args.firstIndex(of: "--shots") {
  let n = i + 1 < args.count ? Int(args[i + 1]) ?? 400 : 400
  let out = FileManager.default.currentDirectoryPath + "/shots"
  try? FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
  try shots(ticks: n, outDir: out, cube: cubeArg)
  exit(0)
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let engine: Scene = cubeMode ? try Engine(cube: cubeArg) : try World(size: worldArg)
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
app.activate(ignoringOtherApps: true)
app.run()
