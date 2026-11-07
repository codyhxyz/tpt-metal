// Drawing the world: the rays, their shading, and the present pass. Included by World.metal (one translation unit).
// Owner: the render piece. Contract with the other pieces: Shaders/Surface.h.
// ---------------------------------------------------------------------------------------------------------------
// Rays. One loop walks all three levels: at each step it looks up the box containing the current cell (a whole
// chunk, a brick or one cell, whichever is uniform first) and, if that box is empty, jumps to where the ray leaves
// it. Water is a medium the ray passes through, not a wall: it remembers where it entered and how far it went.

struct WCam {
  float4 pos, fwd, right, up, sun;
  float tanHalf, aspect; uint shadows, stats;
  float2 jitter, pad;
};

struct Hit { float t; int3 cell; int axis; uint mat; float wet; float tWater; int3 wcell; int waxis; uint steps; };

// lodK: cells a pixel spans per unit of distance. Where a brick is smaller than a pixel, the ray stops walking its
// cells and takes it as one block: the top solid cell of the brick's column under the ray.
inline Hit trace(float3 o, float3 d, float tMax, float lodK, device const uint* grid, device const uint* tables, device const uchar* bricks) {
  Hit h; h.t = -1; h.mat = AIR; h.wet = 0; h.tWater = -1; h.steps = 0; h.axis = -1; h.waxis = -1;
  float3 inv = 1.0f / d, dims = float3(NW, NH, NW);
  float3 ta = (0.0f - o) * inv, tb = (dims - o) * inv, tn = min(ta, tb), tf = max(ta, tb);
  float t = max(max(tn.x, tn.y), max(tn.z, 0.0f)), tExit = min(min(tf.x, tf.y), min(tf.z, tMax));
  if (t >= tExit) return h;
  int axis = -1;
  if (t > 0) axis = tn.x >= tn.y && tn.x >= tn.z ? 0 : (tn.y >= tn.z ? 1 : 2);
  int3 c = clamp(int3(floor(o + d * t)), int3(0), int3(NW, NH, NW) - 1);
  if (axis >= 0) c[axis] = d[axis] > 0 ? 0 : int(dims[axis]) - 1;
  for (uint i = 0; i < 4096u; i++) {
    h.steps++;
    uint ce = grid[chunkIdx(c >> 6)], m; int s;
    if (ce & W_UNI) { m = ce & 255u; s = 64; }
    else {
      uint be = tables[ulong(ce) * 512ul + loc((c >> 3) & 7)];
      if (be & W_UNI) { m = be & 255u; s = 8; }
      else if (t * lodK >= 8.0f) {
        ulong b = ulong(be) * 512ul;
        int3 l = c & 7;
        m = AIR;
        for (int y = 7; y >= 0 && (m == AIR || isClear(m)); y--) {
          uint v = bricks[b + loc(int3(l.x, y, l.z))];
          if (v != AIR) m = v;
        }
        s = 8;
      }
      else { m = bricks[ulong(be) * 512ul + loc(c & 7)]; s = 1; }
    }
    if (!isClear(m)) { h.t = t; h.cell = c; h.axis = axis; h.mat = matOf(m); return h; }
    int3 lo = c & ~(s - 1);
    float3 te = (float3(lo) + select(float3(0), float3(s), d > 0) - o) * inv;
    int k = te.x <= te.y && te.x <= te.z ? 0 : (te.y <= te.z ? 1 : 2);
    float tn2 = te[k];
    if (m != AIR) {                                   // a clear liquid
      if (h.tWater < 0) { h.tWater = t; h.wcell = c; h.waxis = i == 0u && axis != 1 ? 1 : axis; }
      h.wet += min(tn2, tExit) - t;
    }
    if (tn2 >= tExit) return h;
    t = tn2; axis = k;
    int3 n = clamp(int3(floor(o + d * t)), lo, lo + s - 1);
    n[k] = d[k] > 0 ? lo[k] + s : lo[k] - 1;
    c = n;
  }
  return h;
}

inline float3 skyColor(float3 d, float3 L) {
  float3 sky = mix(float3(0.72, 0.80, 0.90), float3(0.26, 0.46, 0.82), pow(saturate(d.y), 0.6f));
  float sd = max(dot(d, L), 0.0f);
  return sky + float3(1.0, 0.85, 0.6) * (pow(sd, 900.0f) * 30.0f + pow(sd, 12.0f) * 0.25f);
}

kernel void w_render(texture2d<float, access::write> out [[texture(0)]], texture2d<float, access::write> depthOut [[texture(1)]],
                     texture2d<float, access::write> motionOut [[texture(2)]],
                     device const uint* grid [[buffer(0)]], device const uint* tables [[buffer(1)]], device const uchar* bricks [[buffer(2)]],
                     constant WCam& C [[buffer(3)]], constant WCam& P [[buffer(4)]], device atomic_int* ctr [[buffer(5)]],
                     uint2 g [[thread_position_in_grid]]) {
  uint W = out.get_width(), H = out.get_height();
  if (g.x >= W || g.y >= H) return;
  float2 uv = (float2(g) + 0.5f + C.jitter) / float2(W, H) * 2.0f - 1.0f; uv.y = -uv.y;
  float3 d = normalize(C.fwd.xyz + uv.x * C.tanHalf * C.aspect * C.right.xyz + uv.y * C.tanHalf * C.up.xyz);
  d = select(d, float3(1e-7f), abs(d) < 1e-7f);
  float3 o = C.pos.xyz, L = normalize(C.sun.xyz);
  float3 sunCol = float3(1.0, 0.94, 0.84) * 1.25f;
  // The endless sea outside the world: a ray that reaches the sea's surface outside the world stops there.
  float tSea = d.y < 0 && o.y > float(W_SEA) ? (float(W_SEA) - o.y) / d.y : 1e30f;
  float2 ps = o.xz + d.xz * tSea;
  bool seaOut = tSea < 1e29f && (ps.x < 0 || ps.y < 0 || ps.x > float(NW) || ps.y > float(NW));
  Hit h = trace(o, d, seaOut ? tSea : 1e30f, 2.0f * C.tanHalf / float(H), grid, tables, bricks);
  uint steps = h.steps;
  float3 col; float tt;
  bool center = g.x == W / 2 && g.y == H / 2;
  if (h.t >= 0) {
    float3 n = float3(0); int k = h.axis;
    if (k >= 0) n[k] = d[k] > 0 ? -1.0f : 1.0f; else n = -d;
    float footprint = h.t * 2.0f * C.tanHalf / float(H);
    float r = float(hash3(h.cell.x, h.cell.y, h.cell.z, 3u) & 1023u) / 1023.0f;
    float3 base = matColor(h.mat) * (1.0f + MATS[h.mat].var * (2.0f * r - 1.0f) / max(1.0f, footprint));
    // Minecraft's per-side shade (top 1, north/south 0.8, east/west 0.6, bottom 0.5) on the sky's light, plus the sun
    float side = n.y > 0.5f ? 1.0f : (n.y < -0.5f ? 0.5f : (abs(n.x) > 0.5f ? 0.6f : 0.8f));
    col = base * (W_AMBIENT * side + sunCol * max(dot(n, L), 0.0f));
    tt = h.t;
  } else {
    col = skyColor(d, L);
    tt = 1e30f;
    // the endless sea around the island
    if (seaOut && h.tWater < 0) {
      // over a sand floor 60 cells down, like the world's own sea far from the island
      h.tWater = tSea; h.waxis = 1; h.wet = 60.0f / max(-d.y, 0.05f); tt = tSea;
      col = matColor(SAND) * (W_AMBIENT + sunCol * max(L.y, 0.0f));
    }
  }
  if (h.tWater >= 0) {
    // light through water: absorbed per cell travelled (red first), then the surface reflects the sky (Fresnel)
    float3 sigma = float3(0.11, 0.035, 0.018);
    float3 tr = exp(-sigma * h.wet);
    float3 deep = float3(0.02, 0.10, 0.16);
    col = col * tr + deep * (1.0f - tr);
    float3 wn = float3(0, 1, 0);
    bool edge = h.wcell.x == 0 || h.wcell.z == 0 || h.wcell.x == NW - 1 || h.wcell.z == NW - 1;   // the sea at the world's rim
    if (h.waxis >= 0 && h.waxis != 1 && !edge) { wn = float3(0); wn[h.waxis] = d[h.waxis] > 0 ? -1.0f : 1.0f; }
    float cosv = saturate(-dot(d, wn));
    if (o.y >= float(W_SEA) || h.waxis != 1) {
      float F = 0.02f + 0.98f * pow(1.0f - cosv, 5.0f);
      col = mix(col, skyColor(reflect(d, wn), L), F);
    }
    if (h.t < 0) tt = h.tWater;
  }
  if (tt < 1e29f) {
    // haze thins with height (scale height 400 cells above the sea): the optical depth along the ray, in closed form
    float Hs = 400.0f, k = 1.0f / 9000.0f, y0 = o.y - float(W_SEA), dy = d.y * tt;
    float depth = abs(dy) < 1e-3f * Hs ? tt * exp(-y0 / Hs) : tt * Hs / dy * (exp(-y0 / Hs) - exp(-(y0 + dy) / Hs));
    col = mix(col, skyColor(float3(d.x, 0.02f, d.z), L), 1.0f - exp(-k * max(depth, 0.0f)));
  }
  if (C.stats != 0u) atomic_fetch_add_explicit(&ctr[C_STEPS], int(steps), memory_order_relaxed);
  if (center) {
    // what the crosshair is on: the first thing that isn't air (water counts), and the face the ray came through
    bool wetFirst = h.tWater >= 0 && !(seaOut && h.t < 0 && h.tWater == tSea) && (h.t < 0 || h.tWater < h.t);
    bool got = wetFirst || h.t >= 0;
    int3 pc = wetFirst ? h.wcell : h.cell;
    int ax = wetFirst ? h.waxis : h.axis;
    int3 nn = int3(0);
    if (got && ax >= 0) nn[ax] = d[ax] > 0 ? -1 : 1;
    atomic_store_explicit(&ctr[C_PICK], pc.x, memory_order_relaxed);
    atomic_store_explicit(&ctr[C_PICK + 1], pc.y, memory_order_relaxed);
    atomic_store_explicit(&ctr[C_PICK + 2], pc.z, memory_order_relaxed);
    atomic_store_explicit(&ctr[C_PICK + 3], got ? 1 + (nn.x + 1) + 3 * (nn.y + 1) + 9 * (nn.z + 1) : 0, memory_order_relaxed);
  }
  out.write(float4(col, 1.0f), g);
  float dist = tt < 1e29f ? tt : 1e9f;
  depthOut.write(float4(tt < 1e29f ? W_NEAR / max(dist * dot(d, C.fwd.xyz), W_NEAR) : 0.0f), g);
  float3 v = o + d * dist - P.pos.xyz;
  float z = dot(v, P.fwd.xyz);
  float2 motion = float2(0.0f);
  if (z > 0.0f) {
    float2 q = float2(dot(v, P.right.xyz) / (z * P.tanHalf * P.aspect), dot(v, P.up.xyz) / (z * P.tanHalf));
    motion = float2(q.x * 0.5f + 0.5f, 0.5f - q.y * 0.5f) * float2(W, H) - (float2(g) + 0.5f + C.jitter);
  }
  motionOut.write(float4(motion, 0.0f, 0.0f), g);
}

// --- present: MetalFX's output to the window, with a crosshair
struct WVOut { float4 pos [[position]]; float2 uv; };
vertex WVOut w_presentV(uint vid [[vertex_id]]) {
  float2 p = float2((vid << 1) & 2, vid & 2);
  WVOut o; o.pos = float4(p * 2.0f - 1.0f, 0, 1); o.uv = float2(p.x, 1.0f - p.y); return o;
}
fragment float4 w_presentF(WVOut in [[stage_in]], texture2d<float> tex [[texture(0)]]) {
  constexpr sampler s(filter::linear, address::clamp_to_edge);
  float3 c = tex.sample(s, in.uv).rgb;
  float2 px = abs(in.pos.xy - float2(tex.get_width(), tex.get_height()) * 0.5f);
  if ((px.x < 1.5f && px.y < 12.0f) || (px.y < 1.5f && px.x < 12.0f)) c = float3(1.0f) - saturate(c);
  return float4(c, 1.0f);
}
