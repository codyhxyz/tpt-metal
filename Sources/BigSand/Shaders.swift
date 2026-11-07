// The GPU side: four kernels and a present pass. Compiled at launch from this source.
//
// Layout: the world is columns of 1-bit cells, 64 cells per 64-bit word, WORDS words tall.
// Columns are grouped in 8x8 tiles; inside a tile, word w of all 64 columns sits side by side
// (addr = (tile*WORDS + w)*64 + lane), so a tile's 64 threads read each word in one 512-byte burst.

func shaderSource(wx: Int, wz: Int, words: Int) -> String { """
#include <metal_stdlib>
using namespace metal;
#define WX \(wx)
#define WZ \(wz)
#define WORDS \(words)
#define HY (WORDS * 64)
#define TX (WX / 8)
#define TZ (WZ / 8)
#define SX (WX / 64)
#define SZ (WZ / 64)
#define WAKE 16u   // ticks a tile stays awake after the last change near it (4 full slide-direction cycles)

inline uint addr(int x, int z, int w) {
  uint tile = uint(z >> 3) * TX + uint(x >> 3);
  return ((tile * WORDS + uint(w)) << 6) | uint(((z & 7) << 3) | (x & 7));
}
inline bool inWorld(int x, int z) { return x >= 0 && z >= 0 && x < WX && z < WZ; }
inline uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
// 64 coin flips for the 64 cells of word w of column (x, z) on this tick
inline ulong coins(int x, int z, int w, uint tick) {
  uint h = pcg(uint(x) ^ pcg(uint(z) ^ pcg(uint(w) ^ pcg(tick))));
  return (ulong(pcg(h ^ 0x9e3779b9u)) << 32) | ulong(h);
}

struct Tick { uint tick; uint fall; uint sub; uint pad; };

// Per column: the grounded run (sand touching the floor, unbroken) is one number, its height h.
// cap = HY - (lowest sand above the run), 0 if none: the run may grow by free = HY - cap - h.
inline uint colIdx(int x, int z) { return uint(z) * WX + uint(x); }
inline uint tileOf(int x, int z) { return uint(z >> 3) * TX + uint(x >> 3); }

// Which tiles run this tick: any tile with a change in its 3x3 neighbourhood in the last WAKE ticks.
kernel void plan(device const uint* changedLast [[buffer(0)]], device uint* changedNow [[buffer(1)]],
                 device uint* countdown [[buffer(2)]], device uint* awake [[buffer(3)]],
                 device atomic_uint* args [[buffer(4)]], device uint* active [[buffer(5)]],
                 uint t [[thread_position_in_grid]]) {
  if (t >= TX * TZ) return;
  int tx = int(t % TX), tz = int(t / TX);
  bool any = false;
  for (int dz = -1; dz <= 1; dz++) for (int dx = -1; dx <= 1; dx++) {
    int x = tx + dx, z = tz + dz;
    if (x >= 0 && z >= 0 && x < TX && z < TZ && changedLast[z * TX + x] != 0) any = true;
  }
  uint c = any ? WAKE : (countdown[t] > 0 ? countdown[t] - 1 : 0);
  countdown[t] = c; awake[t] = c; changedNow[t] = 0;
  if (c > 0) active[atomic_fetch_add_explicit(args, 1, memory_order_relaxed)] = t;
}

// Fall: only columns with sand in the air. Column-local, so it updates in place.
// `fall` sub-steps; in each, every unsupported run of sand drops one cell together.
// seeds = run bottoms resting on support; s + seeds clears exactly those runs (the carry ripples up them).
// Also rewrites the tile's exact height bounds (lowest and highest sand) for the rays.
kernel void fallStep(device ulong* vox [[buffer(0)]], device uint* H [[buffer(1)]],
                     device const uint* active [[buffer(2)]], device uint* changed [[buffer(3)]],
                     device uint2* bounds [[buffer(4)]], constant Tick& T [[buffer(5)]],
                     uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  threadgroup ulong col[WORDS][64];
  threadgroup atomic_uint tgChanged, tgLo, tgHi;
  uint tile = active[tg];
  int x = int(tile % TX) * 8 + int(lane & 7), z = int(tile / TX) * 8 + int(lane >> 3);
  uint2 tb = bounds[tile];
  if (lane == 0) {
    atomic_store_explicit(&tgChanged, 0u, memory_order_relaxed);
    atomic_store_explicit(&tgLo, 0xffffffffu, memory_order_relaxed);
    atomic_store_explicit(&tgHi, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint hc = H[colIdx(x, z)];
  int lo = -1, hi = 0;
  if ((hc >> 16) != 0u) {                       // something above the grounded run
    // every word below w0 is empty in this whole tile; sand moves down < 64 cells per tick
    int w0 = max(0, int(tb.x >> 6) - 1), w1 = min(WORDS - 1, int(tb.y + 63u) >> 6);
    for (int w = w0; w <= w1; w++) col[w][lane] = vox[addr(x, z, w)];
    bool moved = false;
    for (uint s = 0; s < T.fall; s++) {
      ulong carry = w0 == 0 ? 1ul : 0ul, Sp = 0ul, Mp = 0ul; bool any = false;
      for (int w = w0; w <= w1; w++) {
        ulong sw = col[w][lane];
        ulong S = sw & ~(sw + (sw & carry));
        ulong M = sw & ~S;
        any = any || M != 0ul;
        if (w > w0) col[w - 1][lane] = Sp | (Mp >> 1) | (M << 63);
        Sp = S; Mp = M; carry = S >> 63;
      }
      col[w1][lane] = Sp | (Mp >> 1);
      if (!any) break;
      moved = true;
    }
    // new grounded height h, the lowest sand above it, and this column's extent
    int h = 0, next = -1;
    for (int w = w0; w <= w1; w++) {
      ulong v = col[w][lane];
      if (v != 0ul) { if (lo < 0) lo = w * 64 + int(ctz(v)); hi = w * 64 + 64 - int(clz(v)); }
      if (next >= 0) continue;
      if (h == w * 64) { ulong ones = ~v; int run = ones == 0ul ? 64 : int(ctz(ones)); h += run; v &= (run == 64) ? 0ul : ~((1ul << ulong(run)) - 1ul); }
      if (v != 0ul) next = w * 64 + int(ctz(v));
    }
    H[colIdx(x, z)] = uint(h) | (uint(next < 0 ? 0 : HY - next) << 16);
    if (moved) {
      for (int w = w0; w <= w1; w++) vox[addr(x, z, w)] = col[w][lane];
      atomic_store_explicit(&tgChanged, 1u, memory_order_relaxed);
    }
  } else if ((hc & 0xffffu) != 0u) { lo = 0; hi = int(hc & 0xffffu); }   // only a grounded run
  if (lo >= 0) {
    atomic_fetch_min_explicit(&tgLo, uint(lo), memory_order_relaxed);
    atomic_fetch_max_explicit(&tgHi, uint(hi), memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    uint L = atomic_load_explicit(&tgLo, memory_order_relaxed), U = atomic_load_explicit(&tgHi, memory_order_relaxed);
    bounds[tile] = L < U ? uint2(L, U) : uint2(0u);
    if (atomic_load_explicit(&tgChanged, memory_order_relaxed) != 0u) changed[tile] = 1u;
  }
}

// Granular flow on the grounded heights: the virtual-pipes model (Mei, Decaudin & Hu 2007, "Fast Hydraulic
// Erosion Simulation") with Coulomb friction for sand. Each column owns the flow on its +x and +z edges
// (cells per round, positive = away from it). The height difference pushes the flow (gravity); sliding friction
// takes off PUSH x REPOSE every round (gravity's pull at the angle of repose: 1 cell per column, 45 degrees, The
// Powder Toy's and Noita's piles); sand at rest stays at rest on slopes up to repose. Moving sand carries its
// momentum past 45 degrees, so a collapsing column runs out the way real ones do (Lube et al. 2004: a column as
// tall as it is wide keeps its height and spreads to ~2.2x its radius). Both columns of an edge apply the same
// number, so sand is conserved exactly. Tuned on a CPU copy (scratch flow2.py): 128-cube runout 277 vs 287 predicted.
#define REPOSE 1
#define PUSH 0.5f
#define FRICTION (PUSH * float(REPOSE))
#define DAMP 0.98f
kernel void flux(device const uint* H [[buffer(0)]], device short2* F [[buffer(1)]], device uint* h0 [[buffer(2)]],
                 device const uint* active [[buffer(3)]], device const uint* awake [[buffer(4)]], device uint* changed [[buffer(5)]],
                 constant Tick& T [[buffer(6)]],
                 uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  uint tile = active[tg];
  int x = int(tile % TX) * 8 + int(lane & 7), z = int(tile / TX) * 8 + int(lane >> 3);
  uint me = colIdx(x, z), hc = H[me];
  int h = int(hc & 0xffffu), freeMe = HY - int(hc >> 16) - h;
  if (T.sub == 0u) h0[me] = uint(h);
  short2 f = F[me];
  int out[2] = { 0, 0 };
  bool busy = false;
  const int2 nb[4] = { int2(1, 0), int2(0, 1), int2(-1, 0), int2(0, -1) };
  for (int k = 0; k < 4; k++) {
    int nx = x + nb[k].x, nz = z + nb[k].y;
    if (!inWorld(nx, nz)) continue;
    uint o = H[colIdx(nx, nz)];
    int hn = int(o & 0xffffu), freeN = HY - int(o >> 16) - hn;
    if (awake[tileOf(nx, nz)] == 0u) { if (abs(h - hn) > REPOSE) busy = true; continue; }   // wake it next tick
    if (k >= 2) continue;                       // -x and -z edges belong to those neighbours
    int d = h - hn;
    float prev = float(k == 0 ? f.x : f.y);
    float fl = prev * DAMP + PUSH * float(d);
    fl = sign(fl) * max(abs(fl) - FRICTION, 0.0f);                       // sliding friction
    if (abs(d) <= REPOSE && abs(prev) < FRICTION) fl = 0.0f;              // at rest, below repose: stays put
    uint key = pcg(me * 2u + uint(k) ^ pcg(T.sub ^ pcg(T.tick)));
    int fi = int(floor(fl + float(key & 0xffffu) / 65536.0f));
    // No climbing: an edge carries at most 1/8 of its height difference (rounded at random) and nothing across a
    // step of REPOSE or less. With 4 edges, no receiver can end up above a giver: no spires, height energy never rises.
    int lim = abs(d) > REPOSE ? int(floor(float(abs(d)) / 8.0f + float(key >> 16) / 65536.0f)) : 0;
    if (fi > 0) fi = min(fi, min(min(h / 4, freeN / 4), lim));
    if (fi < 0) fi = max(fi, -min(min(hn / 4, freeMe / 4), lim));
    out[k] = fi;
    if (fi != 0) busy = true;
  }
  F[me] = short2(short(out[0]), short(out[1]));
  if (busy) changed[tile] = 1u;
}
kernel void heights(device uint* H [[buffer(0)]], device const short2* F [[buffer(1)]], device const uint* active [[buffer(2)]],
                    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  uint tile = active[tg];
  int x = int(tile % TX) * 8 + int(lane & 7), z = int(tile / TX) * 8 + int(lane >> 3);
  uint me = colIdx(x, z);
  short2 f = F[me];
  int dh = -int(f.x) - int(f.y);
  if (x > 0) dh += int(F[me - 1].x);           // what the -x neighbour sent along its +x edge
  if (z > 0) dh += int(F[me - WX].y);
  if (dh != 0) { uint hc = H[me]; H[me] = uint(int(hc & 0xffffu) + dh) | (hc & 0xffff0000u); }
}

// Write the toppled heights back into the cells: set or clear the cells between the old and new height.
kernel void commit(device ulong* vox [[buffer(0)]], device const uint* H [[buffer(1)]], device const uint* h0 [[buffer(2)]],
                   device const uint* active [[buffer(3)]], device atomic_uint* bounds [[buffer(4)]],
                   uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  uint tile = active[tg];
  int x = int(tile % TX) * 8 + int(lane & 7), z = int(tile / TX) * 8 + int(lane >> 3);
  int a = int(h0[colIdx(x, z)]), b = int(H[colIdx(x, z)] & 0xffffu);
  if (a == b) return;
  int lo = min(a, b), hi = max(a, b);
  for (int w = lo >> 6; w <= (hi - 1) >> 6; w++) {
    int l = max(lo - w * 64, 0), u = min(hi - w * 64, 64);
    ulong m = (u - l == 64) ? ~0ul : (((1ul << ulong(u - l)) - 1ul) << ulong(l));
    if (b > a) vox[addr(x, z, w)] |= m; else vox[addr(x, z, w)] &= ~m;
  }
  if (b > a) { atomic_fetch_min_explicit(&bounds[tile * 2], 0u, memory_order_relaxed);
              atomic_fetch_max_explicit(&bounds[tile * 2 + 1], uint(b), memory_order_relaxed); }
}

// Height bounds of each 64x64-column block (8x8 tiles), for the rays to skip empty space.
kernel void superBounds(device const uint2* tb [[buffer(0)]], device uint2* sb [[buffer(1)]], uint s [[thread_position_in_grid]]) {
  if (s >= SX * SZ) return;
  int sx = int(s % SX), sz = int(s / SX); uint lo = 0xffffffffu, hi = 0u;
  for (int z = 0; z < 8; z++) for (int x = 0; x < 8; x++) {
    uint2 b = tb[(sz * 8 + z) * TX + sx * 8 + x];
    if (b.y > b.x) { lo = min(lo, b.x); hi = max(hi, b.y); }
  }
  sb[s] = lo < hi ? uint2(lo, hi) : uint2(0u);
}

struct Seed { int x0; int z0; int size; int y0; int y1; };
kernel void seed(device ulong* a [[buffer(0)]], device uint* H [[buffer(1)]], device uint* changed [[buffer(2)]],
                 device uint2* tb [[buffer(3)]], constant Seed& S [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  if (int(g.x) >= S.size || int(g.y) >= S.size) return;
  int x = S.x0 + int(g.x), z = S.z0 + int(g.y);
  for (int w = S.y0 >> 6; w <= (S.y1 - 1) >> 6; w++) {
    int lo = max(S.y0, w * 64) - w * 64, hi = min(S.y1, w * 64 + 64) - w * 64;
    ulong m = (hi - lo == 64) ? ~0ul : (((1ul << ulong(hi - lo)) - 1ul) << ulong(lo));
    a[addr(x, z, w)] |= m;
  }
  H[colIdx(x, z)] = (S.y0 == 0) ? uint(S.y1) : (uint(HY - S.y0) << 16);
  uint tile = uint(z >> 3) * TX + uint(x >> 3);
  changed[tile] = 1u; tb[tile] = uint2(uint(S.y0), uint(S.y1));
}

// Grain census: sand is conserved, so this must not change while the world runs.
kernel void census(device const ulong* a [[buffer(0)]], device atomic_uint* out [[buffer(1)]],
                   uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
  uint total = TX * TZ * WORDS * 64u, c = 0;
  for (uint k = i; k < total; k += n) c += popcount(a[k]);
  atomic_fetch_add_explicit(&out[i & 1023u], c, memory_order_relaxed);
}

// --- rays
struct Cam { float4 pos; float4 fwd; float4 right; float4 up; float4 sun; float tanHalf; float aspect; uint shadows; uint stats; };

inline float exitXZ(float3 o, float3 d, float3 inv, float x0, float z0, float size, thread int& axis) {
  float tx = ((d.x > 0 ? x0 + size : x0) - o.x) * inv.x;
  float tz = ((d.z > 0 ? z0 + size : z0) - o.z) * inv.z;
  axis = tx < tz ? 0 : 2; return min(tx, tz);
}
inline bool overlap(float oy, float dy, float ta, float tb, uint2 b) {
  if (b.y <= b.x) return false;
  float ya = oy + dy * ta, yb = oy + dy * tb;
  return max(ya, yb) >= float(b.x) && min(ya, yb) < float(b.y);
}
inline ulong rangeMask(int w, int y0, int y1) {
  int lo = max(y0 - w * 64, 0), hi = min(y1 - w * 64, 63);
  return (hi - lo == 63) ? ~0ul : (((1ul << ulong(hi - lo + 1)) - 1ul) << ulong(lo));
}

// First sand cell along the ray before tMax: walks 64x64 blocks, then 8x8 tiles, then columns,
// skipping any whose sand height range the ray misses; in a column, one masked word test per 64 cells.
float trace(float3 o, float3 d, float tMax, device const ulong* vox, device const uint2* tb, device const uint2* sb,
            thread float3& n, thread int3& cell, thread uint& steps) {
  float3 inv = 1.0f / d;
  float3 ta = -o * inv, tc = (float3(WX, HY, WZ) - o) * inv;
  float3 tmin = min(ta, tc), tmax = max(ta, tc);
  float tin = max(max(tmin.x, tmin.y), max(tmin.z, 0.0f));
  float tout = min(min(tmax.x, tmax.y), min(tmax.z, tMax));
  if (tin >= tout) return -1.0f;
  int axis = tin > 0 ? ((tmin.x >= tmin.y && tmin.x >= tmin.z) ? 0 : (tmin.z >= tmin.y ? 2 : 1)) : -1;
  float t = tin;
  for (uint i = 0; i < 4096u && t < tout; i++) {
    steps++;
    float3 p = o + d * (t + 1e-3f + t * 2e-6f);
    int cx = clamp(int(floor(p.x)), 0, WX - 1), cz = clamp(int(floor(p.z)), 0, WZ - 1);
    int ax; float te;
    int sx = cx >> 6, sz = cz >> 6;
    te = exitXZ(o, d, inv, float(sx * 64), float(sz * 64), 64.0f, ax);
    if (!overlap(o.y, d.y, t, min(te, tout), sb[sz * SX + sx])) { t = max(te, t + 1e-4f); axis = ax; continue; }
    int tx = cx >> 3, tz = cz >> 3;
    te = exitXZ(o, d, inv, float(tx * 8), float(tz * 8), 8.0f, ax);
    uint2 b = tb[tz * TX + tx];
    if (!overlap(o.y, d.y, t, min(te, tout), b)) { t = max(te, t + 1e-4f); axis = ax; continue; }
    te = exitXZ(o, d, inv, float(cx), float(cz), 1.0f, ax);
    float t1 = min(te, tout);
    float ya = o.y + d.y * t, yb = o.y + d.y * t1;
    int y0 = max(int(floor(min(ya, yb))), int(b.x)), y1 = min(int(floor(max(ya, yb))), int(b.y) - 1);
    if (y0 <= y1) {
      int hy = -1;
      if (d.y < 0) {
        for (int w = y1 >> 6; w >= (y0 >> 6) && hy < 0; w--) { ulong m = vox[addr(cx, cz, w)] & rangeMask(w, y0, y1); if (m != 0ul) hy = w * 64 + 63 - int(clz(m)); }
      } else {
        for (int w = y0 >> 6; w <= (y1 >> 6) && hy < 0; w++) { ulong m = vox[addr(cx, cz, w)] & rangeMask(w, y0, y1); if (m != 0ul) hy = w * 64 + int(ctz(m)); }
      }
      if (hy >= 0) {
        if (hy == int(floor(ya)) && axis >= 0 && axis != 1) {
          n = axis == 0 ? float3(-sign(d.x), 0, 0) : float3(0, 0, -sign(d.z));
          cell = int3(cx, hy, cz); return t;
        }
        float yp = d.y < 0 ? float(hy + 1) : float(hy);
        n = float3(0, d.y < 0 ? 1 : -1, 0); cell = int3(cx, hy, cz);
        return (yp - o.y) * inv.y;
      }
    }
    t = max(te, t + 1e-4f); axis = ax;
  }
  return -1.0f;
}

kernel void render(texture2d<float, access::write> out [[texture(0)]],
                   device const ulong* vox [[buffer(0)]], device const uint2* tb [[buffer(1)]], device const uint2* sb [[buffer(2)]],
                   constant Cam& C [[buffer(3)]], device atomic_uint* stepsOut [[buffer(4)]],
                   uint2 g [[thread_position_in_grid]]) {
  uint W = out.get_width(), H = out.get_height();
  if (g.x >= W || g.y >= H) return;
  float2 uv = (float2(g) + 0.5f) / float2(W, H) * 2.0f - 1.0f; uv.y = -uv.y;
  float3 d = normalize(C.fwd.xyz + uv.x * C.tanHalf * C.aspect * C.right.xyz + uv.y * C.tanHalf * C.up.xyz);
  d = select(d, float3(1e-7f), abs(d) < 1e-7f);
  float3 o = C.pos.xyz, L = C.sun.xyz;
  float3 sky = mix(float3(0.70, 0.80, 0.93), float3(0.30, 0.50, 0.85), saturate(d.y * 2.0f));
  float3 sunCol = float3(1.0, 0.92, 0.80) * 1.15f;
  float tFloor = d.y < 0 ? -o.y / d.y : 1e30f;
  float3 n; int3 cell; uint steps = 0;
  float th = trace(o, d, tFloor, vox, tb, sb, n, cell, steps);
  float3 col; float tt;
  if (th >= 0) {
    float h = float(pcg(uint(cell.x) ^ pcg(uint(cell.y) ^ pcg(uint(cell.z)))) & 1023u) / 1023.0f;
    float3 base = float3(0.78, 0.62, 0.38) * (0.75f + 0.4f * h);
    float lam = max(dot(n, L), 0.0f), sh = 1.0f;
    if (C.shadows != 0u && lam > 0) { float3 n2; int3 c2; if (trace(o + d * th + n * 0.01f, L, 1e30f, vox, tb, sb, n2, c2, steps) >= 0) sh = 0.0f; }
    col = base * (sky * (n.y > 0.5f ? 0.40f : 0.28f) + sunCol * lam * sh);
    tt = th;
  } else if (tFloor < 1e29f) {
    float3 fp = o + d * tFloor;
    bool inside = fp.x >= 0 && fp.z >= 0 && fp.x < WX && fp.z < WZ;
    float chk = fmod(abs(floor(fp.x / 256.0f) + floor(fp.z / 256.0f)), 2.0f);
    float3 base = inside ? mix(float3(0.36, 0.35, 0.33), float3(0.42, 0.41, 0.38), chk) : float3(0.30, 0.34, 0.27);
    float sh = 1.0f;
    if (C.shadows != 0u) { float3 n2; int3 c2; if (trace(fp + float3(0, 0.01f, 0), L, 1e30f, vox, tb, sb, n2, c2, steps) >= 0) sh = 0.0f; }
    col = base * (sky * 0.40f + sunCol * max(L.y, 0.0f) * sh);
    tt = tFloor;
  } else { col = sky; tt = 1e30f; }
  if (tt < 1e29f) col = mix(col, sky, 1.0f - exp(-tt / 30000.0f));
  if (C.stats != 0u) atomic_fetch_add_explicit(stepsOut, steps, memory_order_relaxed);
  out.write(float4(col, 1.0f), g);
}

// --- present: stretch the half-resolution frame over the window
struct VOut { float4 pos [[position]]; float2 uv; };
vertex VOut presentV(uint vid [[vertex_id]]) {
  float2 p = float2((vid << 1) & 2, vid & 2);
  VOut o; o.pos = float4(p * 2.0f - 1.0f, 0, 1); o.uv = float2(p.x, 1.0f - p.y); return o;
}
fragment float4 presentF(VOut in [[stage_in]], texture2d<float> tex [[texture(0)]]) {
  constexpr sampler s(filter::linear, address::clamp_to_edge);
  return float4(tex.sample(s, in.uv).rgb, 1.0f);
}
""" }
