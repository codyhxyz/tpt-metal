// The GPU side. Precompiled to BigSand.metallib by scripts/build-shaders.sh; compiled from this source at launch only as a fallback.
//
// Units: a cell is 1 cm; a tick is 1/60 s.
// Sand is in one of two places:
//   - in the air: per column, one run of cells [lo, hi) that falls together with the column's speed (gravity
//     accelerates it). 8 bytes per column, however tall the run: moving it changes two numbers, not its cells.
//   - in the pile: a shallow granular flow (Savage-Hutter) on 8x8-column cells = the tiles. Each tile holds its exact
//     grain count V and its momentum Q; the pile's surface (one height per column) is rebuilt from them every tick.
// Grains move from the air to the pile when they reach it; the total (sum of V + run lengths) never changes.
// A column's airborne sand is always one solid run: it starts as one, falls as a whole, and only loses grains at its
// bottom, to the pile.
//
// The world's size is set when the pipelines are built (function constants), so one compiled library serves any.

#include <metal_stdlib>
using namespace metal;
constant int WX [[function_constant(0)]];          // columns along x and z: multiples of 64
constant int WZ [[function_constant(1)]];
constant int HY [[function_constant(2)]];          // cells of height, at most 65535 (a run's ends are 16 bits)
constant int SUBSTEPS [[function_constant(3)]];    // flow steps per tick (stability; see below)
#define TX (WX / 8)
#define TZ (WZ / 8)
#define SX (WX / 64)
#define SZ (WZ / 64)
#define WAKE 16u            // ticks a tile stays awake after the last change near it

// --- physics constants
#define G 0.2725f           // gravity: 9.81 m/s^2 = 981 cells/s^2 = 0.2725 cells/tick^2
#define VMAX 30.0f          // fastest fall, cells/tick (18 m/s)
#define MU 0.6494f          // Coulomb friction of dry sand: tan 33 deg, its angle of repose
// SUBSTEPS: for stability (|u| + sqrt(g h)) dt/dx <= 0.5 with |u| <= UMAX; the engine picks it from the deepest
// pile the cube can make.
#define DT (1.0f / float(SUBSTEPS))
#define DX 8.0f             // a flow cell is one 8x8-column tile
#define UMAX 25.0f          // flow speed cap, cells/tick (15 m/s)

// --- the sun (see sunShadow)
#define SUN_DIR float3(-0.55f, 0.70f, -0.35f)   // toward the sun (the camera's default view faces away from it)
#define SUN_M (SUN_DIR.z / abs(SUN_DIR.x))      // z tiles per x tile along a sun line (|SUN_DIR.x| >= |SUN_DIR.z|)
#define SUN_RISE (SUN_DIR.y / abs(SUN_DIR.x) * DX)   // sun-ray height gained per x tile, in cells
#define U0 int(floor(min(SUN_M, 0.0f) * float(TX - 1)))
#define NL (TZ - U0 + int(ceil(max(SUN_M, 0.0f) * float(TX - 1))) + 1)   // lines

inline uint colIdx(int x, int z) { return uint(z) * uint(WX) + uint(x); }

// A column's airborne run, 8 bytes: x = lo | hi << 16 (cells [lo, hi); empty if hi <= lo);
// y = speed in 1/2048 cell per tick | how far into the next cell it has travelled, in 1/65536 cell, << 16.
#define VQ 2048.0f
#define GQ uint(G * VQ + 0.5f)
#define VMAXQ uint(VMAX * VQ)
inline int runLo(uint2 r) { return int(r.x & 0xffffu); }
inline int runHi(uint2 r) { return int(r.x >> 16); }
inline uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }

struct Tick { uint tick; uint sub; uint pad0; uint pad1; };

// Which tiles run this tick: any tile with a change in its 3x3 neighbourhood in the last WAKE ticks.
kernel void plan(device const uint* changedLast [[buffer(0)]], device uint* changedNow [[buffer(1)]],
                 device uint* countdown [[buffer(2)]], device uint* awake [[buffer(3)]],
                 device atomic_uint* args [[buffer(4)]], device uint* active [[buffer(5)]],
                 uint t [[thread_position_in_grid]]) {
  if (t >= uint(TX * TZ)) return;
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
// One thread per active tile for the flow: threadgroups of 64.
// And the sun-height pass: all sun lines, but only on ticks where any tile ran (a world at rest keeps last tick's map).
kernel void flowArgs(device const uint* args [[buffer(0)]], device uint* args2 [[buffer(1)]], device uint* args3 [[buffer(2)]]) {
  args2[0] = (args[0] + 63u) / 64u; args2[1] = 1u; args2[2] = 1u;
  args3[0] = args[0] > 0u ? uint(NL + 63) / 64u : 0u; args3[1] = 1u; args3[2] = 1u;
}

// --- air: every column's run falls with the column's speed: two numbers change, however tall the run.
// Grains that reach the pile's surface (or the floor) this tick leave the run's bottom and join the pile's count.
// Columns whose run is above the highest the pile has been don't need the pile's height at all.
kernel void fall(device uint2* runs [[buffer(0)]], device const uint* HF [[buffer(1)]], device const uint* pileTop [[buffer(2)]],
                 device atomic_int* V [[buffer(3)]], device const uint* active [[buffer(4)]], device uint* changed [[buffer(5)]],
                 device uint2* air [[buffer(6)]],
                 uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint tgLo, tgHi, tgAny;
  threadgroup atomic_int tgLanded;
  uint tile = active[tg];
  uint2 ae = air[tile];
  if (ae.y <= ae.x) return;                  // nothing in the air over this tile (uniform for the whole group)
  int x = int(tile % TX) * 8 + int(lane & 7), z = int(tile / TX) * 8 + int(lane >> 3);
  if (lane == 0) {
    atomic_store_explicit(&tgLo, 0xffffffffu, memory_order_relaxed); atomic_store_explicit(&tgHi, 0u, memory_order_relaxed);
    atomic_store_explicit(&tgAny, 0u, memory_order_relaxed); atomic_store_explicit(&tgLanded, 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint ci = colIdx(x, z);
  uint2 r = runs[ci];
  int lo = runLo(r), hi = runHi(r), landed = 0, nlo = 0, nhi = 0;
  if (hi > lo) {
    uint v = min((r.y & 0xffffu) + GQ, VMAXQ);
    uint s = (r.y >> 16) + v * uint(65536.0f / VQ);
    int d = int(s >> 16);                    // whole cells fallen this tick
    int cut = lo >= int(*pileTop) + d ? lo : int(HF[ci]) + d;   // everything below this reaches the pile (or the floor)
    int keep = clamp(cut, lo, hi);
    landed = keep - lo;
    nlo = keep - d; nhi = hi - d;
    uint2 o = nhi > nlo ? uint2(uint(nlo) | (uint(nhi) << 16), v | ((s & 0xffffu) << 16)) : uint2(0u);
    runs[ci] = o;
  }
  if (landed > 0) atomic_fetch_add_explicit(&tgLanded, landed, memory_order_relaxed);
  if (nhi > nlo) { atomic_fetch_min_explicit(&tgLo, uint(nlo), memory_order_relaxed); atomic_fetch_max_explicit(&tgHi, uint(nhi), memory_order_relaxed); }
  if (nhi > nlo || landed > 0) atomic_store_explicit(&tgAny, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    int L = atomic_load_explicit(&tgLanded, memory_order_relaxed);
    if (L > 0) atomic_fetch_add_explicit(&V[tile], L, memory_order_relaxed);
    uint A = atomic_load_explicit(&tgLo, memory_order_relaxed), U = atomic_load_explicit(&tgHi, memory_order_relaxed);
    air[tile] = A < U ? uint2(A, U) : uint2(0u);
    if (atomic_load_explicit(&tgAny, memory_order_relaxed) != 0u) changed[tile] = 1u;
  }
}

// --- the pile: shallow granular flow (Savage & Hutter 1989), depth-averaged over the pile's full depth, on tiles.
// h = V/64 cells deep, momentum Q = h u. Finite volumes with the Rusanov flux; Coulomb basal friction mu g per unit
// mass, applied after each step (Mangeney et al. 2003): a tile whose speed friction can cancel this step stops.
// A face between two tiles at rest whose step is within the angle of repose carries nothing (a pile at rest stays
// at rest exactly); it pushes on both like a wall. Grains crossing a face are rounded the same way by both tiles.
inline float3 rusanov(float hL, float unL, float utL, float hR, float unR, float utR) {
  float a = max(abs(unL) + sqrt(G * hL), abs(unR) + sqrt(G * hR));
  float qL = hL * unL, qR = hR * unR;
  return float3(0.5f * (qL + qR) - 0.5f * a * (hR - hL),
                0.5f * (qL * unL + 0.5f * G * hL * hL + qR * unR + 0.5f * G * hR * hR) - 0.5f * a * (qR - qL),
                0.5f * (qL * utL + qR * utR) - 0.5f * a * (hR * utR - hL * utL));
}
kernel void flow(device const int* Vin [[buffer(0)]], device const float2* Qin [[buffer(1)]],
                 device int* Vout [[buffer(2)]], device float2* Qout [[buffer(3)]],
                 device const uint* active [[buffer(4)]], device const uint* args [[buffer(5)]],
                 device const uint* awake [[buffer(6)]], device uint* changed [[buffer(7)]],
                 constant Tick& T [[buffer(8)]], uint i [[thread_position_in_grid]]) {
  if (i >= args[0]) return;
  uint t = active[i];
  int tx = int(t % TX), tz = int(t / TX);
  int V = Vin[t]; float2 Q = Qin[t];
  float h = float(V) / 64.0f;
  float2 u = h > 1e-3f ? Q / h : float2(0.0f);
  bool still = Q.x == 0.0f && Q.y == 0.0f;
  int dV = 0; float2 dQ = float2(0.0f); bool wake = false;
  const int2 nb[4] = { int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1) };
  for (int k = 0; k < 4; k++) {
    int nx = tx + nb[k].x, nz = tz + nb[k].y;
    bool ax = k < 2;                       // face normal along x
    float sgn = (k & 1) ? -1.0f : 1.0f;    // +1: this tile is on the low side of the face
    bool wall = nx < 0 || nz < 0 || nx >= TX || nz >= TZ;
    uint n = wall ? t : uint(nz * TX + nx);
    int Vn = Vin[n]; float2 Qn = Qin[n];
    float hn = float(Vn) / 64.0f;
    if (!wall && awake[n] == 0u) { if (abs(hn - h) / DX > MU) wake = true; wall = true; }
    if (!wall && still && Qn.x == 0.0f && Qn.y == 0.0f && abs(hn - h) / DX <= MU) wall = true;   // at rest, within repose
    if (wall) {                            // no grains cross; the face pushes back with this tile's own pressure
      float p = 0.5f * G * h * h * DT / DX;
      if (ax) dQ.x -= sgn * p; else dQ.y -= sgn * p;
      continue;
    }
    float2 un = hn > 1e-3f ? Qn / hn : float2(0.0f);
    float uN = ax ? u.x : u.y, uT = ax ? u.y : u.x, nN = ax ? un.x : un.y, nT = ax ? un.y : un.x;
    float3 F = sgn > 0.0f ? rusanov(h, uN, uT, hn, nN, nT) : rusanov(hn, nN, nT, h, uN, uT);
    uint lowT = min(t, n);
    uint key = pcg((lowT * 2u + (ax ? 0u : 1u)) ^ pcg(T.sub ^ pcg(T.tick)));
    int gi = int(floor(F.x * DX * DT + float(key & 0xffffu) / 65536.0f));   // grains low -> high this step
    int Vlow = sgn > 0.0f ? V : Vn, Vhigh = sgn > 0.0f ? Vn : V;
    gi = gi > 0 ? min(gi, Vlow / 4) : max(gi, -(Vhigh / 4));                 // never more than a quarter of the giver
    dV += sgn > 0.0f ? -gi : gi;
    if (ax) { dQ.x -= sgn * F.y * DT / DX; dQ.y -= sgn * F.z * DT / DX; }
    else    { dQ.y -= sgn * F.y * DT / DX; dQ.x -= sgn * F.z * DT / DX; }
  }
  int Vnew = V + dV;
  float2 Qnew = Q + dQ;
  float hnew = float(Vnew) / 64.0f;
  if (hnew < 1e-3f) Qnew = float2(0.0f);
  else {
    float2 v = Qnew / hnew; float s = length(v), dec = MU * G * DT;
    v = s <= dec ? float2(0.0f) : v * (1.0f - dec / s);                       // Coulomb friction
    s = length(v); if (s > UMAX) v *= UMAX / s;
    Qnew = v * hnew;
  }
  Vout[t] = Vnew; Qout[t] = Qnew;
  if (dV != 0 || Qnew.x != 0.0f || Qnew.y != 0.0f || wake) changed[t] = 1u;
}

// The pile's surface: per tile a plane through its mean height, sloped toward its neighbours (minmod-limited, so
// no new peaks), integer heights whose sum is exactly the tile's grain count.
inline float minmod(float a, float b) { return a * b <= 0.0f ? 0.0f : (abs(a) < abs(b) ? a : b); }
kernel void surface(device const int* V [[buffer(0)]], device uint* HF [[buffer(1)]], device const uint* active [[buffer(2)]],
                    device const uint2* air [[buffer(3)]], device uint2* bounds [[buffer(4)]], device atomic_uint* pileTop [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  threadgroup atomic_int tgSum; threadgroup atomic_uint tgTop;
  uint t = active[tg];
  int tx = int(t % TX), tz = int(t / TX), i = int(lane & 7), j = int(lane >> 3);
  if (lane == 0) { atomic_store_explicit(&tgSum, 0, memory_order_relaxed); atomic_store_explicit(&tgTop, 0u, memory_order_relaxed); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int Vt = V[t];
  float hb = float(Vt) / 64.0f;
  float hl = tx > 0 ? float(V[t - 1]) / 64.0f : hb, hr = tx < TX - 1 ? float(V[t + 1]) / 64.0f : hb;
  float hd = tz > 0 ? float(V[t - TX]) / 64.0f : hb, hu = tz < TZ - 1 ? float(V[t + TX]) / 64.0f : hb;
  float sx = minmod((hb - hl) / DX, (hr - hb) / DX), sz = minmod((hb - hd) / DX, (hu - hb) / DX);
  float reach = 3.5f * (abs(sx) + abs(sz));
  if (reach > hb && reach > 0.0f) { sx *= hb / reach; sz *= hb / reach; }   // no column below the floor
  int fl = max(0, int(floor(hb + sx * (float(i) - 3.5f) + sz * (float(j) - 3.5f))));
  atomic_fetch_add_explicit(&tgSum, fl, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int R = Vt - atomic_load_explicit(&tgSum, memory_order_relaxed);
  int rank = int((lane * 37u) & 63u);      // a fixed scattered order for handing out the leftover grains
  int hf = fl + ((R > 0 && rank < R) ? 1 : 0) - ((R < 0 && rank < -R && fl > 0) ? 1 : 0);
  HF[colIdx(tx * 8 + i, tz * 8 + j)] = uint(hf);
  atomic_fetch_max_explicit(&tgTop, uint(hf), memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    uint top = atomic_load_explicit(&tgTop, memory_order_relaxed);
    atomic_fetch_max_explicit(pileTop, top, memory_order_relaxed);
    uint2 a = air[t];
    bool hasAir = a.y > a.x;
    uint lo = top > 0u ? 0u : (hasAir ? a.x : 0u), hi = max(top, hasAir ? a.y : 0u);
    bounds[t] = hi > lo ? uint2(lo, hi) : uint2(0u);
  }
}

// Height bounds of each 64x64-column block (8x8 tiles), for the rays to skip empty space: of all sand (sb), and of
// the sand in the air only (sa), which is all a shadow ray has to look for.
kernel void superBounds(device const uint2* tb [[buffer(0)]], device uint2* sb [[buffer(1)]],
                        device const uint2* air [[buffer(2)]], device uint2* sa [[buffer(3)]], uint s [[thread_position_in_grid]]) {
  if (s >= uint(SX * SZ)) return;
  int sx = int(s % SX), sz = int(s / SX); uint lo = 0xffffffffu, hi = 0u, alo = 0xffffffffu, ahi = 0u;
  for (int z = 0; z < 8; z++) for (int x = 0; x < 8; x++) {
    uint t = (sz * 8 + z) * TX + sx * 8 + x;
    uint2 b = tb[t], a = air[t];
    if (b.y > b.x) { lo = min(lo, b.x); hi = max(hi, b.y); }
    if (a.y > a.x) { alo = min(alo, a.x); ahi = max(ahi, a.y); }
  }
  sb[s] = lo < hi ? uint2(lo, hi) : uint2(0u);
  sa[s] = alo < ahi ? uint2(alo, ahi) : uint2(0u);
}

// --- the sun's height over the pile: the height below which the pile itself blocks the sun, so a pixel's pile shadow
// is one lookup instead of a ray. Rebuilt only on ticks where something moved; a world at rest keeps it.
// One sweep per line of tiles toward the sun: a line steps one tile in x (the sun's major axis) and SUN_M tiles in z,
// and carries the running answer S(i) = max(h(i-1), S(i-1)) - rise, where h is the pile's mean height at the line's
// previous point. One sample per tile per line, instead of a march of up to (pile height / rise) steps per tile.
// Lines are indexed by u = z + SUN_M * x (constant along a line); the map stores S at (column x, line u - U0).
inline float tileMeanZ(device const int* V, int x, float zc) {   // mean height at tile column x, tile row zc (centres at integers)
  if (zc <= -1.0f || zc >= float(TZ)) return 0.0f;
  int z0 = int(floor(zc)); float f = zc - float(z0);
  float a = z0 >= 0 ? float(V[uint(z0) * uint(TX) + uint(x)]) : 0.0f;
  float b = z0 + 1 < TZ ? float(V[uint(z0 + 1) * uint(TX) + uint(x)]) : 0.0f;
  return mix(a, b, f) / 64.0f;
}
kernel void sunShadow(device const int* V [[buffer(0)]], device float* S [[buffer(1)]], uint j [[thread_position_in_grid]]) {
  if (int(j) >= NL) return;
  float u = float(int(j) + U0);
  int x0 = SUN_DIR.x < 0.0f ? 0 : TX - 1, step = SUN_DIR.x < 0.0f ? 1 : -1;   // start at the sun's side
  float carry = 0.0f;                         // max(h, S) at the line's previous point
  for (int k = 0; k < TX; k++) {
    int x = x0 + k * step;
    float sv = max(carry - SUN_RISE, 0.0f);
    S[uint(x) * uint(NL) + j] = sv;
    carry = max(tileMeanZ(V, x, u - SUN_M * float(x)), sv);
  }
}
inline float sunHeight(device const float* S, float x, float z) {   // bilinear in (column, line), in cells
  float cx = clamp(x / DX - 0.5f, 0.0f, float(TX - 1)), cz = z / DX - 0.5f;
  float cu = clamp(cz + SUN_M * cx - float(U0), 0.0f, float(NL - 1));
  int2 i = min(int2(int(cx), int(cu)), int2(TX - 2, NL - 2)); float2 f = float2(cx, cu) - float2(i);
  uint t = uint(i.x) * uint(NL) + uint(i.y);
  return mix(mix(S[t], S[t + 1], f.y), mix(S[t + uint(NL)], S[t + uint(NL) + 1], f.y), f.x);
}

struct Seed { int x0; int z0; int size; int y0; int y1; };
kernel void seed(device uint2* runs [[buffer(0)]], device uint* changed [[buffer(1)]], device uint2* air [[buffer(2)]],
                 device uint2* tb [[buffer(3)]], constant Seed& S [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  if (int(g.x) >= S.size || int(g.y) >= S.size) return;
  int x = S.x0 + int(g.x), z = S.z0 + int(g.y);
  runs[colIdx(x, z)] = uint2(uint(S.y0) | (uint(S.y1) << 16), 0u);   // at rest
  uint tile = uint(z >> 3) * TX + uint(x >> 3);
  changed[tile] = 1u; air[tile] = uint2(uint(S.y0), uint(S.y1)); tb[tile] = uint2(uint(S.y0), uint(S.y1));
}

// Grain census: airborne run lengths + every tile's pile count. Sand is conserved, so this never changes.
kernel void census(device const uint2* runs [[buffer(0)]], device const int* V [[buffer(1)]], device atomic_uint* out [[buffer(2)]],
                   uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
  uint total = uint(WX) * uint(WZ), c = 0;
  for (uint k = i; k < total; k += n) { uint2 r = runs[k]; c += uint(max(runHi(r) - runLo(r), 0)); }
  for (uint k = i; k < uint(TX * TZ); k += n) c += uint(V[k]);
  atomic_fetch_add_explicit(&out[i & 1023u], c, memory_order_relaxed);
}

// --- rays
struct Cam { float4 pos; float4 fwd; float4 right; float4 up; float4 sun; float tanHalf; float aspect; uint shadows; uint stats; float2 jitter; float2 pad; };

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

// First sand cell along the ray before tMax: walks 64x64 blocks, then 8x8 tiles, then columns, skipping any whose
// sand height range the ray misses. A column is sand below its pile height, plus its airborne run.
float trace(float3 o, float3 d, float tMax, device const uint2* runs, device const uint* HF, device const uint2* tb,
            device const uint2* sb, bool withPile, thread float3& n, thread int3& cell, thread uint& steps) {
  float3 inv = 1.0f / d;
  float3 ta = -o * inv, tc = (float3(WX, HY, WZ) - o) * inv;
  float3 tmin = min(ta, tc), tmax = max(ta, tc);
  float tin = max(max(tmin.x, tmin.y), max(tmin.z, 0.0f));
  float tout = min(min(tmax.x, tmax.y), min(tmax.z, tMax));
  if (tin >= tout) return -1.0f;
  int axis = tin > 0 ? ((tmin.x >= tmin.y && tmin.x >= tmin.z) ? 0 : (tmin.z >= tmin.y ? 2 : 1)) : -1;
  float t = tin;
  for (uint i = 0; i < 16384u && t < tout; i++) {
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
      uint ci = colIdx(cx, cz);
      int pile = withPile ? int(HF[ci]) : 0;   // cells [0, pile) are sand
      uint2 r = runs[ci];
      int alo = runLo(r), ahi = runHi(r);         // and cells [alo, ahi)
      bool inAir = alo <= y1 && ahi - 1 >= y0;
      int hy = -1;
      if (d.y < 0) {                              // going down: the highest sand cell in [y0, y1]
        if (inAir) hy = min(y1, ahi - 1);
        if (pile - 1 >= y0) hy = max(hy, min(y1, pile - 1));
      } else {                                    // going up: the lowest
        if (y0 < pile) hy = y0;
        else if (inAir) hy = max(y0, alo);
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

// 1 if the sun reaches point p, else 0: under the pile's sun height (one lookup), or behind sand in the air (a ray that
// only visits tiles with sand in the air, so it costs nothing once everything has landed).
#define SUN_SLACK 1.0f   // cells: the sun-height map is per tile, the surface per column
inline float sunlit(float3 p, float3 L, device const uint2* runs, device const uint* HF, device const float* S,
                    device const uint2* air, device const uint2* superAir, thread uint& steps) {
  if (p.x >= 0 && p.z >= 0 && p.x < WX && p.z < WZ && p.y + SUN_SLACK < sunHeight(S, p.x, p.z)) return 0.0f;
  float3 n2; int3 c2;
  return trace(p, L, 1e30f, runs, HF, air, superAir, false, n2, c2, steps) >= 0 ? 0.0f : 1.0f;
}

// Writes MetalFX's inputs: colour, depth (reversed: NEAR / view distance, 0 = sky), and motion: per pixel, how many
// pixels away this surface point was on the previous frame (its spot under the previous camera minus its spot now).
#define NEAR 1.0f
kernel void render(texture2d<float, access::write> out [[texture(0)]], texture2d<float, access::write> depthOut [[texture(1)]],
                   texture2d<float, access::write> motionOut [[texture(2)]],
                   device const uint2* runs [[buffer(0)]], device const uint* HF [[buffer(1)]], device const uint2* tb [[buffer(2)]],
                   device const uint2* sb [[buffer(3)]], constant Cam& C [[buffer(4)]], device atomic_uint* stepsOut [[buffer(5)]],
                   constant Cam& P [[buffer(6)]], device const float* S [[buffer(7)]], device const uint2* air [[buffer(8)]],
                   device const uint2* superAir [[buffer(9)]], uint2 g [[thread_position_in_grid]]) {
  uint W = out.get_width(), H = out.get_height();
  if (g.x >= W || g.y >= H) return;
  float2 uv = (float2(g) + 0.5f + C.jitter) / float2(W, H) * 2.0f - 1.0f; uv.y = -uv.y;   // jittered within the pixel (TAA)
  float3 d = normalize(C.fwd.xyz + uv.x * C.tanHalf * C.aspect * C.right.xyz + uv.y * C.tanHalf * C.up.xyz);
  d = select(d, float3(1e-7f), abs(d) < 1e-7f);
  float3 o = C.pos.xyz, L = normalize(SUN_DIR);
  float3 sky = mix(float3(0.70, 0.80, 0.93), float3(0.30, 0.50, 0.85), saturate(d.y * 2.0f));
  float3 sunCol = float3(1.0, 0.92, 0.80) * 1.15f;
  float tFloor = d.y < 0 ? -o.y / d.y : 1e30f;
  float3 n; int3 cell; uint steps = 0;
  float th = trace(o, d, tFloor, runs, HF, tb, sb, true, n, cell, steps);
  float3 col; float tt;
  if (th >= 0) {
    // The pile is a smooth surface stored as whole-cell heights: light it with its real slope (central differences
    // of the heights), not the 1 cm stair faces, whose risers alias into contour rings at a distance.
    int pile = int(HF[colIdx(cell.x, cell.z)]);
    bool onPile = cell.y < pile;
    if (onPile) {
      int xl = max(cell.x - 1, 0), xr = min(cell.x + 1, WX - 1), zl = max(cell.z - 1, 0), zr = min(cell.z + 1, WZ - 1);
      float gx = (float(HF[colIdx(xr, cell.z)]) - float(HF[colIdx(xl, cell.z)])) / float(xr - xl);
      float gz = (float(HF[colIdx(cell.x, zr)]) - float(HF[colIdx(cell.x, zl)])) / float(zr - zl);
      n = normalize(float3(-gx, 1.0f, -gz));
    }
    // Grain colour varies per cell; a pixel covering k x k grains shows their average, which varies 1/k as much
    // (the mipmap rule), so the variation fades with the pixel's footprint instead of shimmering.
    float footprint = th * 2.0f * C.tanHalf / float(H);          // cells per pixel at the hit
    float h = float(pcg(uint(cell.x) ^ pcg(uint(cell.y) ^ pcg(uint(cell.z)))) & 1023u) / 1023.0f;
    float3 base = float3(0.78, 0.62, 0.38) * (0.95f + (0.4f * h - 0.2f) / max(1.0f, footprint));
    float lam = max(dot(n, L), 0.0f), sh = 1.0f;
    if (C.shadows != 0u && lam > 0) sh = sunlit(o + d * th + n * (onPile ? 1.5f : 0.01f), L, runs, HF, S, air, superAir, steps);
    col = base * (sky * mix(0.28f, 0.40f, saturate(n.y)) + sunCol * lam * sh);
    tt = th;
  } else if (tFloor < 1e29f) {
    float3 fp = o + d * tFloor;
    bool inside = fp.x >= 0 && fp.z >= 0 && fp.x < WX && fp.z < WZ;
    float chk = fmod(abs(floor(fp.x / 256.0f) + floor(fp.z / 256.0f)), 2.0f);
    float3 base = inside ? mix(float3(0.36, 0.35, 0.33), float3(0.42, 0.41, 0.38), chk) : float3(0.30, 0.34, 0.27);
    float sh = 1.0f;
    if (C.shadows != 0u) sh = sunlit(fp + float3(0, 0.01f, 0), L, runs, HF, S, air, superAir, steps);
    col = base * (sky * 0.40f + sunCol * max(L.y, 0.0f) * sh);
    tt = tFloor;
  } else { col = sky; tt = 1e30f; }
  if (tt < 1e29f) col = mix(col, sky, 1.0f - exp(-tt / (7.32f * float(WX))));   // haze: 30000 cells in a 4096 world
  if (C.stats != 0u) atomic_fetch_add_explicit(stepsOut, steps, memory_order_relaxed);
  out.write(float4(col, 1.0f), g);
  float dist = tt < 1e29f ? tt : 1e9f;
  depthOut.write(float4(tt < 1e29f ? NEAR / max(dist * dot(d, C.fwd.xyz), NEAR) : 0.0f), g);
  float3 v = o + d * dist - P.pos.xyz;
  float z = dot(v, P.fwd.xyz);
  float2 motion = float2(0.0f);
  if (z > 0.0f) {
    float2 q = float2(dot(v, P.right.xyz) / (z * P.tanHalf * P.aspect), dot(v, P.up.xyz) / (z * P.tanHalf));
    float2 prevPx = float2(q.x * 0.5f + 0.5f, 0.5f - q.y * 0.5f) * float2(W, H);
    motion = prevPx - (float2(g) + 0.5f + C.jitter);
  }
  motionOut.write(float4(motion, 0.0f, 0.0f), g);
}

// --- present: copy MetalFX's full-resolution output to the window
struct VOut { float4 pos [[position]]; float2 uv; };
vertex VOut presentV(uint vid [[vertex_id]]) {
  float2 p = float2((vid << 1) & 2, vid & 2);
  VOut o; o.pos = float4(p * 2.0f - 1.0f, 0, 1); o.uv = float2(p.x, 1.0f - p.y); return o;
}
fragment float4 presentF(VOut in [[stage_in]], texture2d<float> tex [[texture(0)]]) {
  constexpr sampler s(filter::linear, address::clamp_to_edge);
  return float4(tex.sample(s, in.uv).rgb, 1.0f);
}
