// World: a general voxel world. Every cell is one byte of material (0 = air), stored sparsely in three levels:
//   chunk  64^3 cells: one grid entry each. A uniform chunk stores its material; the rest point to a table.
//   brick   8^3 cells: one table entry each (512 per table). A uniform brick stores its material; the rest point
//                      to 512 bytes in the brick pool.
//   cell    one byte.
// An entry with the top bit set is uniform (material in its low byte); otherwise it is an index into a pool.
// Tables and bricks live in two Metal 4 placement-sparse buffers: huge virtual ranges, with memory mapped only as
// the pools fill. Memory follows surfaces (where materials meet), not volume.
//
// Heights are computed once and stored per column; caves come from integer value noise, so every kernel that
// evaluates a cell gets exactly the same answer.
#include <metal_stdlib>
using namespace metal;

constant int NW [[function_constant(0)]];   // cells across, in x and in z (multiple of 64)
constant int NH [[function_constant(1)]];   // cells tall (multiple of 64)

#define W_CX (NW / 64)
#define W_CY (NH / 64)
#define W_BX (NW / 8)
#define W_UNI 0x80000000u
#define W_NONE 0xffffffffu
#define W_SEA 192            // sea level, in cells
#define W_CAVE_D 120         // caves reach this far below the surface
#define W_CAVE_T 3300        // cave radius in noise units (of 32767): tunnels about 10 cells across
#define W_TOPMAX 5           // deepest surface layer (beach sand)
#define W_NEAR 1.0f
#define W_AMBIENT float3(0.32f, 0.37f, 0.45f)   // light from the sky

// counters (one int buffer)
#define C_TBUMP 0            // tables handed out from the end of the pool
#define C_BBUMP 1            // bricks handed out from the end of the pool
#define C_TFREE 2            // tables on the free stack
#define C_BFREE 3            // bricks on the free stack
#define C_CAND 4             // generation: candidate bricks in this batch
#define C_OVER 5             // set when an allocation found the pool full
#define C_TCAP 6             // tables mapped (written by the CPU)
#define C_BCAP 7             // bricks mapped (written by the CPU)
#define C_ARGS 8             // 8..10: indirect dispatch arguments
#define C_STEPS 11           // render: DDA steps, when counting
#define C_PICK 12            // 12..15: the cell under the crosshair and its face (0 = nothing)
// physics (one step: see "Physics" below)
#define C_L0 20              // bricks in awake list 0
#define C_L1 21              // bricks in awake list 1
#define C_N 22               // bricks that just woke (their storage is checked at the start of the next step)
#define C_S 23               // those bricks and their +x/+y/+z neighbours, once each
#define C_CH 24              // uniform chunks those need a table in
#define C_W 25               // bricks edits changed since the last step
#define A_N 26               // 26..46: indirect dispatch arguments, three words per dispatch
#define A_S 29
#define A_CH 32
#define A_SB 35
#define A_L 38
#define A_W 41
#define A_LC 44
#define W_MAXL (1u << 20)    // awake bricks at most (512 M cells)
#define W_MAXN (1u << 18)
#define W_MAXS (1u << 21)
#define W_MAXC (1u << 16)
#define W_MAXW (1u << 16)
// water flow (see "Water" below)
#define C_FBUMP 48           // flow slots handed out from the end
#define C_FFREE 49           // flow slots on the free stack
#define C_FCAP 50            // flow slots there are (written by the CPU)
#define W_MAXF (1u << 18)    // bricks with flowing water at most

namespace wld {

// Materials. A cell's byte is its code: codes 0..127 are materials that aren't liquids, codes 128..255 cells of
// liquid. Liquids come in fill levels (1/LIQ_L of a cell each), the finest that lets every liquid in the table share
// the 128 liquid codes: 1 liquid gets 128 levels, 2 get 64, 3-4 get 32, 5-8 get 16, 9-16 get 8. Liquid i holding
// u levels (1..LIQ_L) is code 128 + i * LIQ_L + u - 1. Keep the list in step with World.swift's names.
enum : uint {
  AIR, ROCK, SAND, PLANT, SNOW, STONE, BASALT, WALL,
  CLAY, DUST, ASH, SAWDUST, SALT, SHARDS, SCRAP, RUBBLE, NUGGETS, CRYSTAL_DUST, COAL_DUST,
  WOOD, PLANKS, METAL, BRICK, COAL, SANDSTONE, LIMESTONE, TITANIUM, TUNGSTEN, GOLD,
  WATER, SALTWATER, OIL, MERCURY, MUD, SLIME, BLOOD, TOXIC,   // the liquids, last: liquid i is material WATER + i
  NMAT
};
enum : uint {
  NLIQ = NMAT - WATER,
  LIQ_SHIFT = NLIQ <= 1 ? 7 : NLIQ <= 2 ? 6 : NLIQ <= 4 ? 5 : NLIQ <= 8 ? 4 : NLIQ <= 16 ? 3 : NLIQ <= 32 ? 2 : 1,
  LIQ_L = 1u << LIQ_SHIFT,   // fill levels in a cell of liquid
};
static_assert(NMAT <= 128 && NLIQ <= 64, "materials fit the codes below 128; liquids fit 128 codes at 2+ levels each");
enum : uchar { ST_STATIC = 0, ST_GAS = 1, ST_POWDER = 2, ST_LIQUID = 3 };

// One row per material, from Powder Toy 3D's element table (src/elements.js): colour, per-grain colour variation,
// state, whether light passes through it, density in tenths (elements.js dens x 10; anything heavier sinks under
// anything lighter, static ones never move).
struct Mat { uint hex; float var; uchar state, clear; ushort dens; };
constant Mat MATS[NMAT] = {
  { 0x000000, 0.00f, ST_GAS,    1,    10 },   // air
  { 0x6a6560, 0.12f, ST_STATIC, 0, 65535 },   // rock
  { 0xdcbc74, 0.22f, ST_POWDER, 0,   160 },   // sand
  { 0x3da236, 0.25f, ST_STATIC, 0, 65535 },   // plant
  { 0xeef4ff, 0.06f, ST_POWDER, 0,    90 },   // snow
  { 0x78726c, 0.18f, ST_POWDER, 0,   260 },   // stone
  { 0x353331, 0.10f, ST_STATIC, 0, 65535 },   // basalt
  { 0x59606e, 0.05f, ST_STATIC, 0, 65535 },   // wall
  { 0xd9d2c3, 0.10f, ST_POWDER, 0,   120 },   // clay
  { 0xefdcae, 0.12f, ST_POWDER, 0,    55 },   // dust (floats on water)
  { 0x9b968d, 0.20f, ST_POWDER, 0,    40 },   // ash (floats)
  { 0xc99a62, 0.20f, ST_POWDER, 0,    40 },   // sawdust (floats)
  { 0xf0eee8, 0.08f, ST_POWDER, 0,   130 },   // salt
  { 0xb9dbe3, 0.15f, ST_POWDER, 0,   250 },   // broken glass
  { 0x8e939c, 0.10f, ST_POWDER, 0,   780 },   // scrap metal
  { 0xb0705a, 0.20f, ST_POWDER, 0,   190 },   // brick rubble
  { 0xe7bd45, 0.15f, ST_POWDER, 0,  1930 },   // gold nuggets
  { 0xc3aee0, 0.15f, ST_POWDER, 0,   320 },   // crystal dust
  { 0x39383c, 0.20f, ST_POWDER, 0,   135 },   // broken coal
  { 0x7a4a26, 0.12f, ST_STATIC, 0, 65535 },   // wood
  { 0xa8794a, 0.12f, ST_STATIC, 0, 65535 },   // planks
  { 0xa9afba, 0.04f, ST_STATIC, 0, 65535 },   // metal
  { 0xa4553d, 0.10f, ST_STATIC, 0, 65535 },   // brick
  { 0x2c2b2f, 0.10f, ST_STATIC, 0, 65535 },   // coal
  { 0xc39a64, 0.12f, ST_STATIC, 0, 65535 },   // sandstone
  { 0xc9c3b2, 0.08f, ST_STATIC, 0, 65535 },   // limestone
  { 0x9a9ca3, 0.03f, ST_STATIC, 0, 65535 },   // titanium
  { 0x7b7d82, 0.03f, ST_STATIC, 0, 65535 },   // tungsten
  { 0xdcad2c, 0.04f, ST_STATIC, 0, 65535 },   // gold
  { 0x2a78d4, 0.00f, ST_LIQUID, 1,   100 },   // water
  { 0x3f8ccc, 0.00f, ST_LIQUID, 1,   120 },   // salt water (sinks under fresh water)
  { 0x5a3c12, 0.04f, ST_LIQUID, 0,    80 },   // oil (floats on water)
  { 0xb7b9bd, 0.03f, ST_LIQUID, 0,  1350 },   // mercury
  { 0x9a8f7b, 0.08f, ST_LIQUID, 0,   174 },   // mud
  { 0xb45cae, 0.05f, ST_LIQUID, 0,   105 },   // slime
  { 0x8a0f12, 0.05f, ST_LIQUID, 0,   106 },   // blood
  { 0x3fb52c, 0.05f, ST_LIQUID, 0,   120 },   // toxic sludge
};

inline float3 matColor(uint m) {
  uint h = MATS[min(m, uint(NMAT) - 1u)].hex;
  return pow(float3((h >> 16) & 255u, (h >> 8) & 255u, h & 255u) / 255.0f, 2.2f);
}
inline bool isLiquid(uint m) { return m >= 128u; }
inline uint liqOf(uint m) { return (m - 128u) >> LIQ_SHIFT; }                        // a liquid code's liquid
inline uint units(uint m) { return m >= 128u ? ((m - 128u) & (LIQ_L - 1u)) + 1u : 0u; }   // its fill levels (0: not liquid)
inline uint liqCode(uint liq, uint u) { return u == 0u ? uint(AIR) : 128u + (liq << LIQ_SHIFT) + min(u, uint(LIQ_L)) - 1u; }
inline uint matOf(uint m) { return m >= 128u ? uint(WATER) + liqOf(m) : m; }        // a code's material
inline uint cellCode(uint mat) { return mat >= uint(WATER) ? liqCode(mat - uint(WATER), LIQ_L) : mat; }   // a full cell of it
inline bool isClear(uint m) { return MATS[matOf(m)].clear != 0; }                    // light passes (air, water)
inline bool solid(uint m) { return m != AIR && !isLiquid(m); }

inline uint chunkIdx(int3 c) { return (uint(c.z) * uint(W_CX) + uint(c.x)) * uint(W_CY) + uint(c.y); }
inline int3 chunkOf(uint ci) { uint cy = ci % uint(W_CY), r = ci / uint(W_CY); return int3(r % uint(W_CX), cy, r / uint(W_CX)); }
inline uint loc(int3 l) { return (uint(l.y) * 8u + uint(l.z)) * 8u + uint(l.x); }    // within a table or a brick
inline int3 unloc(uint b) { return int3(b & 7u, b >> 6, (b >> 3) & 7u); }

inline uint cellAt(int3 c, device const uint* grid, device const uint* tables, device const uchar* bricks) {
  uint ce = grid[chunkIdx(c >> 6)];
  if (ce & W_UNI) return ce & 255u;
  uint be = tables[ulong(ce) * 512ul + loc((c >> 3) & 7)];
  if (be & W_UNI) return be & 255u;
  return bricks[ulong(be) * 512ul + loc(c & 7)];
}
inline bool inWorld(int3 c) { return all(c >= 0) && c.x < NW && c.y < NH && c.z < NW; }

// --- noise
inline uint hash3(int x, int y, int z, uint seed) {
  uint h = uint(x) * 0x8da6b343u ^ uint(y) * 0xd8163841u ^ uint(z) * 0xcb1ab31fu ^ seed * 0x9e3779b9u;
  h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16; return h;
}
inline float hval(uint h) { return float(h >> 8) * (2.0f / 16777216.0f) - 1.0f; }
inline float vnoise2(float2 p, uint seed) {
  float2 i = floor(p), f = p - i, u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
  int x = int(i.x), y = int(i.y);
  float a = hval(hash3(x, y, 0, seed)), b = hval(hash3(x + 1, y, 0, seed));
  float c = hval(hash3(x, y + 1, 0, seed)), d = hval(hash3(x + 1, y + 1, 0, seed));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
inline float2 rot(float2 p) { return float2(1.6f * p.x - 1.2f * p.y, 1.2f * p.x + 1.6f * p.y); }   // x2, turned 37 deg
inline float fbm2(float2 p, int oct, uint seed) {
  float s = 0, a = 0.5f, n = 0;
  for (int i = 0; i < oct; i++) { s += a * vnoise2(p, seed + uint(i) * 7919u); n += a; p = rot(p); a *= 0.5f; }
  return s / n;
}
inline float ridged2(float2 p, int oct, uint seed) {   // sharp crests: 1 at the noise's zero lines
  float s = 0, a = 0.5f, n = 0, w = 1;
  for (int i = 0; i < oct; i++) {
    float r = 1.0f - abs(vnoise2(p, seed + uint(i) * 104729u)); r *= r;
    s += a * r * w; n += a; w = saturate(r * 1.5f); p = rot(p); a *= 0.5f;
  }
  return s / n;
}

// Integer cave noise, exactly reproducible in every kernel. Value noise on a 64-cell grid (32 cells vertically:
// tunnels run mostly level), sampled on a lattice every 4 cells; cells interpolate the lattice linearly.
constant int FADE16[16] = { 0, 9, 66, 200, 424, 737, 1127, 1573, 2048, 2523, 2969, 3359, 3672, 3896, 4030, 4087 };
constant int FADE8[8] = { 0, 66, 424, 1127, 2048, 2969, 3672, 4030 };
inline int ilerp(int a, int b, int w, int sh) { return a + (((b - a) * w) >> sh); }   // stays within [a, b]
inline int hv(int x, int y, int z, uint seed) { return int(hash3(x, y, z, seed) >> 16) - 32768; }
inline int caveNoise(int3 L, uint seed) {
  int3 g = int3(L.x >> 4, L.y >> 3, L.z >> 4);
  int wx = FADE16[L.x & 15], wy = FADE8[L.y & 7], wz = FADE16[L.z & 15];
  int a = ilerp(hv(g.x, g.y, g.z, seed), hv(g.x + 1, g.y, g.z, seed), wx, 12);
  int b = ilerp(hv(g.x, g.y + 1, g.z, seed), hv(g.x + 1, g.y + 1, g.z, seed), wx, 12);
  int c = ilerp(hv(g.x, g.y, g.z + 1, seed), hv(g.x + 1, g.y, g.z + 1, seed), wx, 12);
  int d = ilerp(hv(g.x, g.y + 1, g.z + 1, seed), hv(g.x + 1, g.y + 1, g.z + 1, seed), wx, 12);
  return ilerp(ilerp(a, b, wy, 12), ilerp(c, d, wy, 12), wz, 12);
}
#define SEED1 0x51u
#define SEED2 0xa7u
// A value at local cell l (0..7) of a brick, from its 3x3x3 lattice values v[(i.y*3 + i.z)*3 + i.x].
inline int latInterp(thread const int* v, int3 l) {
  int3 i = l >> 2, f = l & 3;
  int b = (i.y * 3 + i.z) * 3 + i.x;
  int a0 = ilerp(v[b], v[b + 1], f.x, 2), a1 = ilerp(v[b + 3], v[b + 4], f.x, 2);
  int a2 = ilerp(v[b + 9], v[b + 10], f.x, 2), a3 = ilerp(v[b + 12], v[b + 13], f.x, 2);
  return ilerp(ilerp(a0, a1, f.z, 2), ilerp(a2, a3, f.z, 2), f.y, 2);
}
inline bool isCave(int n1, int n2) {
  return abs(n1) < W_CAVE_T && abs(n2) < W_CAVE_T && n1 * n1 + n2 * n2 < W_CAVE_T * W_CAVE_T;
}
inline bool caveZone(int y, int h) { return y >= W_SEA + 4 && y < h && h - 1 - y <= W_CAVE_D; }

// One column: height (bits 0-15), surface material (16-23) and its depth in cells (24-31).
inline uint material(int y, uint col, bool cave) {
  int h = int(col & 0xffffu);
  if (y >= h) return y < W_SEA ? cellCode(WATER) : AIR;
  if (cave) return AIR;
  if (h - 1 - y < int(col >> 24)) return (col >> 16) & 255u;
  return ROCK;
}

// A box of cells [y0, y1) over columns whose heights lie in [hmin, hmax]: its material if it is certainly uniform,
// else 0 ("look closer").
inline uint classify(int y0, int y1, int hmin, int hmax, bool caveFree) {
  if (y0 >= max(hmax, W_SEA)) return W_UNI | AIR;
  if (y0 >= hmax && y1 <= W_SEA) return W_UNI | cellCode(WATER);
  if (y1 <= hmin - W_TOPMAX && caveFree) return W_UNI | ROCK;
  return 0;
}
inline bool caveReach(int y0, int y1, int hmin, int hmax) {   // could any cell of the box be in the cave zone?
  return y1 - 1 >= W_SEA + 4 && y0 < hmax && y1 >= hmin - W_CAVE_D;
}

// --- pools: pop from the free stack, else take a fresh one from the end; W_NONE when the mapped pool is full
inline uint alloc(device atomic_int* ctr, int bump, int freeTop, int cap, device const uint* freeList) {
  int s = atomic_fetch_sub_explicit(&ctr[freeTop], 1, memory_order_relaxed);
  if (s > 0) return freeList[s - 1];
  int i = atomic_fetch_add_explicit(&ctr[bump], 1, memory_order_relaxed);
  if (i >= atomic_load_explicit(&ctr[cap], memory_order_relaxed)) {
    atomic_store_explicit(&ctr[C_OVER], 1, memory_order_relaxed);
    return W_NONE;
  }
  return uint(i);
}
// A brick's coordinates packed in one word (x 11 bits, y 7, z 11), and its bit in the awake bitmap.
inline uint packB(int3 b) { return uint(b.x) | (uint(b.y) << 11) | (uint(b.z) << 18); }
inline int3 unpackB(uint v) { return int3(v & 2047u, (v >> 11) & 127u, v >> 18); }
inline bool brickInWorld(int3 b) { return all(b >= 0) && b.x < W_BX && b.y < NH / 8 && b.z < W_BX; }
inline uint bitOf(int3 b) { return (uint(b.z) * uint(W_BX) + uint(b.x)) * uint(NH / 8) + uint(b.y); }
// Adds b to a list once: the first thread to set its bit appends it.
inline void insertOnce(int3 b, device atomic_uint* bits, device atomic_int* ctr, int count, device uint* list, uint cap) {
  uint i = bitOf(b), m = 1u << (i & 31u);
  if (atomic_fetch_or_explicit(&bits[i >> 5], m, memory_order_relaxed) & m) return;
  int j = atomic_fetch_add_explicit(&ctr[count], 1, memory_order_relaxed);
  if (uint(j) < cap) list[j] = packB(b);
}
// Adds b to the next awake list once; if it wasn't awake this step, also to the list of newly awake bricks.
inline void insertNext(int3 b, device atomic_uint* bitsNext, device const uint* bitsCur, device atomic_int* ctr, int count,
                       device uint* next, device uint* fresh) {
  uint i = bitOf(b), m = 1u << (i & 31u);
  if (atomic_fetch_or_explicit(&bitsNext[i >> 5], m, memory_order_relaxed) & m) return;
  int j = atomic_fetch_add_explicit(&ctr[count], 1, memory_order_relaxed);
  if (uint(j) < W_MAXL) next[j] = packB(b);
  if (bitsCur[i >> 5] & m) return;
  int k = atomic_fetch_add_explicit(&ctr[C_N], 1, memory_order_relaxed);
  if (uint(k) < W_MAXN) fresh[k] = packB(b);
}
inline void wakeAppend(device atomic_int* ctr, device uint* wake, int3 b) {
  int j = atomic_fetch_add_explicit(&ctr[C_W], 1, memory_order_relaxed);
  if (uint(j) < W_MAXW) wake[j] = packB(b);
}

inline void release(device atomic_int* ctr, int freeTop, device uint* freeList, uint idx) {
  int s = atomic_fetch_add_explicit(&ctr[freeTop], 1, memory_order_relaxed);
  freeList[s] = idx;
}

// A table whose 512 bricks are all the same uniform material becomes a uniform chunk again.
inline void tryCollapseTable(uint t, uint ci, device uint* grid, device uint* tables, device uint* owner,
                             device atomic_int* ctr, device uint* tableFree) {
  ulong base = ulong(t) * 512ul;
  uint e0 = tables[base];
  if (!(e0 & W_UNI)) return;
  for (uint i = 1; i < 512u; i++) if (tables[base + i] != e0) return;
  grid[ci] = e0; owner[t] = W_NONE;
  release(ctr, C_TFREE, tableFree, t);
}

} // namespace wld
using namespace wld;

// ---------------------------------------------------------------------------------------------------------------
// Generation: an island of rock under grass, sand beaches and snowy peaks, with caves, in an endless sea.

kernel void w_height(device uint* col [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= uint(NW) || g.y >= uint(NW)) return;
  float2 q = (float2(g) + 0.5f) / float(NW) - 0.5f;                       // -0.5 .. 0.5 across the world
  float2 r = q + 0.07f * float2(fbm2(q * 3.0f + 11.3f, 4, 1u), fbm2(q * 3.0f - 7.1f, 4, 2u));   // warped
  float coast = length(r) / 0.42f + 0.18f * fbm2(q * 9.0f, 4, 3u);
  float island = 1.0f - smoothstep(0.55f, 1.0f, coast);
  // hills and ranges have a fixed size in cells, so a bigger world has more of them, not wider ones
  float2 rc = r * float(NW);
  float cont = fbm2(rc / 1400.0f, 6, 4u);
  float mount = smoothstep(-0.05f, 0.45f, cont) * island * island;
  float ridge = ridged2(rc / 1000.0f + 3.0f, 7, 5u);
  float detail = fbm2(float2(g) / 40.0f, 6, 6u);
  float h = float(W_SEA) - 60.0f + island * (75.0f + 110.0f * cont) + mount * ridge * 560.0f
          + detail * (3.0f + 18.0f * mount);
  col[g.y * uint(NW) + g.x] = uint(clamp(h, 4.0f, float(NH - 48)));
}

// Surface layers from height and slope: sand on beaches and the sea floor, snow on high gentle ground, grass on
// gentle ground below that, bare rock on cliffs.
kernel void w_surface(device uint* col [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= uint(NW) || g.y >= uint(NW)) return;
  int x = int(g.x), z = int(g.y);
  auto H = [&](int xx, int zz) { return float(col[uint(clamp(zz, 0, NW - 1)) * uint(NW) + uint(clamp(xx, 0, NW - 1))] & 0xffffu); };
  float h = H(x, z);
  float slope = max(abs(H(x + 2, z) - H(x - 2, z)), abs(H(x, z + 2) - H(x, z - 2))) * 0.25f;
  float n = fbm2(float2(g) / 64.0f, 3, 9u);
  uint top = ROCK, depth = 0;
  if (h <= float(W_SEA) + 2.0f + 3.0f * n) { top = SAND; depth = 5; }
  else if (h >= 620.0f + 50.0f * n && slope < 1.6f) { top = SNOW; depth = 3; }
  else if (slope < 0.9f + 0.3f * n && h < 600.0f) { top = PLANT; depth = 1; }
  else if (slope < 1.4f) { top = STONE; depth = 2; }
  col[g.y * uint(NW) + g.x] = uint(h) | (top << 16) | (depth << 24);
}

// Height range of each 8x8-column brick column.
kernel void w_bands(device const uint* col [[buffer(0)]], device uint* bcol [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= uint(W_BX) || g.y >= uint(W_BX)) return;
  uint lo = 0xffffu, hi = 0;
  for (uint z = 0; z < 8u; z++) for (uint x = 0; x < 8u; x++) {
    uint h = col[(g.y * 8u + z) * uint(NW) + g.x * 8u + x] & 0xffffu;
    lo = min(lo, h); hi = max(hi, h);
  }
  bcol[g.y * uint(W_BX) + g.x] = lo | (hi << 16);
}

// Each chunk: uniform, or a table. mode 0 counts tables; mode 1 hands them out and writes the grid.
kernel void w_chunks(device uint* grid [[buffer(0)]], device const uint* bcol [[buffer(1)]], device atomic_int* ctr [[buffer(2)]],
                     device uint* owner [[buffer(3)]], constant uint& mode [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
  if (g.x >= uint(W_CX) || g.y >= uint(W_CY) || g.z >= uint(W_CX)) return;
  int lo = 0xffff, hi = 0;
  for (uint z = 0; z < 8u; z++) for (uint x = 0; x < 8u; x++) {
    uint b = bcol[(g.z * 8u + z) * uint(W_BX) + g.x * 8u + x];
    lo = min(lo, int(b & 0xffffu)); hi = max(hi, int(b >> 16));
  }
  int y0 = int(g.y) * 64, y1 = y0 + 64;
  uint ci = chunkIdx(int3(g));
  uint u = classify(y0, y1, lo, hi, !caveReach(y0, y1, lo, hi));
  if (u) { if (mode == 1u) grid[ci] = u; return; }
  int t = atomic_fetch_add_explicit(&ctr[C_TBUMP], 1, memory_order_relaxed);
  if (mode == 1u) { grid[ci] = uint(t); owner[t] = ci; }
}

// Each brick of tables [range.x, range.x + range.y): uniform (written now), or a candidate for w_brickFill.
kernel void w_brickClass(device uint* tables [[buffer(0)]], device const uint* owner [[buffer(1)]], device const uint* bcol [[buffer(2)]],
                         device uint* cand [[buffer(3)]], device atomic_int* ctr [[buffer(4)]], constant uint2& range [[buffer(5)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid >= range.y * 512u) return;
  uint t = range.x + gid / 512u, b = gid % 512u;
  int3 bc = chunkOf(owner[t]) * 8 + unloc(b);
  uint band = bcol[uint(bc.z) * uint(W_BX) + uint(bc.x)];
  int lo = int(band & 0xffffu), hi = int(band >> 16), y0 = bc.y * 8, y1 = y0 + 8;
  bool caveFree = true;
  if (caveReach(y0, y1, lo, hi)) {
    // the lattice values bound the interpolated ones: if either noise stays outside (-T, T), no cell is a cave
    int lo1 = 1 << 30, hi1 = -(1 << 30), lo2 = lo1, hi2 = hi1;
    for (int i = 0; i < 27; i++) {
      int3 L = bc * 2 + int3(i % 3, i / 9, (i / 3) % 3);
      int a = caveNoise(L, SEED1), c = caveNoise(L, SEED2);
      lo1 = min(lo1, a); hi1 = max(hi1, a); lo2 = min(lo2, c); hi2 = max(hi2, c);
    }
    caveFree = lo1 >= W_CAVE_T || hi1 <= -W_CAVE_T || lo2 >= W_CAVE_T || hi2 <= -W_CAVE_T;
  }
  uint u = classify(y0, y1, lo, hi, caveFree);
  if (u) { tables[ulong(t) * 512ul + b] = u; return; }
  int i = atomic_fetch_add_explicit(&ctr[C_CAND], 1, memory_order_relaxed);
  cand[i] = t * 512u + b;
}

kernel void w_candArgs(device int* ctr [[buffer(0)]]) {
  ctr[C_ARGS] = ctr[C_CAND]; ctr[C_ARGS + 1] = 1; ctr[C_ARGS + 2] = 1;
}

// One candidate brick per 128-thread group, 4 cells per thread: evaluate its 512 cells; store them if they differ.
kernel void w_brickFill(device uint* tables [[buffer(0)]], device uint* bricks [[buffer(1)]], device const uint* owner [[buffer(2)]],
                        device const uint* cand [[buffer(3)]], device const uint* col [[buffer(4)]], device atomic_int* ctr [[buffer(5)]],
                        uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint cols[64];
  threadgroup int lat1[27], lat2[27];
  threadgroup uint first, idxS;
  threadgroup atomic_uint mixed;
  uint code = cand[tg], tab = code >> 9, b = code & 511u;
  int3 bc = chunkOf(owner[tab]) * 8 + unloc(b);
  if (t < 64u) cols[t] = col[(uint(bc.z) * 8u + (t >> 3)) * uint(NW) + uint(bc.x) * 8u + (t & 7u)];
  if (t < 27u) {
    int3 L = bc * 2 + int3(t % 3u, t / 9u, (t / 3u) % 3u);
    lat1[t] = caveNoise(L, SEED1); lat2[t] = caveNoise(L, SEED2);
  }
  if (t == 0u) atomic_store_explicit(&mixed, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int v1[27], v2[27];
  for (int i = 0; i < 27; i++) { v1[i] = lat1[i]; v2[i] = lat2[i]; }
  int y = int(t >> 4), z = int((t >> 1) & 7u), x0 = int(t & 1u) * 4, wy = bc.y * 8 + y;
  uint packed = 0;
  for (int i = 0; i < 4; i++) {
    int3 l = int3(x0 + i, y, z);
    uint c = cols[z * 8 + x0 + i];
    bool cave = caveZone(wy, int(c & 0xffffu)) && isCave(latInterp(v1, l), latInterp(v2, l));
    packed |= material(wy, c, cave) << (8 * i);
  }
  if (t == 0u) first = packed & 255u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (packed != first * 0x01010101u) atomic_store_explicit(&mixed, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong te = ulong(tab) * 512ul + b;
  if (atomic_load_explicit(&mixed, memory_order_relaxed) == 0u) { if (t == 0u) tables[te] = W_UNI | first; return; }
  if (t == 0u) {
    int i = atomic_fetch_add_explicit(&ctr[C_BBUMP], 1, memory_order_relaxed);
    if (i >= atomic_load_explicit(&ctr[C_BCAP], memory_order_relaxed)) { atomic_store_explicit(&ctr[C_OVER], 1, memory_order_relaxed); idxS = W_NONE; }
    else { idxS = uint(i); tables[te] = uint(i); }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (idxS != W_NONE) bricks[ulong(idxS) * 128ul + t] = packed;
}

// Tables [range.x, range.x + range.y) that came out all one material become uniform chunks; their tables are freed.
kernel void w_collapse(device uint* grid [[buffer(0)]], device uint* tables [[buffer(1)]], device uint* owner [[buffer(2)]],
                       device atomic_int* ctr [[buffer(3)]], device uint* tableFree [[buffer(4)]], constant uint2& range [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid >= range.y) return;
  uint t = range.x + gid, ci = owner[t];
  if (ci == W_NONE || grid[ci] != t) return;
  tryCollapseTable(t, ci, grid, tables, owner, ctr, tableFree);
}

// ---------------------------------------------------------------------------------------------------------------
// Editing: a sphere of cells dug out (mode 0) or filled with a material where there is air or liquid (mode 1).
// Work is proportional to the sphere, never to the world.

struct WEdit { int4 c; int4 chunkLo; int4 brickLo; int r2; uint mat; uint mode; uint pad; };

inline uint applyEdit(uint old, bool inside, constant WEdit& E) {
  if (!inside) return old;
  if (E.mode == 0u) return AIR;
  return (old == AIR || isLiquid(old)) ? cellCode(E.mat) : old;
}
inline bool boxTouches(int3 lo, int size, constant WEdit& E) {   // does the sphere reach the box [lo, lo + size)?
  int3 q = clamp(E.c.xyz, lo, lo + size - 1) - E.c.xyz;
  return q.x * q.x + q.y * q.y + q.z * q.z <= E.r2;
}

// 1. Uniform chunks the edit would change get a table (all 512 bricks = the chunk's material).
kernel void w_editChunks(device uint* grid [[buffer(0)]], device uint* tables [[buffer(1)]], device uint* owner [[buffer(2)]],
                         device atomic_int* ctr [[buffer(3)]], device const uint* tableFree [[buffer(4)]], constant WEdit& E [[buffer(5)]],
                         device uint* slots [[buffer(6)]], uint3 g [[thread_position_in_grid]]) {
  int3 cc = E.chunkLo.xyz + int3(g);
  if (any(cc < 0) || cc.x >= W_CX || cc.y >= W_CY || cc.z >= W_CX || !boxTouches(cc * 64, 64, E)) return;
  uint ci = chunkIdx(cc), ce = grid[ci];
  if (!(ce & W_UNI) || applyEdit(ce & 255u, true, E) == (ce & 255u)) return;
  uint t = alloc(ctr, C_TBUMP, C_TFREE, C_TCAP, tableFree);
  if (t == W_NONE) return;
  for (uint i = 0; i < 512u; i++) { tables[ulong(t) * 512ul + i] = ce; slots[ulong(t) * 512ul + i] = 0; }
  owner[t] = ci; grid[ci] = t;
}

// Pops can drive a free stack's count below zero (they then took fresh entries instead): put it back at zero.
kernel void w_fixFree(device atomic_int* ctr [[buffer(0)]]) {
  const int freeTop[3] = { C_TFREE, C_BFREE, C_FFREE }, bump[3] = { C_TBUMP, C_BBUMP, C_FBUMP }, cap[3] = { C_TCAP, C_BCAP, C_FCAP };
  for (int k = 0; k < 3; k++) {
    if (atomic_load_explicit(&ctr[freeTop[k]], memory_order_relaxed) < 0) atomic_store_explicit(&ctr[freeTop[k]], 0, memory_order_relaxed);
    int c = atomic_load_explicit(&ctr[cap[k]], memory_order_relaxed);
    if (atomic_load_explicit(&ctr[bump[k]], memory_order_relaxed) > c) atomic_store_explicit(&ctr[bump[k]], c, memory_order_relaxed);
  }
}

// 2. Each brick the sphere touches, one 128-thread group, 4 cells per thread: new cells; uniform bricks that change
//    get storage.
kernel void w_editBricks(device uint* grid [[buffer(0)]], device uint* tables [[buffer(1)]], device uint* bricks [[buffer(2)]],
                         device atomic_int* ctr [[buffer(3)]], device const uint* brickFree [[buffer(4)]], constant WEdit& E [[buffer(5)]],
                         device uint* wake [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint changed;
  threadgroup uint idxS;
  int3 bc = E.brickLo.xyz + int3(tg);
  if (any(bc < 0) || bc.x >= W_BX || bc.y >= NH / 8 || bc.z >= W_BX || !boxTouches(bc * 8, 8, E)) return;
  uint ce = grid[chunkIdx(bc >> 3)];
  if (ce & W_UNI) return;                         // step 1 gave a table to every chunk this edit changes
  ulong te = ulong(ce) * 512ul + loc(bc & 7);
  uint be = tables[te];
  if (t == 0u) atomic_store_explicit(&changed, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint old = (be & W_UNI) ? (be & 255u) * 0x01010101u : bricks[ulong(be) * 128ul + t];
  int y = int(t >> 4), z = int((t >> 1) & 7u), x0 = int(t & 1u) * 4;
  uint packed = 0;
  for (int i = 0; i < 4; i++) {
    int3 d = bc * 8 + int3(x0 + i, y, z) - E.c.xyz;
    packed |= applyEdit((old >> (8 * i)) & 255u, d.x * d.x + d.y * d.y + d.z * d.z <= E.r2, E) << (8 * i);
  }
  if (packed != old) atomic_store_explicit(&changed, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (atomic_load_explicit(&changed, memory_order_relaxed) == 0u) return;
  if (t == 0u) wakeAppend(ctr, wake, bc);           // the physics looks at it next step
  if (!(be & W_UNI)) { if (packed != old) bricks[ulong(be) * 128ul + t] = packed; return; }
  if (t == 0u) { idxS = alloc(ctr, C_BBUMP, C_BFREE, C_BCAP, brickFree); if (idxS != W_NONE) tables[te] = idxS; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (idxS != W_NONE) bricks[ulong(idxS) * 128ul + t] = packed;
}

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

// ---------------------------------------------------------------------------------------------------------------
// Physics. One byte per cell: powders fall and pile at 45 degrees, liquids fall and slide down, anything heavier
// sinks under anything lighter. A liquid with a free surface also flows by its head ("Liquids" below), which levels it.
// No heat, reactions or air.
//
// A step cuts space into 2x2x2 blocks (shifted by one cell on alternate steps, so cells cross block edges) and one
// thread rearranges each block's 8 cells. Every cell belongs to exactly one block, so no two threads touch a cell,
// and cells only swap places, so nothing is ever created or lost. It runs in place on the bricks.
//
// Only awake bricks run. A brick is awake if a cell next to it changed last step, or an edit changed it. A step:
//   storage  bricks that just woke, and their +x/+y/+z neighbours (a block owned by a brick reaches at most one cell
//            into those), get a chunk table and brick storage if they are still one material. Nothing is collapsed
//            during play, so a brick that stays awake costs nothing here.
//   move     one 64-thread group per awake brick, one thread per block it owns; each changed cell wakes the bricks
//            holding it and its 26 neighbouring cells for the next step
//   wake     every brick an edit changed, and its 26 neighbours, join the next step
// The awake list and its bitmap come in two copies, this step's and the next, which swap each step.

inline uint stateOf(uint m) { return MATS[matOf(m)].state; }
inline bool movable(uint m) { return stateOf(m) != ST_STATIC; }
inline bool sinksInto(uint a, uint b) { return movable(a) && movable(b) && MATS[matOf(a)].dens > MATS[matOf(b)].dens; }   // a above b swaps

// The block's cells c[x + 2z + 4y]: 0-3 the bottom layer, 4-7 the top. True if anything moved.
inline bool stepBlock(thread uint* c, uint r) {
  uint done = 0;   // bit k: cell k already moved this step
  bool any = false;
  // fall: each column, top into bottom
  for (uint k = 0; k < 4u; k++) if (sinksInto(c[k + 4], c[k])) {
    uint t = c[k]; c[k] = c[k + 4]; c[k + 4] = t; done |= (1u << k) | (1u << (k + 4)); any = true;
  }
  // liquid on a part-filled cell of the same liquid pours its levels into it
  for (uint k = 0; k < 4u; k++) {
    uint top = c[k + 4], bot = c[k];
    if (!isLiquid(top) || !isLiquid(bot) || liqOf(top) != liqOf(bot) || units(bot) >= LIQ_L) continue;
    uint ut = units(top), ub = units(bot), m = min(ut, LIQ_L - ub), l = liqOf(top);
    c[k] = liqCode(l, ub + m); c[k + 4] = liqCode(l, ut - m); done |= (1u << k) | (1u << (k + 4)); any = true;
  }
  // slide: powders and liquids that couldn't fall go to a lower cell beside them (x, z or diagonal), random order
  for (uint i = 0; i < 4u; i++) {
    uint k = (i + r) & 3u, a = c[k + 4];
    if ((done >> (k + 4)) & 1u || stateOf(a) < ST_POWDER) continue;
    uint s = (r >> (2 + 2 * i)) % 3u;
    for (uint j = 0; j < 3u; j++) {
      uint b = k ^ (1u + (s + j) % 3u);
      if ((done >> b) & 1u || !sinksInto(a, c[b])) continue;
      c[k + 4] = c[b]; c[b] = a; done |= (1u << (k + 4)) | (1u << b); any = true;
      break;
    }
  }
  return any;
}

// Indirect arguments for a list: ceil(count / per) threadgroups, the count capped at the list's capacity; then up to
// four counters set to zero (their lists are used up; ~0 = none).
struct ArgSpec { uint4 a; uint4 zero; };   // a: count index, per, argument index, capacity
kernel void w_args(device int* ctr [[buffer(0)]], constant ArgSpec& A [[buffer(1)]]) {
  uint n = min(uint(max(ctr[A.a.x], 0)), A.a.w);
  ctr[A.a.z] = int((n + A.a.y - 1) / A.a.y); ctr[A.a.z + 1] = 1; ctr[A.a.z + 2] = 1;
  for (int i = 0; i < 4; i++) if (A.zero[i] != ~0u) ctr[A.zero[i]] = 0;
}

// Each newly awake brick and its 7 neighbours in +x/+y/+z, once each (through the idle bitmap, which is clear)
kernel void w_storageList(device const uint* N [[buffer(0)]], device atomic_int* ctr [[buffer(1)]], device atomic_uint* bits [[buffer(2)]],
                          device uint* S [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= min(uint(atomic_load_explicit(&ctr[C_N], memory_order_relaxed)), W_MAXN)) return;
  int3 b = unpackB(N[gid]);
  for (int i = 0; i < 8; i++) {
    int3 n = b + int3(i & 1, i >> 2, (i >> 1) & 1);
    if (brickInWorld(n)) insertOnce(n, bits, ctr, C_S, S, W_MAXS);
  }
}

kernel void w_clearBits(device const uint* list [[buffer(0)]], device atomic_int* ctr [[buffer(1)]], device atomic_uint* bits [[buffer(2)]],
                        constant uint2& countCap [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= min(uint(atomic_load_explicit(&ctr[countCap.x], memory_order_relaxed)), countCap.y)) return;
  uint i = bitOf(unpackB(list[gid]));
  atomic_fetch_and_explicit(&bits[i >> 5], ~(1u << (i & 31u)), memory_order_relaxed);
}

// End of a step: this step's bitmap back to clear; a brick that isn't awake next step gives back its flow slot.
kernel void w_retire(device const uint* list [[buffer(0)]], device atomic_int* ctr [[buffer(1)]], device atomic_uint* bitsCur [[buffer(2)]],
                     constant uint2& countCap [[buffer(3)]], device const uint* bitsNext [[buffer(4)]], device const uint* grid [[buffer(5)]],
                     device uint* slots [[buffer(6)]], device uint* flowFree [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= min(uint(atomic_load_explicit(&ctr[countCap.x], memory_order_relaxed)), countCap.y)) return;
  int3 b = unpackB(list[gid]);
  uint i = bitOf(b), m = 1u << (i & 31u);
  atomic_fetch_and_explicit(&bitsCur[i >> 5], ~m, memory_order_relaxed);
  if (bitsNext[i >> 5] & m) return;
  uint ce = grid[chunkIdx(b >> 3)];
  if (ce & W_UNI) return;
  ulong se = ulong(ce) * 512ul + loc(b & 7);
  uint sl = slots[se];
  if (sl == 0u) return;
  slots[se] = 0;
  release(ctr, C_FFREE, flowFree, sl - 1u);
}

// The uniform chunks that bricks of S lie in, once each (cbits: one bit per chunk)
kernel void w_storageChunks(device const uint* S [[buffer(0)]], device atomic_int* ctr [[buffer(1)]], device const uint* grid [[buffer(2)]],
                            device atomic_uint* cbits [[buffer(3)]], device uint* CL [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= min(uint(atomic_load_explicit(&ctr[C_S], memory_order_relaxed)), W_MAXS)) return;
  uint ci = chunkIdx(unpackB(S[gid]) >> 3);
  if (!(grid[ci] & W_UNI)) return;
  uint m = 1u << (ci & 31u);
  if (atomic_fetch_or_explicit(&cbits[ci >> 5], m, memory_order_relaxed) & m) return;
  int j = atomic_fetch_add_explicit(&ctr[C_CH], 1, memory_order_relaxed);
  if (uint(j) < W_MAXC) CL[j] = ci;
}

// Each of those chunks gets a table of 512 copies of its material (and no flow slots).
kernel void w_expandChunks(device uint* grid [[buffer(0)]], device uint* tables [[buffer(1)]], device uint* owner [[buffer(2)]],
                           device atomic_int* ctr [[buffer(3)]], device const uint* tableFree [[buffer(4)]], device const uint* CL [[buffer(5)]],
                           device atomic_uint* cbits [[buffer(6)]], device uint* slots [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= min(uint(atomic_load_explicit(&ctr[C_CH], memory_order_relaxed)), W_MAXC)) return;
  uint ci = CL[gid], ce = grid[ci];
  atomic_fetch_and_explicit(&cbits[ci >> 5], ~(1u << (ci & 31u)), memory_order_relaxed);
  if (!(ce & W_UNI)) return;
  uint t = alloc(ctr, C_TBUMP, C_TFREE, C_TCAP, tableFree);
  if (t == W_NONE) return;
  for (uint i = 0; i < 512u; i++) { tables[ulong(t) * 512ul + i] = ce; slots[ulong(t) * 512ul + i] = 0; }
  owner[t] = ci; grid[ci] = t;
}

// Each brick of S still stored as one material gets 512 bytes of it. One 64-thread group per brick.
kernel void w_expandBricks(device const uint* grid [[buffer(0)]], device uint* tables [[buffer(1)]], device uint* bricks [[buffer(2)]],
                           device atomic_int* ctr [[buffer(3)]], device const uint* brickFree [[buffer(4)]], device const uint* S [[buffer(5)]],
                           uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint idxS, fill;
  if (tg >= min(uint(atomic_load_explicit(&ctr[C_S], memory_order_relaxed)), W_MAXS)) return;
  int3 b = unpackB(S[tg]);
  uint ce = grid[chunkIdx(b >> 3)];
  if (ce & W_UNI) return;                                  // no table (the pool was full): it stays as it is
  ulong te = ulong(ce) * 512ul + loc(b & 7);
  uint be = tables[te];
  if (!(be & W_UNI)) return;
  if (t == 0u) { idxS = alloc(ctr, C_BBUMP, C_BFREE, C_BCAP, brickFree); fill = (be & 255u) * 0x01010101u; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (idxS == W_NONE) return;
  bricks[ulong(idxS) * 128ul + t] = fill; bricks[ulong(idxS) * 128ul + t + 64u] = fill;
  threadgroup_barrier(mem_flags::mem_device);
  if (t == 0u) tables[te] = idxS;
}

// The step. One 64-thread group per awake brick; thread t owns the block whose low corner is
// brick * 8 + parity + 2 * (t & 3, t >> 4, (t >> 2) & 3). A block reaches at most one cell into the brick's
// +x/+y/+z neighbours; a cell in a brick without storage (the pool was full) counts as a wall.
struct MoveParams { uint cur, parity, step, next; };
kernel void w_move(device const uint* grid [[buffer(0)]], device const uint* tables [[buffer(1)]], device uchar* bricks [[buffer(2)]],
                   device atomic_int* ctr [[buffer(3)]], device const uint* L [[buffer(4)]], device uint* next [[buffer(5)]],
                   device uint* fresh [[buffer(6)]], device atomic_uint* bitsNext [[buffer(7)]], device const uint* bitsCur [[buffer(8)]],
                   constant MoveParams& P [[buffer(9)]], device uchar* wetFlag [[buffer(10)]],
                   uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup ulong base[8];          // where the brick and its forward neighbours start in the pool (~0: no storage)
  threadgroup atomic_uint wakeMask;   // bit (ox+1) + 3(oz+1) + 9(oy+1): the brick at that offset wakes next step
  threadgroup atomic_uint sawWater;   // any liquid among the cells this group read: the liquid pass looks at it
  if (tg >= min(uint(atomic_load_explicit(&ctr[P.cur], memory_order_relaxed)), W_MAXL)) return;
  int3 b = unpackB(L[tg]);
  if (t < 8u) {
    int3 n = b + int3(t & 1u, t >> 2, (t >> 1) & 1u);
    ulong a = ~0ul;
    if (brickInWorld(n)) {
      uint ce = grid[chunkIdx(n >> 3)];
      if (!(ce & W_UNI)) { uint be = tables[ulong(ce) * 512ul + loc(n & 7)]; if (!(be & W_UNI)) a = ulong(be) * 512ul; }
    }
    base[t] = a;
  }
  if (t == 0u) { atomic_store_explicit(&wakeMask, 0u, memory_order_relaxed); atomic_store_explicit(&sawWater, 0u, memory_order_relaxed); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int3 lo = b * 8 + int(P.parity) + 2 * int3(t & 3u, t >> 4, (t >> 2) & 3u);
  ulong addr[8];
  uint c[8], old[8];
  for (uint i = 0; i < 8u; i++) {                      // cell i = x + 2z + 4y of the block
    int3 w = lo + int3(i & 1u, i >> 2, (i >> 1) & 1u);
    int3 o = (w >> 3) - b;
    ulong a = base[o.x + 2 * o.z + 4 * o.y];
    addr[i] = a == ~0ul ? ~0ul : a + loc(w & 7);
    c[i] = addr[i] == ~0ul ? uint(WALL) : uint(bricks[addr[i]]);
    old[i] = c[i];
  }
  bool wet = false;
  for (uint i = 0; i < 8u; i++) wet |= isLiquid(old[i]);
  if (wet) atomic_store_explicit(&sawWater, 1u, memory_order_relaxed);
  if (stepBlock(c, hash3(lo.x, lo.y, lo.z, P.step))) {
    uint mask = 0;
    for (uint i = 0; i < 8u; i++) {
      if (c[i] == old[i]) continue;
      bricks[addr[i]] = uchar(c[i]);
      // the bricks holding this cell and its 26 neighbours
      int3 w = lo + int3(i & 1u, i >> 2, (i >> 1) & 1u);
      int3 o0 = ((w - 1) >> 3) - b, o1 = ((w + 1) >> 3) - b;
      for (int oy = o0.y; oy <= o1.y; oy++) for (int oz = o0.z; oz <= o1.z; oz++) for (int ox = o0.x; ox <= o1.x; ox++)
        mask |= 1u << uint((ox + 1) + 3 * (oz + 1) + 9 * (oy + 1));
    }
    atomic_fetch_or_explicit(&wakeMask, mask, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) wetFlag[tg] = uchar(atomic_load_explicit(&sawWater, memory_order_relaxed));
  if (t < 27u && (atomic_load_explicit(&wakeMask, memory_order_relaxed) >> t) & 1u) {
    int3 n = b + int3(int(t % 3u), int(t / 9u), int((t / 3u) % 3u)) - 1;
    if (brickInWorld(n)) insertNext(n, bitsNext, bitsCur, ctr, int(P.next), next, fresh);
  }
}

// Each brick an edit changed, and its 26 neighbours, join the next step.
kernel void w_wake(device const uint* wake [[buffer(0)]], device atomic_int* ctr [[buffer(1)]], device atomic_uint* bitsNext [[buffer(2)]],
                   device const uint* bitsCur [[buffer(3)]], device uint* next [[buffer(4)]], device uint* fresh [[buffer(5)]],
                   constant uint& nextCount [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= min(uint(atomic_load_explicit(&ctr[C_W], memory_order_relaxed)), W_MAXW)) return;
  int3 b = unpackB(wake[gid]);
  for (int i = 0; i < 27; i++) {
    int3 n = b + int3(i % 3, i / 9, (i / 3) % 3) - 1;
    if (brickInWorld(n)) insertNext(n, bitsNext, bitsCur, ctr, int(nextCount), next, fresh);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Liquids. A still liquid presses with the depth of liquid above it, so it flows from where its surface (its head)
// is higher to where it is lower, with momentum, slowed by friction with the ground: the shallow-water equations on
// the free surface, in "virtual pipe" form between neighbouring columns. Every constant comes from physics:
//   force     moving one unit (one fill level, 1/LIQ_L cell) from a surface at h1 to one at h2 releases energy only
//             if h1 - h2 is more than one unit, so the push is the head difference less one unit, and a liquid rests
//             flat to one unit
//   gravity   G = g dt^2 / dx (cells per step per step) times the liquid's depth: waves run at sqrt(g depth)
//   friction  Manning's law for a natural bed (n = 0.03 s m^-1/3): deceleration g n^2 |u| u / depth^(4/3), applied
//             implicitly (unconditionally stable)
//   time step the CPU splits a physics step into as many sub-steps as keep a wave under one column per sub-step in
//             2D (the explicit scheme's stability limit)
//
// A column is one of a brick's 64 columns of 8 cells. Its head is its surface in units from the brick's floor: the
// top of its highest non-air cell (a liquid cell counts its units). It is wet if that cell is liquid with air above;
// what it can give is the liquid of that kind stacked on top. A cell holds one liquid, so a liquid reaching a column
// topped by another one (or by ground) lands from the next whole cell up: as that liquid sees it, the column's head
// is rounded up to a whole cell. Oil spreads over water this way; heavier under lighter is the move step's job.
// A column owns the edges on its +x and +z sides: their flow (half float) and the units crossing them this sub-step
// with the giver's liquid, 8 bytes per column, in a per-brick flow slot held while the brick is awake.
//
// A sub-step is two passes over the awake bricks, one thread per column:
//   flow   each column updates its two edges from both sides' heads, and sets the units crossing them: no more than
//          the edge's share of what the giver has on top and of the room the taker has. A column's four edges share
//          n units as the counts of i < n with i = edge + sub-step (mod 4): they add up to n, so a column never gives
//          more than it holds or takes more than fits, and the odd units rotate round the edges, so even a film one
//          unit thin flows. Units cross only into a brick that is also in this step's list (else it is woken for the
//          next step). The slot is stamped with the sub-step.
//   apply  each column takes what leaves through its edges off its top, then puts on what arrives, its own liquid
//          first (read only from slots stamped this sub-step), and writes only its own cells. Liquid is conserved
//          exactly, in units, with no atomics. (Two kinds arriving at once leave a part-empty cell between them; if
//          that leaves too little room at the brick's top, the rest fills part-empty cells lower in the column, as
//          their liquid: volume is kept, a few units change kind.)
// An edge with no push and less than a unit of flow is at rest; a brick with nothing flowing sleeps.

#define W_MANNING 0.03f       // Manning roughness of a natural bed, s m^-1/3
#define W_CELL_M 0.3f         // metres per cell
#define W_U int(LIQ_L)        // units in a cell
#define W_S int(LIQ_SHIFT)
#define W_FULL (8 * W_U)      // units in a column
#define W_DRY 0xffu           // no liquid on top

struct WaterParams { uint cur, stamp, next, pad; float G, dt, pad1, pad2; };
// Flow on the +x and +z edges, and what crosses them this sub-step: units + 512 (positive toward +x / +z) in bits
// 0-9, the giver's liquid in bits 10-15.
struct Edge { half2 q; ushort2 a; };
inline ushort packA(int a, uint liq) { return ushort(uint(a + 512) | (liq << 10)); }
inline int unitsA(ushort e) { return int(e & 1023u) - 512; }
inline uint liqA(ushort e) { return uint(e) >> 10; }
static_assert(W_FULL / 4 < 512 && NLIQ <= 64, "an edge's units and liquid fit its 16 bits");

inline int ceilCell(int h) { return (h + W_U - 1) & ~(W_U - 1); }
// Edge j's share of n units this sub-step (j: 0 +x, 1 +z, 2 -x, 3 -z): the shares of the four edges add up to n.
inline int share(int n, int j, uint stamp) { return (n - int((uint(j) + stamp) & 3u) + 3) >> 2; }

// One column's 8 cells (y = 0..7): its head in units, whether it is wet, the liquid on top and how much of it.
struct Col { int head; bool wet; int avail; uint liq; };
inline Col column(device const uchar* bricks, ulong base, int x, int z, uint above) {
  Col c; c.head = 0; c.wet = false; c.avail = 0; c.liq = W_DRY;
  if (base == ~0ul) { c.head = W_FULL; return c; }              // no storage: full and dry
  uint cell[8], nonAir = 0;
  for (int y = 0; y < 8; y++) { cell[y] = bricks[base + loc(int3(x, y, z))]; nonAir |= uint(cell[y] != AIR) << y; }
  if (!nonAir) return c;
  int top = 31 - int(clz(nonAir));
  uint m = cell[top];
  c.head = top * W_U + (isLiquid(m) ? int(units(m)) : W_U);
  if (!isLiquid(m)) return c;
  if (top == 7 && isLiquid(above)) { c.head = W_FULL; return c; }   // the surface is above this brick
  c.wet = true; c.liq = liqOf(m);
  for (int y = top; y >= 0 && isLiquid(cell[y]) && liqOf(cell[y]) == c.liq; y--) c.avail += int(units(cell[y]));
  return c;
}

// Take k units off the top of a column whose head is `head` (k never exceeds what it can give).
inline void takeUnits(device uchar* bricks, ulong base, int x, int z, int head, int k) {
  for (int y = (head - 1) >> W_S; k > 0 && y >= 0; y--) {
    ulong a = base + loc(int3(x, y, z));
    uint m = bricks[a];
    int u = int(units(m)), d = min(u, k);
    bricks[a] = uchar(liqCode(liqOf(m), uint(u - d))); k -= d;
  }
}
// Put k units of liquid `liq` on top (from the next whole cell if the top cell holds another liquid); `head` follows.
// Returns what didn't fit below the brick's top.
inline int putUnits(device uchar* bricks, ulong base, int x, int z, thread int& head, uint liq, int k) {
  int y = head >> W_S, f = head & (W_U - 1);
  if (f && liqOf(bricks[base + loc(int3(x, y, z))]) != liq) { y++; f = 0; }
  for (; k > 0 && y < 8; y++, f = 0) {
    ulong a = base + loc(int3(x, y, z));
    int u = f ? int(units(bricks[a])) : 0, d = min(W_U - u, k);
    bricks[a] = uchar(liqCode(liq, uint(u + d))); k -= d; head = y * W_U + u + d;
  }
  return k;
}
// Units that found no room at the top go into part-empty liquid cells below, as their liquid. Volume is kept.
inline void fillGaps(device uchar* bricks, ulong base, int x, int z, int k) {
  for (int y = 7; k > 0 && y >= 0; y--) {
    ulong a = base + loc(int3(x, y, z));
    uint m = bricks[a];
    if (!isLiquid(m) || units(m) >= LIQ_L) continue;
    int u = int(units(m)), d = min(W_U - u, k);
    bricks[a] = uchar(liqCode(liqOf(m), uint(u + d))); k -= d;
  }
}

// The brick's storage and that of its +x/+y/+z neighbours (~0: none); t < 8 computes entry t.
inline ulong brickBase(int3 n, device const uint* grid, device const uint* tables) {
  if (!brickInWorld(n)) return ~0ul;
  uint ce = grid[chunkIdx(n >> 3)];
  if (ce & W_UNI) return ~0ul;
  uint be = tables[ulong(ce) * 512ul + loc(n & 7)];
  return (be & W_UNI) ? ~0ul : ulong(be) * 512ul;
}
inline uint slotOf(int3 n, device const uint* grid, device const uint* slots) {   // flow slot + 1, or 0
  if (!brickInWorld(n)) return 0u;
  uint ce = grid[chunkIdx(n >> 3)];
  return (ce & W_UNI) ? 0u : slots[ulong(ce) * 512ul + loc(n & 7)];
}
inline uint aboveCell(int3 w, device const uint* grid, device const uint* tables, device const uchar* bricks) {
  return inWorld(w) ? cellAt(w, grid, tables, bricks) : uint(AIR);
}
inline void wakeInsert(uint mask, uint t, int3 b, device atomic_uint* bitsNext, device const uint* bitsCur, device atomic_int* ctr,
                       uint nextCount, device uint* next, device uint* fresh) {
  if (t < 27u && (mask >> t) & 1u) {
    int3 n = b + int3(int(t % 3u), int(t / 9u), int((t / 3u) % 3u)) - 1;
    if (brickInWorld(n)) insertNext(n, bitsNext, bitsCur, ctr, int(nextCount), next, fresh);
  }
}

// Pass 1: each column's +x and +z edges.
kernel void w_waterFlow(device const uint* grid [[buffer(0)]], device const uint* tables [[buffer(1)]], device const uchar* bricks [[buffer(2)]],
                        device atomic_int* ctr [[buffer(3)]], device const uint* L [[buffer(4)]], device uint* next [[buffer(5)]],
                        device uint* fresh [[buffer(6)]], device atomic_uint* bitsNext [[buffer(7)]], device const uint* bitsCur [[buffer(8)]],
                        device uint* slots [[buffer(9)]], device Edge* edges [[buffer(10)]], device const uint* flowFree [[buffer(11)]],
                        constant WaterParams& P [[buffer(12)]], device const uchar* wetFlag [[buffer(13)]], device uint* stamps [[buffer(14)]],
                        uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup ulong base[8];
  threadgroup short headS[64], availS[64];
  threadgroup bool wetS[64];
  threadgroup uchar liqS[64];
  threadgroup uint slotS;
  threadgroup bool newSlot, sendX, sendZ;
  threadgroup atomic_uint anyWet, wakeMask;
  if (tg >= min(uint(atomic_load_explicit(&ctr[P.cur], memory_order_relaxed)), W_MAXL)) return;
  int3 b = unpackB(L[tg]);
  uint ce0 = grid[chunkIdx(b >> 3)];
  if (ce0 & W_UNI) return;
  ulong se = ulong(ce0) * 512ul + loc(b & 7);
  // nothing to do unless this brick holds liquid, still flows, or a +x/+z neighbour does (this brick owns the edge between)
  if (!wetFlag[tg] && slots[se] == 0u && slotOf(b + int3(1, 0, 0), grid, slots) == 0u && slotOf(b + int3(0, 0, 1), grid, slots) == 0u) return;
  if (t < 8u) base[t] = brickBase(b + int3(t & 1u, t >> 2, (t >> 1) & 1u), grid, tables);
  if (t == 0u) {
    atomic_store_explicit(&anyWet, 0u, memory_order_relaxed); atomic_store_explicit(&wakeMask, 0u, memory_order_relaxed);
    int3 bx = b + int3(1, 0, 0), bz = b + int3(0, 0, 1);
    uint ix = bitOf(bx), iz = bitOf(bz);
    sendX = brickInWorld(bx) && ((bitsCur[ix >> 5] >> (ix & 31u)) & 1u);    // the +x brick applies this sub-step too
    sendZ = brickInWorld(bz) && ((bitsCur[iz >> 5] >> (iz & 31u)) & 1u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (base[0] == ~0ul) return;
  int x = int(t & 7u), z = int(t >> 3);
  Col me = column(bricks, base[0], x, z, base[4] != ~0ul ? uint(bricks[base[4] + loc(int3(x, 0, z))]) : aboveCell(b * 8 + int3(x, 8, z), grid, tables, bricks));
  headS[t] = short(me.head); availS[t] = short(me.avail); wetS[t] = me.wet; liqS[t] = uchar(me.liq);
  if (me.wet) atomic_store_explicit(&anyWet, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    uint sl = slots[se];
    newSlot = false;
    if (sl == 0u) {
      uint f = alloc(ctr, C_FBUMP, C_FFREE, C_FCAP, flowFree);
      if (f != W_NONE) { sl = f + 1u; slots[se] = sl; newSlot = true; }
    }
    slotS = sl;
    if (sl) stamps[sl - 1u] = P.stamp;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (slotS == 0u) return;
  device Edge* E = edges + ulong(slotS - 1u) * 64ul;
  Edge ed = newSlot ? Edge{ half2(0), ushort2(packA(0, 0)) } : E[t];
  Col nb[2];
  if (x < 7) { nb[0].head = headS[t + 1]; nb[0].wet = wetS[t + 1]; nb[0].avail = availS[t + 1]; nb[0].liq = liqS[t + 1]; }
  else nb[0] = column(bricks, base[1], 0, z, aboveCell(b * 8 + int3(8, 8, z), grid, tables, bricks));
  if (z < 7) { nb[1].head = headS[t + 8]; nb[1].wet = wetS[t + 8]; nb[1].avail = availS[t + 8]; nb[1].liq = liqS[t + 8]; }
  else nb[1] = column(bricks, base[2], x, 0, aboveCell(b * 8 + int3(x, 8, 8), grid, tables, bricks));
  bool send[2] = { x < 7 || sendX, z < 7 || sendZ };
  uint mask = 0;
  float qs[2] = { float(ed.q.x), float(ed.q.y) };
  int as[2] = { 0, 0 };
  uint ls[2] = { 0, 0 };
  for (int e = 0; e < 2; e++) {
    Col n = nb[e];
    // each side's head as the other side's liquid sees it: another liquid lands from the next whole cell up
    bool same = me.liq == n.liq;
    int hmFor = same ? me.head : ceilCell(me.head), hnFor = same ? n.head : ceilCell(n.head);
    int push = (me.wet ? max(me.head - hnFor - 1, 0) : 0) - (n.wet ? max(n.head - hmFor - 1, 0) : 0);
    float depth = max(float(max(me.wet ? me.avail : 0, n.wet ? n.avail : 0)), 1.0f) / float(W_U);   // cells
    float q = qs[e] + P.G * depth * float(push);
    float u = abs(q) / float(W_U) / depth * W_CELL_M / P.dt, dm = depth * W_CELL_M;   // flow speed (m/s), depth (m)
    q /= 1.0f + P.dt * 9.8f * W_MANNING * W_MANNING * u / pow(dm, 4.0f / 3.0f);
    if ((q > 0 && (!me.wet || hnFor >= W_FULL)) || (q < 0 && (!n.wet || hmFor >= W_FULL))) q = 0;
    if (push == 0 && abs(q) < 1.0f) q = 0;                                    // no unit will ever cross: at rest
    int a = int(q);                                                           // whole units, toward the slower side
    if (a > 0) a = min(a, min(share(me.avail, e, P.stamp), share(W_FULL - hnFor, e + 2, P.stamp)));
    if (a < 0) a = -min(-a, min(share(n.avail, e + 2, P.stamp), share(W_FULL - hmFor, e, P.stamp)));
    if (!send[e]) { a = 0; if (q != 0) mask |= 1u << (e == 0 ? 14 : 16); }     // the other brick isn't running: wake it
    qs[e] = q; as[e] = a; ls[e] = a > 0 ? me.liq : (a < 0 ? n.liq : 0u);
    if (q != 0) mask |= 1u << 13;
    if (q != 0 && (e == 0 ? x == 7 : z == 7)) mask |= 1u << (e == 0 ? 14 : 16);
  }
  E[t] = Edge{ half2(half(qs[0]), half(qs[1])), ushort2(packA(as[0], ls[0]), packA(as[1], ls[1])) };
  // a wet column next to air across the brick's -x or -z side wakes that brick, which owns the edge between them
  if (me.wet) {
    int y = (me.head - 1) >> W_S;
    if (x == 0 && brickInWorld(b - int3(1, 0, 0)) && cellAt(b * 8 + int3(-1, y, z), grid, tables, bricks) == AIR) mask |= 1u << 12;
    if (z == 0 && brickInWorld(b - int3(0, 0, 1)) && cellAt(b * 8 + int3(x, y, -1), grid, tables, bricks) == AIR) mask |= 1u << 10;
  }
  if (mask) atomic_fetch_or_explicit(&wakeMask, mask, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  wakeInsert(atomic_load_explicit(&wakeMask, memory_order_relaxed), t, b, bitsNext, bitsCur, ctr, P.next, next, fresh);
}

// Pass 2: each column applies what crosses its four edges, writing only its own cells.
// One edge's crossing as this column sees it (`toward`: a positive count moves toward this column).
inline void edgeAt(ushort e, bool toward, thread int& out, thread uint* inL, thread int* inK, thread int& nIn) {
  int a = toward ? unitsA(e) : -unitsA(e);
  if (a > 0) { inL[nIn] = liqA(e); inK[nIn] = a; nIn++; }
  else out -= a;
}
kernel void w_waterApply(device const uint* grid [[buffer(0)]], device const uint* tables [[buffer(1)]], device uchar* bricks [[buffer(2)]],
                         device atomic_int* ctr [[buffer(3)]], device const uint* L [[buffer(4)]], device uint* next [[buffer(5)]],
                         device uint* fresh [[buffer(6)]], device atomic_uint* bitsNext [[buffer(7)]], device const uint* bitsCur [[buffer(8)]],
                         device const uint* slots [[buffer(9)]], device const Edge* edges [[buffer(10)]],
                         constant WaterParams& P [[buffer(12)]], device const uint* stamps [[buffer(14)]],
                         uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint wakeMask;
  if (tg >= min(uint(atomic_load_explicit(&ctr[P.cur], memory_order_relaxed)), W_MAXL)) return;
  int3 b = unpackB(L[tg]);
  // the slots that hold this brick's edges: its own (+x, +z) and those of its -x and -z neighbours, if written now
  uint s0 = slotOf(b, grid, slots), sx = slotOf(b - int3(1, 0, 0), grid, slots), sz = slotOf(b - int3(0, 0, 1), grid, slots);
  if (s0 && stamps[s0 - 1u] != P.stamp) s0 = 0;
  if (sx && stamps[sx - 1u] != P.stamp) sx = 0;
  if (sz && stamps[sz - 1u] != P.stamp) sz = 0;
  if (!s0 && !sx && !sz) return;
  ulong base = brickBase(b, grid, tables);
  if (base == ~0ul) return;
  if (t == 0u) atomic_store_explicit(&wakeMask, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int x = int(t & 7u), z = int(t >> 3);
  // what leaves through the four edges, and what arrives (units and liquid) through each
  int out = 0, nIn = 0, inK[4];
  uint inL[4];
  if (s0) { Edge e = edges[ulong(s0 - 1u) * 64ul + t]; edgeAt(e.a.x, false, out, inL, inK, nIn); edgeAt(e.a.y, false, out, inL, inK, nIn); }
  if (x > 0) { if (s0) edgeAt(edges[ulong(s0 - 1u) * 64ul + t - 1u].a.x, true, out, inL, inK, nIn); }
  else if (sx) edgeAt(edges[ulong(sx - 1u) * 64ul + uint(z * 8 + 7)].a.x, true, out, inL, inK, nIn);
  if (z > 0) { if (s0) edgeAt(edges[ulong(s0 - 1u) * 64ul + t - 8u].a.y, true, out, inL, inK, nIn); }
  else if (sz) edgeAt(edges[ulong(sz - 1u) * 64ul + uint(56 + x)].a.y, true, out, inL, inK, nIn);
  uint mask = 0;
  if (out || nIn) {
    Col c = column(bricks, base, x, z, aboveCell(b * 8 + int3(x, 8, z), grid, tables, bricks));
    int h = c.head, lo = h, hi = h, left = 0;
    if (out) { takeUnits(bricks, base, x, z, h, out); h -= out; lo = h; }
    // the top cell's own liquid first, then each other kind in turn
    uint cur = (h & (W_U - 1)) ? liqOf(bricks[base + loc(int3(x, h >> W_S, z))]) : W_DRY;
    for (uint todo = (1u << nIn) - 1u; todo; ) {
      uint pick = W_DRY;
      for (int i = 0; i < nIn; i++) if ((todo >> i) & 1u) { if (inL[i] == cur) { pick = cur; break; } pick = min(pick, inL[i]); }
      int k = 0;
      for (int i = 0; i < nIn; i++) if (((todo >> i) & 1u) && inL[i] == pick) { k += inK[i]; todo &= ~(1u << i); }
      left += putUnits(bricks, base, x, z, h, pick, k);
      cur = pick;
    }
    hi = max(hi, h);
    if (left) { fillGaps(bricks, base, x, z, left); lo = 0; }
    // wake the bricks next to the cells that changed
    int y0 = (lo - 1) >> W_S, y1 = (hi - 1) >> W_S;
    int3 o0 = int3(x == 0 ? -1 : 0, y0 <= 0 ? -1 : 0, z == 0 ? -1 : 0), o1 = int3(x == 7 ? 1 : 0, y1 >= 7 ? 1 : 0, z == 7 ? 1 : 0);
    for (int oy = o0.y; oy <= o1.y; oy++) for (int oz = o0.z; oz <= o1.z; oz++) for (int ox = o0.x; ox <= o1.x; ox++)
      mask |= 1u << uint((ox + 1) + 3 * (oz + 1) + 9 * (oy + 1));
  }
  if (mask) atomic_fetch_or_explicit(&wakeMask, mask, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  wakeInsert(atomic_load_explicit(&wakeMask, memory_order_relaxed), t, b, bitsNext, bitsCur, ctr, P.next, next, fresh);
}
