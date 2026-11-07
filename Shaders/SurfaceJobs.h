// Smooth surfaces: which mesh chunks get meshed, and the small GPU jobs that keep the surfaces piece's state (dirty
// bits, coverage, TLAS instances). Owner: the surfaces piece (Sources/BigSand/Surfaces.swift). Contract: Surface.h.
// Included by World.metal right after Surface.h.

static_assert(S_MC == 32, "an mc is 32 cells: mc coordinates are cells >> 5");

// ---- dirty marks (physics and edits)
// A group's changed cells as one bit per local coordinate on each axis (x bits 0-9, y 10-19, z 20-29, local to the
// group's brick corner, 0..9): OR is per axis, so the bits give the changed cells' bounding box exactly. Threads 0-7
// of the group then set the bits of the mcs that box, grown by S_HALO, reaches: at most 2 per axis (the box is at
// most 9 cells across, 15 with the halo).
inline uint sAxisBits(int3 l) { return (1u << uint(l.x)) | (1u << (10u + uint(l.y))) | (1u << (20u + uint(l.z))); }
inline void sMarkDirty(int3 org, uint bits, uint t, device atomic_uint* dirty) {
  if (t >= 8u || bits == 0u) return;
  uint3 m = uint3(bits & 1023u, (bits >> 10) & 1023u, bits >> 20);
  int3 lo = (org + int3(ctz(m)) - S_HALO) >> 5, hi = (org + int3(31u - clz(m)) + S_HALO) >> 5;
  int3 mc = lo + int3(t & 1u, (t >> 1) & 1u, t >> 2);
  if (any(mc > hi) || !mcInWorld(mc)) return;
  uint i = mcIdx(mc);
  atomic_fetch_or_explicit(&dirty[i >> 5], 1u << (i & 31u), memory_order_relaxed);
}

// ---- the dirty scan: every frame, the solidDirty bits of the mcs within `keep2` (mc units squared, centre to the
// camera's mc centre) are taken (cleared) and listed for the CPU, `cap` at most (the rest wait a frame). One thread
// per mc column around the camera; a column's mcs are consecutive bits.
struct SScan { int4 ref; int keep2, reach, cap, pad; };   // ref: the camera's mc; reach: the box's half side, in mcs

inline int sIsqrt(int v) {   // floor(sqrt(v)), exactly (the CPU walks the same spans)
  int h = int(sqrt(float(v)));
  while (h * h > v) h--;
  while ((h + 1) * (h + 1) <= v) h++;
  return h;
}
inline uint sBits(uint lo, uint hi) { return (hi >= 31u ? ~0u : (2u << hi) - 1u) & ~((1u << lo) - 1u); }   // bits lo..hi

kernel void s_scan(device atomic_uint* dirty [[buffer(0)]], device uint* list [[buffer(1)]], device atomic_uint* count [[buffer(2)]],
                   constant SScan& P [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
  int dx = int(g.x) - P.reach, dz = int(g.y) - P.reach;
  if (dx > P.reach || dz > P.reach) return;
  int x = P.ref.x + dx, z = P.ref.z + dz;
  if (x < 0 || z < 0 || x >= S_MCX || z >= S_MCX) return;
  int r = P.keep2 - dx * dx - dz * dz;
  if (r < 0) return;
  int h = sIsqrt(r), y0 = max(P.ref.y - h, 0), y1 = min(P.ref.y + h, S_MCY - 1);
  if (y0 > y1) return;
  uint base = (uint(z) * uint(S_MCX) + uint(x)) * uint(S_MCY);
  for (uint i = base + uint(y0), e = base + uint(y1); i <= e; i = (i | 31u) + 1u) {
    uint w = i >> 5;
    uint want = atomic_load_explicit(&dirty[w], memory_order_relaxed) & sBits(i & 31u, min(31u, (i & 31u) + (e - i)));
    if (!want) continue;
    uint n = popcount(want), at = atomic_fetch_add_explicit(count, n, memory_order_relaxed);
    if (at >= uint(P.cap)) return;
    for (uint k = n; k > uint(P.cap) - at; k--) want &= ~(1u << (31u - clz(want)));   // keep what fits
    atomic_fetch_and_explicit(&dirty[w], ~want, memory_order_relaxed);
    for (; want; want &= want - 1u) list[at++] = w * 32u + ctz(want);
  }
}

// ---- the pre-filter: one 64-thread group per job (an mc the CPU wants meshed). The mc's region (the mc and S_HALO
// cells round it) spans 6 bricks per axis; if every one of them is stored as one material and those are all solid or
// all not, no solid surface can be there: the mc is covered with no surface at once and listed in `empties`. The rest
// go to the mesher, in `meshJobs`. Counters: [0] mesh jobs, [1] empties. Bricks outside the world are skipped.
kernel void s_prefilter(device const uint* grid [[buffer(0)]], device const uint* tables [[buffer(1)]], device const uint* jobs [[buffer(2)]],
                        device uint* meshJobs [[buffer(3)]], device uint* empties [[buffer(4)]], device atomic_uint* cnt [[buffer(5)]],
                        device atomic_uint* covered [[buffer(6)]], uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint seen;   // bit 0: a non-solid uniform brick, 1: a solid one, 2: a stored (mixed) one
  if (t == 0u) atomic_store_explicit(&seen, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint m = jobs[tg];
  int3 b0 = mcOf(m) * (S_MC / 8) - 1;   // (32 m - 3) >> 3 .. (32 m + 34) >> 3
  uint f = 0;
  for (uint k = t; k < 216u; k += 64u) {
    int3 b = b0 + int3(k % 6u, k / 36u, (k / 6u) % 6u);
    if (!brickInWorld(b)) continue;
    uint ce = grid[chunkIdx(b >> 3)], e = ce;
    if (!(ce & W_UNI)) e = tables[ulong(ce) * 512ul + loc(b & 7)];
    f |= (e & W_UNI) ? (solid(e & 255u) ? 2u : 1u) : 4u;
  }
  if (f) atomic_fetch_or_explicit(&seen, f, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0u) return;
  f = atomic_load_explicit(&seen, memory_order_relaxed);
  if ((f & 4u) || f == 3u) { meshJobs[atomic_fetch_add_explicit(&cnt[0], 1u, memory_order_relaxed)] = m; return; }
  atomic_fetch_or_explicit(&covered[m >> 5], 1u << (m & 31u), memory_order_relaxed);
  empties[atomic_fetch_add_explicit(&cnt[1], 1u, memory_order_relaxed)] = m;
}

// ---- coverage changes, in frame order with the TLAS that matches them: op = mcIdx, bit 31 set to clear.
kernel void s_cover(device atomic_uint* covered [[buffer(0)]], device const uint* ops [[buffer(1)]], constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  uint op = ops[gid], i = op & 0x7fffffffu, bit = 1u << (i & 31u);
  if (op >> 31) atomic_fetch_and_explicit(&covered[i >> 5], ~bit, memory_order_relaxed);
  else atomic_fetch_or_explicit(&covered[i >> 5], bit, memory_order_relaxed);
}

// ---- TLAS instance patches: the CPU's changed descriptors (MTLIndirectAccelerationStructureInstanceDescriptor,
// 18 words) copied into the instance buffer the TLAS builds from. One thread per word; each index appears once.
#define S_INST_WORDS 18
kernel void s_patch(device uint* inst [[buffer(0)]], device const uint* idx [[buffer(1)]], device const uint* words [[buffer(2)]],
                    constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= n * S_INST_WORDS) return;
  inst[ulong(idx[gid / S_INST_WORDS]) * S_INST_WORDS + gid % S_INST_WORDS] = words[gid];
}
