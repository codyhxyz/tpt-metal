// Solid surfaces by surface nets: the mesher piece. Contract: Shaders/Surface.h.
//
// Occupancy is solid(code), one bit per cell; samples sit at cell centres. The dual cube with min cell d (cells
// d .. d+1) holds one vertex when its 8 cells are mixed, and every sample edge whose two cells differ is one quad
// joining the 4 dual cubes around it. Every solid material is rounded, from cells within S_HALO only:
//   base     the mean of the cube's crossed-edge midpoints (SN_V0, by corner configuration)
//   relax 2x each vertex -> the mean of its mesh neighbours (adjacent cubes whose shared face is mixed), kept inside
//            its own dual cube (Gibson's constrained surface nets: smooth slopes and rounded edges, yet a one-cell
//            feature never collapses)
//   normal   minus the occupancy gradient under a 6-cell binomial kernel centred on the cube's centre
// Loose stuff keeps its cells' lumps: a vertex whose dual cube's solid cells are all powders skips both relaxations
// (it stays at its base vertex) and takes its normal from the next binomial kernel down, 4 cells, so a pile reads as
// the cells it is made of, as tpt3d's lightly blurred powders do. A vertex touching any non-powder solid is rounded
// as before. The test reads only the cube's 8 cells, so both mcs at a seam decide alike.
// A vertex at d after two relaxations depends on cells d-2 .. d+3; an mc's vertices have d in [-1, 31] (relative),
// so an mc reads cells [-3, 34]: exactly S_HALO. Two mcs sharing a vertex run the same arithmetic on the same cells:
// seams are bit-identical. Outside the world, cells repeat the nearest edge cell: no faces on the world's walls.
//
// One 512-thread group per mc, in passes split by barriers:
//   1. the region's 6^3 brick entries (uniform bricks are never read); all uniform with one solidness: empty, stop
//   2. occupancy rows along x, 8 cells per 8-byte load, uniform bricks expanded to 0x00 / 0xff without a load
//   3. per row of dual cubes, its vertex mask from 4 occupancy rows with XOR / OR / shifts; per row of cells, its
//      x / y / z crossings the same way; popcounts and threadgroup prefix sums number vertices and quads
//   4. one atomic per counter takes the mc's staging range; each row lists its vertices and quads into staging
//      (vertex i of row r is vstart[r] + popcount(the row's mask below it): no hashing, no per-vertex atomics)
//   5. one thread per vertex: position and normal (the heavy part, balanced across all threads)
//   6. one thread per quad: two triangles, split along the shorter diagonal; material = the solid cell's byte
// Threadgroup memory 22.6 KB: occ 38 x 38 rows x 8 B = 11.3 KB (bit i of row (ry, rz) = cell (i, ry, rz) - 3);
// vmask / vstart 2 x 33 x 33 x 4 B = 8.5 KB (rows of dual cubes, d.y and d.z in [-1, 31]); brick entries 864 B;
// SN_V0 1 KB; gradient row table 256 B; scan sums.

#define SN_T 512                     // threads per group
#define SN_SG (SN_T / 32)
#define SN_R 38                      // region side, cells: the mc and S_HALO on both sides
#define SN_V 33                      // dual-cube rows per side holding the mc's vertices (d in [-1, 31])
#define SN_B 6                       // region side, bricks (cells -3 .. 34 lie in bricks -1 .. 4)
static_assert(S_HALO == 3 && S_MC == 32, "the region layout assumes 32-cell mcs and a 3-cell halo");

// Base vertex of a dual cube by corner configuration (bit i: corner (i & 1, i >> 1 & 1, i >> 2 & 1) solid): the mean
// of the midpoints of the edges whose corners differ, within the cube, unorm8 x | y << 8 | z << 16.
constant uint SN_V0[256] = {
  0x808080, 0x2a2a2a, 0x2a2ad4, 0x404080, 0x2ad42a, 0x408040, 0x2a8080, 0x4c8080,
  0x2ad4d4, 0x2a8080, 0x4080bf, 0x4c8080, 0x40bf80, 0x4c8080, 0x4c8080, 0x808080,
  0xd42a2a, 0x804040, 0x802a80, 0x804c80, 0x80802a, 0x80804c, 0x636363, 0x808080,
  0x808080, 0x5b8080, 0x805b80, 0x808080, 0x80805b, 0x808080, 0x806060, 0xb28080,
  0xd42ad4, 0x802a80, 0x8040bf, 0x804c80, 0x808080, 0x805b80, 0x5b8080, 0x808080,
  0x8080d4, 0x63639c, 0x8080b2, 0x808080, 0x8080a4, 0x80609f, 0x808080, 0xb28080,
  0xbf4080, 0x804c80, 0x804c80, 0x808080, 0x80805b, 0x808080, 0x608060, 0x80b280,
  0x8080a4, 0x60809f, 0x808080, 0x80b280, 0x808080, 0x8080a4, 0x80805b, 0xbfbf80,
  0xd4d42a, 0x80802a, 0x808080, 0x80805b, 0x80bf40, 0x80804c, 0x5b8080, 0x808080,
  0x80d480, 0x639c63, 0x80a480, 0x809f60, 0x80b280, 0x808080, 0x808080, 0xb28080,
  0xbf8040, 0x80804c, 0x805b80, 0x808080, 0x80804c, 0x808080, 0x606080, 0x8080b2,
  0x80a480, 0x609f80, 0x808080, 0x80a480, 0x808080, 0x8080b2, 0x805b80, 0xbf80bf,
  0xd48080, 0x9c6363, 0xa48080, 0x9f8060, 0xa48080, 0x9f6080, 0x808080, 0xa48080,
  0x9c9c9c, 0x808080, 0x9f9f80, 0x9c9c63, 0x9f809f, 0x9c639c, 0xa48080, 0xd48080,
  0xb28080, 0x808080, 0x808080, 0x80b280, 0x808080, 0x8080b2, 0x5b8080, 0x80bfbf,
  0x809f9f, 0x639c9c, 0x80a480, 0x80d480, 0x8080a4, 0x8080d4, 0x808080, 0xd4d4d4,
  0xd4d4d4, 0x808080, 0x8080d4, 0x8080a4, 0x80d480, 0x80a480, 0x639c9c, 0x809f9f,
  0x80bfbf, 0x5b8080, 0x8080b2, 0x808080, 0x80b280, 0x808080, 0x808080, 0xb28080,
  0xd48080, 0xa48080, 0x9c639c, 0x9f809f, 0x9c9c63, 0x9f9f80, 0x808080, 0x9c9c9c,
  0xa48080, 0x808080, 0x9f6080, 0xa48080, 0x9f8060, 0xa48080, 0x9c6363, 0xd48080,
  0xbf80bf, 0x805b80, 0x8080b2, 0x808080, 0x80a480, 0x808080, 0x609f80, 0x80a480,
  0x8080b2, 0x606080, 0x808080, 0x80804c, 0x808080, 0x805b80, 0x80804c, 0xbf8040,
  0xb28080, 0x808080, 0x808080, 0x80b280, 0x809f60, 0x80a480, 0x639c63, 0x80d480,
  0x808080, 0x5b8080, 0x80804c, 0x80bf40, 0x80805b, 0x808080, 0x80802a, 0xd4d42a,
  0xbfbf80, 0x80805b, 0x8080a4, 0x808080, 0x80b280, 0x808080, 0x60809f, 0x8080a4,
  0x80b280, 0x608060, 0x808080, 0x80805b, 0x808080, 0x804c80, 0x804c80, 0xbf4080,
  0xb28080, 0x808080, 0x80609f, 0x8080a4, 0x808080, 0x8080b2, 0x63639c, 0x8080d4,
  0x808080, 0x5b8080, 0x805b80, 0x808080, 0x804c80, 0x8040bf, 0x802a80, 0xd42ad4,
  0xb28080, 0x806060, 0x808080, 0x80805b, 0x808080, 0x805b80, 0x5b8080, 0x808080,
  0x808080, 0x636363, 0x80804c, 0x80802a, 0x804c80, 0x802a80, 0x804040, 0xd42a2a,
  0x808080, 0x4c8080, 0x4c8080, 0x40bf80, 0x4c8080, 0x4080bf, 0x2a8080, 0x2ad4d4,
  0x4c8080, 0x2a8080, 0x408040, 0x2ad42a, 0x404080, 0x2a2ad4, 0x2a2a2a, 0x808080,
};

// Gradient kernel over 6 cells: smoothing B5 = 1 5 10 10 5 1, derivative = the differences of B4 (1 4 6 4 1).
constant int SN_SW[6] = { 1, 5, 10, 10, 5, 1 };
constant int SN_DW[6] = { -1, -3, -2, 2, 3, 1 };
// 4 cells: smoothing B3 = 1 3 3 1, derivative = the differences of B2 (1 2 1).
constant int SN_SW4[4] = { 1, 3, 3, 1 };
constant int SN_DW4[4] = { -1, -1, 1, 1 };

// 8 cells (one brick row along x, little-endian) -> 8 solid bits: the byte is nonzero and below 128 (not liquid).
inline uint snSolid8(ulong w) {
  ulong nz = ((w & 0x7f7f7f7f7f7f7f7ful) + 0x7f7f7f7f7f7f7f7ful) | w;    // byte msb: byte != 0 (no carry escapes)
  ulong s = nz & ~w & 0x8080808080808080ul;
  return uint((((s >> 7) * 0x0102040810204080ul) >> 56));               // byte i's bit lands on bit 56 + i
}
// Occupancy row (region bit i = cell i - 3) -> bits for cells x in [0, 32): the mc's own cells.
inline uint snIn(ulong r) { return uint(r >> 3); }
// Cells x and x + 1 differ, x in [0, 32).
inline uint snCrossX(ulong r) { return snIn(r ^ (r >> 1)); }

// Vertex mask of the dual-cube row (dy, dz), bit d.x + 1 for d.x in [-1, 31]: the cubes some quad of this mc uses.
// A crossing along x at cell c is the quad of cubes c - (0, j, k); along y, c - (i, 0, k); along z, c - (i, j, 0)
// (i, j, k in {0, 1}). Only crossings whose lower cell lies in the mc count, hence the validity flags.
inline ulong snDualRow(int dy, int dz, threadgroup const ulong* occ) {
  uint ry = uint(dy + 3), rz = uint(dz + 3);
  ulong A = occ[rz * SN_R + ry], B = occ[rz * SN_R + ry + 1], C = occ[(rz + 1) * SN_R + ry], D = occ[(rz + 1) * SN_R + ry + 1];
  bool y0 = dy >= 0, y1 = dy < 31, z0 = dz >= 0, z1 = dz < 31;
  uint ux = (y0 && z0 ? snCrossX(A) : 0u) | (y1 && z0 ? snCrossX(B) : 0u) | (y0 && z1 ? snCrossX(C) : 0u) | (y1 && z1 ? snCrossX(D) : 0u);
  uint uyz = (y0 && z0 ? snIn(A ^ B) | snIn(A ^ C) : 0u) | (y0 && z1 ? snIn(C ^ D) : 0u) | (y1 && z0 ? snIn(B ^ D) : 0u);
  return (ulong(ux | uyz) << 1) | ulong(uyz);
}

// Index (within the mc) of the vertex of dual cube d, d in [-1, 31]^3.
inline uint snVtx(int3 d, threadgroup const uint* vmask, threadgroup const uint* vstart) {
  uint r = uint(d.z + 1) * SN_V + uint(d.y + 1), s = vstart[r];
  uint below = (s & 1u) + popcount(vmask[r] & ((1u << uint(max(d.x, 0))) - 1u));
  return (s >> 1) + (d.x < 0 ? 0u : below);
}

// A vertex's 6^3-cell window (cells d - 2 .. d + 3), slice z = k as bits 6 y + x of W[k]. Its dual cubes are a,b,c in
// 0..4 (the vertex's own is 2,2,2); a cube's corner configuration from 4 two-bit pieces.
inline uint snCfg(thread const ulong* W, int a, int b, int c) {
  ulong lo = W[c] >> (6 * b + a), hi = W[c + 1] >> (6 * b + a);
  return uint((lo & 3ul) | ((lo >> 4) & 12ul) | ((hi & 3ul) << 4) | (hi & 192ul));
}
// The face toward neighbour i (+x -x +y -y +z -z) is mixed: the two cubes share a quad edge (mesh neighbours).
inline bool snFace(uint cfg, uint fm) { uint f = cfg & fm; return f != 0u && f != fm; }

// Position (window coordinates: cube k spans [k, k + 1]) after two constrained relaxations.
inline float3 snRelaxed(thread const ulong* W, threadgroup const uint* v0) {
  const int3 dir[6] = { int3(1, 0, 0), int3(-1, 0, 0), int3(0, 1, 0), int3(0, -1, 0), int3(0, 0, 1), int3(0, 0, -1) };
  const uint face[6] = { 0xaau, 0x55u, 0xccu, 0x33u, 0xf0u, 0x0fu };
  uint cc = snCfg(W, 2, 2, 2);
  float3 acc = 0.0f;
  float n = 0.0f;
  #pragma unroll
  for (int i = 0; i < 6; i++) {
    int3 nb = int3(2) + dir[i];
    uint nc = snCfg(W, nb.x, nb.y, nb.z);
    float3 acc1 = 0.0f;
    float n1 = 0.0f;
    #pragma unroll
    for (int j = 0; j < 6; j++) {   // first relaxation of neighbour nb: the mean of its neighbours' base vertices
      int3 q = nb + dir[j];
      float3 p = float3(q) + unpack_unorm4x8_to_float(v0[snCfg(W, q.x, q.y, q.z)]).xyz;
      bool on = snFace(nc, face[j]);
      acc1 += on ? p : 0.0f; n1 += on ? 1.0f : 0.0f;
    }
    float3 v1 = clamp(acc1 / max(n1, 1.0f), float3(nb), float3(nb) + 1.0f);
    bool on = snFace(cc, face[i]);
    acc += on ? v1 : 0.0f; n += on ? 1.0f : 0.0f;
  }
  return clamp(acc / max(n, 1.0f), float3(2.0f), float3(3.0f));
}

// Outward normal: minus the occupancy gradient (integer, so both mcs at a seam agree exactly). Per x row of 6 cells
// the table gives its smoothing (low 16 bits) and derivative (high 16, signed) sums. Zero only for symmetric
// configurations: then the cube's own corner gradient, then up.
inline float3 snNormal(thread const ulong* W, threadgroup const uint* glut) {
  int3 g = 0;
  #pragma unroll
  for (int k = 0; k < 6; k++) {
    #pragma unroll
    for (int j = 0; j < 6; j++) {
      uint e = glut[uint(W[k] >> (6 * j)) & 63u];
      int s = int(e & 0xffffu), d = int(e) >> 16;
      g += int3(d * SN_SW[j] * SN_SW[k], s * SN_DW[j] * SN_SW[k], s * SN_SW[j] * SN_DW[k]);
    }
  }
  if (all(g == 0)) {
    uint c = snCfg(W, 2, 2, 2);
    g = int3(popcount(c & 0xaau), popcount(c & 0xccu), popcount(c & 0xf0u))
      - int3(popcount(c & 0x55u), popcount(c & 0x33u), popcount(c & 0x0fu));
  }
  return all(g == 0) ? float3(0.0f, 1.0f, 0.0f) : -float3(g);
}

// The same over the window's middle 4 cells (cells d - 1 .. d + 2): window bits 1..4 of each row, slices 1..4.
inline float3 snNormal4(thread const ulong* W, threadgroup const uint* glut4) {
  int3 g = 0;
  #pragma unroll
  for (int k = 0; k < 4; k++) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      uint e = glut4[uint(W[k + 1] >> (6 * (j + 1) + 1)) & 15u];
      int s = int(e & 0xffffu), d = int(e) >> 16;
      g += int3(d * SN_SW4[j] * SN_SW4[k], s * SN_DW4[j] * SN_SW4[k], s * SN_SW4[j] * SN_DW4[k]);
    }
  }
  if (all(g == 0)) {
    uint c = snCfg(W, 2, 2, 2);
    g = int3(popcount(c & 0xaau), popcount(c & 0xccu), popcount(c & 0xf0u))
      - int3(popcount(c & 0x55u), popcount(c & 0x33u), popcount(c & 0x0fu));
  }
  return all(g == 0) ? float3(0.0f, 1.0f, 0.0f) : -float3(g);
}

// Every solid cell of dual cube d (cells d .. d + 1, mc-relative, in [-1, 32]) is a powder. cfg: its corner bits.
inline bool snLoose(int3 d, uint cfg, threadgroup const uint* brk, device const uchar* bytes) {
  for (uint i = 0; i < 8u; i++) {
    if (((cfg >> i) & 1u) == 0u) continue;
    int3 s = d + int3(i & 1u, (i >> 1) & 1u, (i >> 2) & 1u);
    uint e = brk[(uint((s.z >> 3) + 1) * 6u + uint((s.y >> 3) + 1)) * 6u + uint((s.x >> 3) + 1)];
    uint mat = (e & W_UNI) ? (e & 255u) : uint(bytes[ulong(e) * 512ul + loc(s & 7)]);
    if (MATS[min(mat, uint(NMAT) - 1u)].state != ST_POWDER) return false;
  }
  return true;
}

// Exclusive prefix sums of two counts over the group; total gets the sums.
inline uint2 snScan(uint2 v, uint lane, uint sg, threadgroup uint2* sums, thread uint2& total) {
  uint2 x = uint2(simd_prefix_exclusive_sum(v.x), simd_prefix_exclusive_sum(v.y));
  if (lane == 31u) sums[sg] = x + v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u) {
    uint2 s = lane < uint(SN_SG) ? sums[lane] : uint2(0);
    uint2 e = uint2(simd_prefix_exclusive_sum(s.x), simd_prefix_exclusive_sum(s.y));
    if (lane < uint(SN_SG)) sums[lane] = e;
    if (lane == uint(SN_SG) - 1u) sums[SN_SG] = e + s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total = sums[SN_SG];
  return x + sums[sg];
}

// Indirect arguments: one group per job, the count capped at the job list's capacity. p: count's word, capacity.
kernel void sn_args(device const uint* count [[buffer(0)]], device uint* args [[buffer(1)]], constant uint2& p [[buffer(2)]]) {
  args[0] = min(count[p.x], p.y); args[1] = 1u; args[2] = 1u;
}

// The solid surface of mc jobs[job] into staging; out[job] says where. scratch: one word per staging vertex.
[[max_total_threads_per_threadgroup(SN_T)]]
kernel void sn_mesh(device const uint* grid [[buffer(0)]], device const uint* tables [[buffer(1)]],
                    device const ulong* bricks [[buffer(2)]], device const uint* jobs [[buffer(3)]],
                    device MeshOut* out [[buffer(4)]], device packed_float3* verts [[buffer(5)]],
                    device uint* indices [[buffer(6)]], device SurfPrim* prims [[buffer(7)]],
                    device atomic_uint* sc [[buffer(8)]], device uint* scratch [[buffer(9)]],
                    uint job [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup ulong occ[SN_R * SN_R];
  threadgroup uint vmask[SN_V * SN_V], vstart[SN_V * SN_V];
  threadgroup uint brk[SN_B * SN_B * SN_B], v0[256], glut[64], glut4[16];
  threadgroup uint2 sums[SN_SG + 1];
  threadgroup uint flags[SN_SG], stage[3];

  uint mc = jobs[job];
  int3 m = mcOf(mc), b0 = m * 4 - 1, c0 = m * S_MC;   // the region's first brick; the mc's first cell

  // 1. brick entries of the region (coordinates clamped into the world), and the tables
  uint f = 0;
  if (t < uint(SN_B * SN_B * SN_B)) {
    int3 b = clamp(b0 + int3(t % 6u, (t / 6u) % 6u, t / 36u), int3(0), int3(W_BX - 1, NH / 8 - 1, W_BX - 1));
    uint ce = grid[chunkIdx(b >> 3)];
    uint e = (ce & W_UNI) ? ce : tables[ulong(ce) * 512ul + loc(b & 7)];
    brk[t] = e;
    f = (e & W_UNI) ? (solid(e & 255u) ? 1u : 2u) : 4u;
  }
  if (t >= 256u) v0[t - 256u] = SN_V0[t - 256u];
  if (t < 64u) {
    int s = 0, d = 0;
    for (int i = 0; i < 6; i++) if ((t >> i) & 1u) { s += SN_SW[i]; d += SN_DW[i]; }
    glut[t] = uint(s) | (uint(d) << 16);
  }
  if (t >= 64u && t < 80u) {
    uint b = t - 64u;
    int s = 0, d = 0;
    for (int i = 0; i < 4; i++) if ((b >> i) & 1u) { s += SN_SW4[i]; d += SN_DW4[i]; }
    glut4[b] = uint(s) | (uint(d) << 16);
  }
  f = simd_or(f);
  if (lane == 0u) flags[sg] = f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  f = 0;
  for (uint i = 0; i < uint(SN_SG); i++) f |= flags[i];
  if (f == 1u || f == 2u) {   // every brick uniform, all solid or all not: no surface
    if (t == 0u) out[job] = MeshOut{ mc, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
    return;
  }

  // 2. occupancy rows
  for (uint r = t; r < uint(SN_R * SN_R); r += SN_T) {
    int wy = clamp(c0.y - 3 + int(r % SN_R), 0, NH - 1), wz = clamp(c0.z - 3 + int(r / SN_R), 0, NW - 1);
    threadgroup const uint* be = brk + (uint((wz >> 3) - b0.z) * 6u + uint((wy >> 3) - b0.y)) * 6u;
    ulong lrow = ulong((wy & 7) * 8 + (wz & 7));                   // the x row's 8-byte word within a brick
    uint s[6];
    for (int k = 0; k < 6; k++) {
      uint e = be[k];
      s[k] = (e & W_UNI) ? (solid(e & 255u) ? 0xffu : 0u) : snSolid8(bricks[ulong(e) * 64ul + lrow]);
    }
    ulong row = ulong(s[0] >> 5) | (ulong(s[1]) << 3) | (ulong(s[2]) << 11) | (ulong(s[3]) << 19) | (ulong(s[4]) << 27)
              | (ulong(s[5] & 7u) << 35);
    if (m.x == 0) row = (row & ~7ul) | ((row >> 3) & 1ul) * 7ul;                                   // x < 0: cell 0
    if (m.x == S_MCX - 1) row = (row & ((1ul << 35) - 1ul)) | ((row >> 34) & 1ul) * (7ul << 35);   // x >= NW: NW - 1
    occ[r] = row;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3. vertex rows (3 per thread at most) and quad rows (2 per thread): masks, counts, numbering
  ulong vr[3];
  uint vc = 0, qc = 0;
  for (uint i = 0; i < 3u; i++) {
    uint r = t + i * SN_T;
    vr[i] = r < uint(SN_V * SN_V) ? snDualRow(int(r % SN_V) - 1, int(r / SN_V) - 1, occ) : 0ul;
    vc += popcount(uint(vr[i])) + popcount(uint(vr[i] >> 32));
  }
  for (uint r = t; r < 1024u; r += SN_T) {
    uint o = (r >> 5) * SN_R + (r & 31u) + 3u * SN_R + 3u;
    ulong R = occ[o];
    qc += popcount(snCrossX(R)) + popcount(snIn(R ^ occ[o + 1])) + popcount(snIn(R ^ occ[o + SN_R]));
  }
  uint2 total;
  uint2 base = snScan(uint2(vc, qc), lane, sg, sums, total);
  uint vi = base.x;
  for (uint i = 0; i < 3u; i++) {
    uint r = t + i * SN_T;
    if (r >= uint(SN_V * SN_V)) break;
    vmask[r] = uint(vr[i] >> 1); vstart[r] = (vi << 1) | uint(vr[i] & 1ul);
    vi += popcount(uint(vr[i])) + popcount(uint(vr[i] >> 32));
  }
  uint nv = total.x, nq = total.y;
  if (nq == 0u) {             // mixed bricks, but no solid / non-solid boundary in the mc
    if (t == 0u) out[job] = MeshOut{ mc, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
    return;
  }

  // 4. staging ranges: one atomic per counter for the whole mc
  if (t == 0u) {
    uint vo = atomic_fetch_add_explicit(&sc[S_SC_VTX], nv, memory_order_relaxed);
    uint to = atomic_fetch_add_explicit(&sc[S_SC_TRI], 2u * nq, memory_order_relaxed);
    bool ok = ulong(vo) + nv <= atomic_load_explicit(&sc[S_SC_VCAP], memory_order_relaxed)
           && ulong(to) + 2u * nq <= atomic_load_explicit(&sc[S_SC_TCAP], memory_order_relaxed);
    stage[0] = vo; stage[1] = to; stage[2] = ok;
    out[job] = MeshOut{ mc, ok ? 2u * nq : S_RETRY, vo, nv, to, 0u, 0u, 0u };
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (stage[2] == 0u) return;
  uint vo = stage[0], to = stage[1];

  //    each row lists its vertices (dual cube + 1, 6 bits per axis) in scratch and its quads (cell, axis, which
  //    side is solid) in the first index word of the quad's first triangle; passes 5 and 6 overwrite both in place
  for (uint i = 0; i < 3u; i++) {
    uint r = t + i * SN_T;
    if (r >= uint(SN_V * SN_V)) break;
    uint yz = ((r / SN_V) << 12) | ((r % SN_V) << 6), k = vstart[r] >> 1;
    for (ulong v = vr[i]; v != 0ul; v &= v - 1ul) scratch[vo + k++] = yz | uint(ctz(v));
  }
  uint qi = base.y;
  for (uint r = t; r < 1024u; r += SN_T) {
    uint o = (r >> 5) * SN_R + (r & 31u) + 3u * SN_R + 3u;
    ulong R = occ[o];
    uint low = snIn(R), yz = ((r & 31u) << 5) | ((r >> 5) << 10);
    uint X[3] = { snCrossX(R), snIn(R ^ occ[o + 1]), snIn(R ^ occ[o + SN_R]) };
    for (uint a = 0; a < 3u; a++)
      for (uint x = X[a]; x != 0u; x &= x - 1u) {
        uint cx = ctz(x);
        indices[3u * (to + 2u * qi++)] = cx | yz | (a << 15) | (((low >> cx) & 1u) << 17);
      }
  }
  threadgroup_barrier(mem_flags::mem_device);

  // 5. vertices: position and normal from the 6^3-cell window
  device const uchar* bytes = (device const uchar*)bricks;
  for (uint i = t; i < nv; i += SN_T) {
    uint desc = scratch[vo + i];
    int3 d = int3(desc & 63u, (desc >> 6) & 63u, desc >> 12) - 1;
    ulong W[6];
    #pragma unroll
    for (int k = 0; k < 6; k++) {
      uint o = uint(d.z + 1 + k) * SN_R + uint(d.y + 1);
      ulong w = 0;
      #pragma unroll
      for (int j = 0; j < 6; j++) w |= ((occ[o + uint(j)] >> uint(d.x + 1)) & 63ul) << (6 * j);
      W[k] = w;
    }
    // window cube 2 is dual cube d, whose vertex box starts at cell centre d + 0.5: world = d + 0.5 + (p - 2)
    uint cfg = snCfg(W, 2, 2, 2);
    bool loose = snLoose(d, cfg, brk, bytes);
    float3 p = loose ? float3(2.0f) + unpack_unorm4x8_to_float(v0[cfg]).xyz : snRelaxed(W, v0);
    verts[vo + i] = packed_float3(float3(c0 + d) + (p - 1.5f));
    scratch[vo + i] = octEncode(loose ? snNormal4(W, glut4) : snNormal(W, glut));
  }
  threadgroup_barrier(mem_flags::mem_device);

  // 6. triangles. Quad of the crossing along axis a at cell c: cubes c - eb - ec, c - ec, c, c - eb ((a, b, c)
  //    cyclic) wind counter-clockwise seen from +a, which is outward when the lower cell is the solid one.
  for (uint q = t; q < nq; q += SN_T) {
    uint tri = to + 2u * q, desc = indices[3u * tri];
    int3 c = int3(desc & 31u, (desc >> 5) & 31u, (desc >> 10) & 31u);
    uint a = (desc >> 15) & 3u;
    bool lowSolid = (desc >> 17) & 1u;
    int3 ea = int3(a == 0u, a == 1u, a == 2u), eb = ea.zxy, ec = ea.yzx;
    uint i0 = snVtx(c - eb - ec, vmask, vstart), i1 = snVtx(c - ec, vmask, vstart);
    uint i2 = snVtx(c, vmask, vstart), i3 = snVtx(c - eb, vmask, vstart);
    if (!lowSolid) { uint s = i1; i1 = i3; i3 = s; }
    float3 p0 = verts[vo + i0], p1 = verts[vo + i1], p2 = verts[vo + i2], p3 = verts[vo + i3];
    bool d02 = distance_squared(p0, p2) <= distance_squared(p1, p3);
    uint3 A = d02 ? uint3(i0, i1, i2) : uint3(i0, i1, i3);
    uint3 B = d02 ? uint3(i0, i2, i3) : uint3(i1, i2, i3);
    uint base3 = 3u * tri;
    indices[base3] = A.x; indices[base3 + 1u] = A.y; indices[base3 + 2u] = A.z;
    indices[base3 + 3u] = B.x; indices[base3 + 4u] = B.y; indices[base3 + 5u] = B.z;
    int3 s = lowSolid ? c : c + ea;                  // the solid cell, in [0, 32]: bricks 1 .. 5 of the region
    uint e = brk[(uint(s.z >> 3) + 1u) * 36u + (uint(s.y >> 3) + 1u) * 6u + uint(s.x >> 3) + 1u];
    uint mat = (e & W_UNI) ? (e & 255u) : uint(bytes[ulong(e) * 512ul + loc(s & 7)]);
    prims[tri] = SurfPrim{ scratch[vo + A.x], scratch[vo + A.y], scratch[vo + A.z], mat };
    prims[tri + 1u] = SurfPrim{ scratch[vo + B.x], scratch[vo + B.y], scratch[vo + B.z], mat };
  }
}
