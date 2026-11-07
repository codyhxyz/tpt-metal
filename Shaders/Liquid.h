// Liquid surfaces as height fields: the liquids piece. Contract: Shaders/Surface.h ("liquid slots"). Included by
// World.metal (one translation unit); the frame flow is in Sources/BigSand/Liquids.swift.
//
// A surface is the top of a liquid cell with air above; the mc holding that cell owns it. A slot reads its columns
// over a band from 2 cells below its mc to 3 above (inside S_HALO, so liquidDirty covers it), and each column takes
// the topmost surface in the band as its height, whichever mc owns it: a surface just across an mc boundary (the sea
// at y = 192 lies exactly on one, and waves cross it) is drawn by both slots from the same vertices instead of
// leaving a hole between them. A vertex is the mean height of the columns around it that have a surface, summed in a
// fixed order, so neighbouring slots compute the same float.
//
// A column without a surface in the band is dry and its triangles collapse. Vertices are shared, so it can only
// collapse by moving vertices that no surface column uses ("free") onto others: each free vertex copies the nearest
// fixed one along its row (or column), looking first one way, else the other, else a point per line. Four ways to
// look (-x, +x, -z, +z first); the slot takes the one that leaves the fewest dry columns open. A dry column that
// still can't collapse is kept when its open triangles stand on its own corners inside its own solid cells (a rock in
// a lake: the surface runs on through it, hidden) and takes a neighbour's liquid. If one can't be kept, or a column
// owns two surfaces (a cave pool under a lake), the mc can't be a height field: its slot collapses entirely and the
// mc isn't covered, so the render walks its voxels.

#define L_TG 256                     // threads per slot (heights) and per mc (full scan): 4 columns each
#define L_BAND 37                    // cells a column reads: y0 - 2 .. y0 + 34 of its mc (bit i: cell y0 - 2 + i)
#define L_TOPS 0xfffffffffull        // bits 0..35: cells that can top a surface (the cell above is in the band)
#define L_OWN 0x3fffffffcull         // bits 2..33: the mc's own cells
#define L_TOL 0.25f                  // a kept dry column's surface may lie this far above its solid
#define L_LINE 0x8000u               // vertex source past the grid: a line's own point (+ the line)

// A frame record's control words (Liquids.swift writes the scan count and zeroes the full count)
#define L_C_SCAN 0                   // scan list: new mcs from the CPU, then dirty ones appended here
#define L_C_FULL 1                   // entries the quick pass left to the full pass
#define L_C_AQ 4                     // 4..6: indirect arguments, quick pass
#define L_C_AF 8                     // 8..10: indirect arguments, full pass

// Ops on the per-mc bits (one word: an mc column), type in bits 28-31 of the word index. CLEAR and UNSLOT run
// before KNOW and SLOT, so an mc freed and given a slot in one frame ends with one.
#define L_OP_CLEAR 0u                // mcs that left the tracked ball: not known, not covered, no slot
#define L_OP_UNSLOT 1u               // mcs whose slot was freed
#define L_OP_KNOW 2u                 // mcs scanned for the first time now (their old dirt is moot)
#define L_OP_SLOT 3u                 // mcs given a slot

kernel void l_ops(device const uint2* ops [[buffer(0)]], constant uint2& range [[buffer(1)]],
                  device atomic_uint* known [[buffer(2)]], device atomic_uint* covered [[buffer(3)]],
                  device atomic_uint* hasSlot [[buffer(4)]], device atomic_uint* dirty [[buffer(5)]],
                  uint gid [[thread_position_in_grid]]) {
  if (gid >= range.y) return;
  uint2 op = ops[range.x + gid];
  uint w = op.x & 0x0fffffffu, m = op.y;
  switch (op.x >> 28) {
    case L_OP_CLEAR:
      atomic_fetch_and_explicit(&known[w], ~m, memory_order_relaxed);
      atomic_fetch_and_explicit(&covered[w], ~m, memory_order_relaxed);
      atomic_fetch_and_explicit(&hasSlot[w], ~m, memory_order_relaxed);
      break;
    case L_OP_UNSLOT: atomic_fetch_and_explicit(&hasSlot[w], ~m, memory_order_relaxed); break;
    case L_OP_KNOW:
      atomic_fetch_or_explicit(&known[w], m, memory_order_relaxed);
      atomic_fetch_and_explicit(&dirty[w], ~m, memory_order_relaxed);
      break;
    default: atomic_fetch_or_explicit(&hasSlot[w], m, memory_order_relaxed); break;
  }
}

// liquidDirty over the tracked square of mc columns (one word each). Dirty known mcs join the scan list. One with a
// slot stays covered: its surface is a frame stale until the refit, which can't be seen, while handing the mc to the
// voxel walk and back would show (two renderers never agree exactly). One without a slot loses coverage: it may have
// just got a surface, which only the walk can draw until a slot is made; with no list (list == 0: every frame record is in flight) they keep their
// dirt for the next frame. Dirt on mcs not known is dropped: they are scanned afresh when they come in.
struct LDirtyArgs { int x0, z0, w, h; uint list, cap, pad0, pad1; };
kernel void l_dirty(device atomic_uint* dirty [[buffer(0)]], device const uint* known [[buffer(1)]], device atomic_uint* covered [[buffer(2)]],
                    device atomic_uint* ctl [[buffer(3)]], device uint* scan [[buffer(4)]], constant LDirtyArgs& A [[buffer(5)]],
                    device const uint* hasSlot [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(A.w * A.h)) return;
  uint w = uint(A.z0 + int(gid) / A.w) * uint(S_MCX) + uint(A.x0 + int(gid) % A.w);
  uint d = atomic_load_explicit(&dirty[w], memory_order_relaxed);
  if (!d) return;
  uint k = d & known[w], listed = 0;
  if (k & ~hasSlot[w]) atomic_fetch_and_explicit(&covered[w], ~(k & ~hasSlot[w]), memory_order_relaxed);
  if (!A.list) return;
  if (k) {
    uint j = atomic_fetch_add_explicit(&ctl[L_C_SCAN], uint(popcount(k)), memory_order_relaxed);
    for (uint r = k; r && j < A.cap; r &= r - 1u, j++) { uint b = ctz(r); scan[j] = w * uint(S_MCY) + b; listed |= 1u << b; }
  }
  atomic_fetch_and_explicit(&dirty[w], ~((d & ~k) | listed), memory_order_relaxed);   // the rest wait for room
}

// Indirect arguments: ceil(min(ctl[a.x], a.y) / a.w) threadgroups at ctl[a.z].
kernel void l_args(device uint* ctl [[buffer(0)]], constant uint4& a [[buffer(1)]]) {
  uint n = min(ctl[a.x], a.y);
  ctl[a.z] = (n + a.w - 1u) / a.w; ctl[a.z + 1] = 1u; ctl[a.z + 2] = 1u;
}

inline void lCover(device atomic_uint* covered, uint mc) {
  atomic_fetch_or_explicit(&covered[mc >> 5], 1u << (mc & 31u), memory_order_relaxed);
}

// Does the mc own a liquid surface? Most answer from their chunk: an mc is an octant of one, and a uniform chunk
// holds no liquid, or liquid with more of it above (the bottom half), or the sea's top half, whose top row is all
// surface where the chunk above is air. The rest go to the full pass. One with none, and no slot, is covered.
kernel void l_scanQuick(device const uint* grid [[buffer(0)]], device atomic_uint* ctl [[buffer(1)]], device const uint* scan [[buffer(2)]],
                        device uchar* res [[buffer(3)]], device uint* full [[buffer(4)]], device const uint* hasSlot [[buffer(5)]],
                        device atomic_uint* covered [[buffer(6)]], constant uint& cap [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= min(atomic_load_explicit(&ctl[L_C_SCAN], memory_order_relaxed), cap)) return;
  uint mc = scan[gid];
  int3 m = mcOf(mc), c = m >> 1;
  uint ce = grid[chunkIdx(c)];
  int r = -1;
  if (ce & W_UNI) {
    uint v = ce & 255u;
    if (!isLiquid(v) || !(m.y & 1)) r = 0;
    else if (c.y + 1 >= W_CY) r = 1;                        // the top of the world: air above
    else { uint up = grid[chunkIdx(c + int3(0, 1, 0))]; if (up & W_UNI) r = (up & 255u) == AIR ? 1 : 0; }
  }
  if (r < 0) { full[atomic_fetch_add_explicit(&ctl[L_C_FULL], 1u, memory_order_relaxed)] = gid; return; }   // at most cap
  res[gid] = uchar(r);
  if (r == 0 && !bitSet(hasSlot, mc)) lCover(covered, mc);
}

// ---- reading columns. A slot's bricks: 6 x 6 x 6 around its mc, x and z from one brick before it (the columns
// beside it), y from the brick under it (the band). Outside the world: wall below and beside, air above.
inline uint lBrick(int3 b, device const uint* grid, device const uint* tables) {
  if (b.y >= NH / 8) return W_UNI | AIR;
  if (any(b < 0) || b.x >= W_BX || b.z >= W_BX) return W_UNI | WALL;
  uint ce = grid[chunkIdx(b >> 3)];
  return (ce & W_UNI) ? ce : tables[ulong(ce) * 512ul + loc(b & 7)];
}
inline void lBricks(threadgroup uint* be, int3 o, uint t, device const uint* grid, device const uint* tables) {
  for (uint k = t; k < 216u; k += L_TG) be[k] = lBrick((o >> 3) - 1 + int3(k % 6u, k / 36u, (k / 6u) % 6u), grid, tables);
}
// Column (lx, lz) of the mc (-1..32: the columns beside it too) over the band: which cells are liquid, which air.
// A uniform brick gives all its cells at once.
struct LCol { ulong liq, air; };
inline LCol lColumn(threadgroup const uint* be, int lx, int lz, device const uchar* bricks) {
  LCol c = { 0ul, 0ul };
  uint bxz = (uint(lz + 8) >> 3) * 6u + (uint(lx + 8) >> 3), in = uint((lz + 8) & 7) * 8u + uint((lx + 8) & 7);
  for (int j = 0; j < 6; j++) {
    uint e = be[uint(j) * 36u + bxz];
    int i0 = max(j * 8 - 6, 0), i1 = min(j * 8 + 2, L_BAND);   // band bits brick row j holds
    if (e & W_UNI) {
      uint m = e & 255u;
      ulong bits = ((1ul << uint(i1 - i0)) - 1ul) << uint(i0);
      if (isLiquid(m)) c.liq |= bits; else if (m == AIR) c.air |= bits;
      continue;
    }
    ulong base = ulong(e) * 512ul + in;
    for (int i = i0; i < i1; i++) {
      uint m = bricks[base + uint((i + 6) & 7) * 64u];
      c.liq |= ulong(isLiquid(m)) << uint(i); c.air |= ulong(m == AIR) << uint(i);
    }
  }
  return c;
}
inline uint lCell(threadgroup const uint* be, int lx, int lz, int i, device const uchar* bricks) {
  uint e = be[(uint(i + 6) >> 3) * 36u + (uint(lz + 8) >> 3) * 6u + (uint(lx + 8) >> 3)];
  if (e & W_UNI) return e & 255u;
  return bricks[ulong(e) * 512ul + uint((i + 6) & 7) * 64u + uint((lz + 8) & 7) * 8u + uint((lx + 8) & 7)];
}
inline ulong lSurf(LCol c) { return c.liq & (c.air >> 1) & L_TOPS; }   // cells topping a surface
inline int lTop(ulong s) { return 63 - int(clz(s)); }

// The full pass: one group per mc the quick pass couldn't decide, reading its cells.
kernel void l_scanFull(device const uint* grid [[buffer(0)]], device const uint* tables [[buffer(1)]], device const uchar* bricks [[buffer(2)]],
                       device const uint* ctl [[buffer(3)]], device const uint* scan [[buffer(4)]], device uchar* res [[buffer(5)]],
                       device const uint* full [[buffer(6)]], device const uint* hasSlot [[buffer(7)]], device atomic_uint* covered [[buffer(8)]],
                       uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint be[216];
  threadgroup atomic_uint owns;
  uint i = full[tg], mc = scan[i];
  lBricks(be, mcOf(mc) * S_MC, t, grid, tables);
  if (t == 0u) atomic_store_explicit(&owns, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong own = 0;
  for (uint k = t; k < 1024u; k += L_TG) own |= lSurf(lColumn(be, int(k & 31u), int(k >> 5), bricks)) & L_OWN;
  if (own) atomic_store_explicit(&owns, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    uint r = atomic_load_explicit(&owns, memory_order_relaxed);
    res[i] = uchar(r);
    if (!r && !bitSet(hasSlot, mc)) lCover(covered, mc);
  }
}

// ---- heights
// Column (lx, lz)'s topmost surface in the band: its height (cells, with the fill level) and its liquid.
inline void lPut(threadgroup float* H, threadgroup uchar* K, threadgroup const uint* be, int lx, int lz, ulong s, int y0,
                 device const uchar* bricks) {
  uint c = uint(lz + 1) * 34u + uint(lx + 1);
  if (!s) { H[c] = -1.0f; K[c] = 0xff; return; }
  int top = lTop(s);
  uint code = lCell(be, lx, lz, top, bricks);
  H[c] = float(y0 - 2 + top) + float(units(code)) / float(LIQ_L);
  K[c] = uchar(matOf(code));
}
// The fixed vertex nearest to place a of a line whose fixed vertices are m: before it first (else after), or after
// it first; -1 if the line has none.
inline int lAlong(ulong m, uint a, bool after) {
  ulong lo = m & ((1ul << a) - 1ul), hi = m & ~((2ul << a) - 1ul);
  if (after) { if (hi) return int(ctz(hi)); if (lo) return lTop(lo); }
  else { if (lo) return lTop(lo); if (hi) return int(ctz(hi)); }
  return -1;
}
// Where vertex (x, z) goes under direction d (0: -x first, 1: +x, 2: -z, 3: +z): a vertex index, or L_LINE + its
// row (d < 2) or column (d >= 2). Equal sources are equal points.
inline uint lSrc(threadgroup const ulong* rowM, threadgroup const ulong* colM, uint x, uint z, uint d) {
  if ((rowM[z] >> x) & 1ul) return z * 33u + x;
  if (d < 2u) { int a = lAlong(rowM[z], x, d == 1u); return a < 0 ? L_LINE + z : z * 33u + uint(a); }
  int a = lAlong(colM[x], z, d == 3u);
  return a < 0 ? L_LINE + x : uint(a) * 33u + x;
}
// Does a triangle at heights a, b, c lie in the column's solid cells (or within L_TOL above them)?
inline bool lInSolid(ulong solid, float a, float b, float c, int y0) {
  int lo = int(ceil(min3(a, b, c) - L_TOL)) - 1 - (y0 - 2), hi = int(ceil(max3(a, b, c) - L_TOL)) - 1 - (y0 - 2);
  if (lo < 0 || hi >= L_BAND) return false;
  ulong need = ((2ul << uint(hi)) - 1ul) & ~((1ul << uint(lo)) - 1ul);
  return (solid & need) == need;
}

// One group per slot to (re)compute (jobs: slot, mc): kinds, vertices and the mc's coverage; jobRes: 0 the mc owns no
// surface (the slot can go), 1 drawn, 2 can't be a height field (collapsed, not covered).
kernel void l_heights(device const uint* grid [[buffer(0)]], device const uint* tables [[buffer(1)]], device const uchar* bricks [[buffer(2)]],
                      device const uint2* jobs [[buffer(3)]], device uint* jobRes [[buffer(4)]], device packed_float3* verts [[buffer(5)]],
                      device uchar* kinds [[buffer(6)]], device atomic_uint* covered [[buffer(7)]],
                      uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint be[216];
  threadgroup float H[34 * 34];        // per column, the ones beside the mc too: topmost surface height, -1: none
  threadgroup uchar K[34 * 34];        // its liquid, 0xff: none
  threadgroup float VH[L_VERTS];       // vertex heights
  threadgroup uchar FV[L_VERTS];       // fixed: a column of the mc with a surface uses it
  threadgroup ulong rowM[33], colM[33];   // fixed vertices: bit x of row z, bit z of column x
  threadgroup uchar CS[1024];          // per column: bit d collapses under direction d; bit 4 has a surface
  threadgroup atomic_uint fails[4], flags, hqA;   // flags: 1 a column owns two surfaces, 2 the mc owns one
  threadgroup atomic_uint decide;     // the direction, + 4: the slot collapses
  uint slot = jobs[tg].x, mc = jobs[tg].y;
  int3 o = mcOf(mc) * S_MC;
  lBricks(be, o, t, grid, tables);
  if (t < 4u) atomic_store_explicit(&fails[t], 0u, memory_order_relaxed);
  if (t == 0u) { atomic_store_explicit(&flags, 0u, memory_order_relaxed); atomic_store_explicit(&hqA, 0u, memory_order_relaxed); }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 1. columns: the mc's 1024, four a thread (their solid cells kept for 3), and the 132 around them
  ulong solid[4];
  for (uint q = 0; q < 4u; q++) {
    uint k = t + q * L_TG;
    int lx = int(k & 31u), lz = int(k >> 5);
    LCol c = lColumn(be, lx, lz, bricks);
    solid[q] = ~(c.liq | c.air) & ((1ul << L_BAND) - 1ul);
    ulong s = lSurf(c), own = s & L_OWN;
    lPut(H, K, be, lx, lz, s, o.y, bricks);
    // two surfaces of its own, or its own under one just above the mc: one height can't show both
    if (own) atomic_fetch_or_explicit(&flags, 2u | uint(popcount(own) > 1 || lTop(s) >= 34), memory_order_relaxed);
  }
  if (t < 132u) {
    int lx = t < 34u ? int(t) - 1 : t < 68u ? int(t) - 35 : t < 100u ? -1 : 32;
    int lz = t < 34u ? -1 : t < 68u ? 32 : t < 100u ? int(t) - 68 : int(t) - 100;
    lPut(H, K, be, lx, lz, lSurf(lColumn(be, lx, lz, bricks)), o.y, bricks);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2. vertices: the mean of the columns around with a surface, (-x -z), (+x -z), (-x +z), (+x +z) in that order
  for (uint v = t; v < uint(L_VERTS); v += L_TG) {
    uint vx = v % 33u, vz = v / 33u, c0 = vz * 34u + vx;
    float h0 = H[c0], h1 = H[c0 + 1u], h2 = H[c0 + 34u], h3 = H[c0 + 35u], sum = 0.0f;
    uint n = 0;
    if (h0 >= 0.0f) { sum += h0; n++; }
    if (h1 >= 0.0f) { sum += h1; n++; }
    if (h2 >= 0.0f) { sum += h2; n++; }
    if (h3 >= 0.0f) { sum += h3; n++; }
    float h = n ? sum / float(n) : 0.0f;
    VH[v] = h;
    bool fx = (vx > 0u && vz > 0u && h0 >= 0.0f) || (vx < 32u && vz > 0u && h1 >= 0.0f) ||
              (vx > 0u && vz < 32u && h2 >= 0.0f) || (vx < 32u && vz < 32u && h3 >= 0.0f);
    FV[v] = uchar(fx);
    if (fx) atomic_fetch_max_explicit(&hqA, uint(h), memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 33u) { ulong m = 0; for (uint x = 0; x < 33u; x++) m |= ulong(FV[t * 33u + x]) << x; rowM[t] = m; }
  else if (t < 66u) { uint x = t - 33u; ulong m = 0; for (uint z = 0; z < 33u; z++) m |= ulong(FV[z * 33u + x]) << z; colM[x] = m; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3. dry columns: under which directions do both triangles collapse, and where would one stay open?
  for (uint q = 0; q < 4u; q++) {
    uint k = t + q * L_TG, x = k & 31u, z = k >> 5;
    if (H[(z + 1u) * 34u + x + 1u] >= 0.0f) { CS[k] = 16; continue; }
    uint v00 = z * 33u + x, v10 = v00 + 1u, v01 = v00 + 33u, v11 = v00 + 34u;
    bool f00 = FV[v00], f10 = FV[v10], f01 = FV[v01], f11 = FV[v11];
    // triangle A = v00 v01 v11, B = v00 v11 v10 (Surface.h); an open one may stay on its own corners inside solid
    bool keepA = f00 && f01 && f11 && lInSolid(solid[q], VH[v00], VH[v01], VH[v11], o.y);
    bool keepB = f00 && f11 && f10 && lInSolid(solid[q], VH[v00], VH[v11], VH[v10], o.y);
    uint cs = 0;
    for (uint d = 0; d < 4u; d++) {
      uint s00 = lSrc(rowM, colM, x, z, d), s10 = lSrc(rowM, colM, x + 1u, z, d);
      uint s01 = lSrc(rowM, colM, x, z + 1u, d), s11 = lSrc(rowM, colM, x + 1u, z + 1u, d);
      bool degA = s00 == s01 || s01 == s11 || s00 == s11, degB = s00 == s11 || s11 == s10 || s00 == s10;
      if (degA && degB) cs |= 1u << d;
      else if (!((degA || keepA) && (degB || keepB))) atomic_fetch_add_explicit(&fails[d], 1u, memory_order_relaxed);
    }
    CS[k] = uchar(cs);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 4. the direction with the fewest open dry columns; none open (and one surface a column), or the slot collapses
  if (t == 0u) {
    uint best = 0, fb = atomic_load_explicit(&fails[0], memory_order_relaxed);
    for (uint d = 1; d < 4u; d++) { uint f = atomic_load_explicit(&fails[d], memory_order_relaxed); if (f < fb) { fb = f; best = d; } }
    uint fl = atomic_load_explicit(&flags, memory_order_relaxed);
    bool empty = !(fl & 2u), bad = !empty && ((fl & 1u) || fb > 0u);
    atomic_store_explicit(&decide, best | (empty || bad ? 4u : 0u), memory_order_relaxed);
    if (bad) atomic_fetch_and_explicit(&covered[mc >> 5], ~(1u << (mc & 31u)), memory_order_relaxed);
    else lCover(covered, mc);
    jobRes[tg] = empty ? 0u : bad ? 2u : 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint dc = atomic_load_explicit(&decide, memory_order_relaxed), best = dc & 3u;
  bool off = dc >= 4u;
  device uchar* kd = kinds + ulong(slot) * 1024ul;
  for (uint k = t; k < 1024u; k += L_TG) {
    uint cs = CS[k], c = ((k >> 5) + 1u) * 34u + (k & 31u) + 1u;
    uchar kind = 0xff;
    if (!off && (cs & 16u)) kind = K[c];
    else if (!off && !((cs >> best) & 1u)) {   // kept open: a neighbour's liquid (one with a surface touches its corners)
      const uint nb[8] = { c - 1u, c + 1u, c - 34u, c + 34u, c - 35u, c - 33u, c + 33u, c + 35u };
      for (uint i = 0; i < 8u && kind == 0xff; i++) kind = K[nb[i]];
    }
    kd[k] = kind;
  }
  device packed_float3* vd = verts + ulong(slot) * ulong(L_VERTS);
  float hq = float(atomic_load_explicit(&hqA, memory_order_relaxed));
  for (uint v = t; v < uint(L_VERTS); v += L_TG) {
    float3 p = float3(o);                    // a collapsed slot: every vertex on one point
    if (!off) {
      uint s = lSrc(rowM, colM, v % 33u, v / 33u, best);
      if (s < L_LINE) p = float3(float(o.x) + float(s % 33u), VH[s], float(o.z) + float(s / 33u));
      else if (best < 2u) p = float3(float(o.x), hq, float(o.z) + float(s - L_LINE));
      else p = float3(float(o.x) + float(s - L_LINE), hq, float(o.z));
    }
    vd[v] = packed_float3(p);
  }
}
