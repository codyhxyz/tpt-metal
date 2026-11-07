// Drawing the world: the rays, their shading, and the present pass. Included by World.metal (one translation unit).
// Owner: the render piece. Contract with the other pieces: Shaders/Surface.h.
// ---------------------------------------------------------------------------------------------------------------
// Rays. Near the camera the world is smooth meshes in a TLAS (Surface.h); a ray asks the TLAS for the nearest solid
// triangle, then for a liquid surface in front of it, then walks the voxels up to there for what the meshes don't
// stand for (mcs whose coverage bit is clear, and everything beyond the surface radius). Without surfaces it walks
// the voxels alone, as before.
//
// The walk: one loop walks all three levels: at each step it looks up the box containing the current cell (a whole
// chunk, a brick or one cell, whichever is uniform first) and, if that box is empty, jumps to where the ray leaves
// it. Water is a medium the ray passes through, not a wall: it remembers where it entered and how far it went.
#include <metal_raytracing>

struct WCam {
  float4 pos, fwd, right, up, sun;
  float tanHalf, aspect; uint shadows, stats;
  float2 jitter, pad;
};

// A walk's result. Cells are packed (x | z << 16, y | (axis + 1) << 16 | mat << 24) to keep the loop's registers few.
struct Walk { float t, tWater, wet; uint2 cell, wcell; uint steps; };
inline uint2 packHit(int3 c, int axis, uint mat) {
  return uint2(uint(c.x) | (uint(c.z) << 16), uint(c.y) | (uint(axis + 1) << 16) | (mat << 24));
}
inline int3 hitCell(uint2 p) { return int3(p.x & 0xffffu, p.y & 0xffffu, p.x >> 16); }
inline int hitAxis(uint2 p) { return int((p.y >> 16) & 255u) - 1; }
inline uint hitMat(uint2 p) { return p.y >> 24; }

// What the meshes stand for, for one walk. Within tCov of the camera the walk reads the coverage bits and leaves out
// what is covered; beyond it no mc is covered (an mc's cells lie within S_RADIUS + 28 of the camera when its centre
// is within S_RADIUS; the rest is slack for bits a frame behind). Clear liquid in [tw0, tw1) is counted from the
// liquid surfaces, so the walk leaves it out of its wet distance.
#define R_TCOV (S_RADIUS + 96.0f)
struct Cover { device const uint* s; device const uint* l; float tCov, tw0, tw1; };

// lodK: cells a pixel spans per unit of distance. Where a brick is smaller than a pixel, the ray stops walking its
// cells and takes it as one block: the top solid cell of the brick's column under the ray.
// COV: the surfaces are on. Off, this is the plain walk, with not a single instruction more.
template <bool COV>
inline Walk trace(float3 o, float3 d, float tMax, float lodK, device const uint* grid, device const uint* tables,
                  device const uchar* bricks, Cover cv) {
  Walk h; h.t = -1; h.tWater = -1; h.wet = 0; h.steps = 0; h.cell = uint2(0); h.wcell = uint2(0);
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
    uint m; int s;
    uint cov = 0u;                                    // bit 0: this mc's solids are in the meshes; bit 1: its liquids
    if (COV && t < cv.tCov) {
      uint mi = mcIdx(c >> 5), b = 1u << (mi & 31u);  // c >> 5: the mc (S_MC = 32)
      cov = ((cv.s[mi >> 5] & b) != 0u ? 1u : 0u) | ((cv.l[mi >> 5] & b) != 0u ? 2u : 0u);
    }
    if (COV && cov == 3u) { m = AIR; s = S_MC; }      // all of it is in the meshes: one jump across the mc
    else {
      uint ce = grid[chunkIdx(c >> 6)];
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
      // a kind of thing the meshes draw here is air to the walk; the box stops at the mc, where coverage may change
      if (COV && m != AIR && (cov & (isLiquid(m) ? 2u : 1u)) != 0u) { m = AIR; s = min(s, S_MC); }
    }
    if (!isClear(m)) { h.t = t; h.cell = packHit(c, axis, matOf(m)); return h; }
    int3 lo = c & ~(s - 1);
    float3 te = (float3(lo) + select(float3(0), float3(s), d > 0) - o) * inv;
    int k = te.x <= te.y && te.x <= te.z ? 0 : (te.y <= te.z ? 1 : 2);
    float tn2 = te[k];
    if (m != AIR) {                                   // a clear liquid
      if (h.tWater < 0) { h.tWater = t; h.wcell = packHit(c, i == 0u && axis != 1 ? 1 : axis, 0u); }
      float e = min(tn2, tExit);
      h.wet += e - t;
      if (COV) h.wet -= max(min(e, cv.tw1) - max(t, cv.tw0), 0.0f);
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

// What a ray saw. The surface it stops at (tHit < 0: none): its material, normal, the cell its colour varies by, and
// whether the normal is smooth (a mesh) or a voxel face. The clear liquid it went through on the way (tWater < 0:
// none): where it entered, the surface normal there, whether that surface reflects the sky, how far it went in liquid.
// For the crosshair: the cell and face (faceCode; -1: not pickable) of each.
struct Seen {
  float tHit; uint mat; float3 n; int3 hc; bool smooth;
  float tWater, wet; float3 wn; bool fres;
  int3 pickH, pickW; int faceH, faceW;
};
inline int faceCode(int3 nn) { return (nn.x + 1) + 3 * (nn.y + 1) + 9 * (nn.z + 1); }   // 13: no face
inline int axisFace(int k, float3 d) { int3 nn = int3(0); if (k >= 0) nn[k] = d[k] > 0 ? -1 : 1; return faceCode(nn); }
inline int domFace(float3 n) {                        // the face a normal mostly points out of
  float3 a = abs(n);
  int k = a.x >= a.y && a.x >= a.z ? 0 : (a.y >= a.z ? 1 : 2);
  int3 nn = int3(0); nn[k] = n[k] >= 0.0f ? 1 : -1; return faceCode(nn);
}

inline void walkHit(thread Seen& s, Walk h, float3 d) {
  int k = hitAxis(h.cell);
  s.tHit = h.t; s.hc = hitCell(h.cell); s.mat = hitMat(h.cell); s.smooth = false;
  s.n = float3(0);
  if (k >= 0) s.n[k] = d[k] > 0 ? -1.0f : 1.0f; else s.n = -d;
  s.pickH = s.hc; s.faceH = axisFace(k, d);
}
inline void walkWater(thread Seen& s, Walk h, float3 o, float3 d) {
  int3 wc = hitCell(h.wcell); int k = hitAxis(h.wcell);
  s.tWater = h.tWater;
  s.wn = float3(0, 1, 0);
  bool edge = wc.x == 0 || wc.z == 0 || wc.x == NW - 1 || wc.z == NW - 1;   // the sea at the world's rim
  if (k >= 0 && k != 1 && !edge) { s.wn = float3(0); s.wn[k] = d[k] > 0 ? -1.0f : 1.0f; }
  s.fres = o.y >= float(W_SEA) || k != 1;
  s.pickW = wc; s.faceW = axisFace(k, d);
}

// Material colour varied per cell (the variation fades where a cell is smaller than a pixel), lit by the sky and the
// sun. Voxel faces: Minecraft's per-side shade on the sky's light (top 1, north/south 0.8, east/west 0.6, bottom 0.5).
// Smooth surfaces: the same falloff from top to bottom, continuous in the normal, so rounded shapes read as round.
inline float3 shade(uint mat, int3 cell, float3 n, bool smooth, float footprint, float3 L, float3 sunCol) {
  float r = float(hash3(cell.x, cell.y, cell.z, 3u) & 1023u) / 1023.0f;
  float3 base = matColor(mat) * (1.0f + MATS[mat].var * (2.0f * r - 1.0f) / max(1.0f, footprint));
  float side = smooth ? 0.7f + n.y * (0.25f + 0.05f * n.y)
                      : (n.y > 0.5f ? 1.0f : (n.y < -0.5f ? 0.5f : (abs(n.x) > 0.5f ? 0.6f : 0.8f)));
  return base * (W_AMBIENT * side + sunCol * max(dot(n, L), 0.0f));
}

kernel void w_render(texture2d<float, access::write> out [[texture(0)]], texture2d<float, access::write> depthOut [[texture(1)]],
                     texture2d<float, access::write> motionOut [[texture(2)]],
                     device const uint* grid [[buffer(0)]], device const uint* tables [[buffer(1)]], device const uchar* bricks [[buffer(2)]],
                     constant WCam& C [[buffer(3)]], constant WCam& P [[buffer(4)]], device atomic_int* ctr [[buffer(5)]],
                     raytracing::instance_acceleration_structure tlas [[buffer(S_BIND_TLAS)]],
                     device const uint* scover [[buffer(S_BIND_SCOVER)]], device const uint* lcover [[buffer(S_BIND_LCOVER)]],
                     device const packed_float3* lverts [[buffer(S_BIND_LVERTS)]], device const uchar* lkinds [[buffer(S_BIND_LKINDS)]],
                     constant SurfInfo& SI [[buffer(S_BIND_INFO)]],
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
  float tLim = seaOut ? tSea : 1e30f, lodK = 2.0f * C.tanHalf / float(H);
  bool center = g.x == W / 2 && g.y == H / 2;
  Seen s; s.tHit = -1; s.tWater = -1; s.wet = 0; s.fres = false; s.smooth = false; s.mat = AIR;
  s.n = float3(0); s.wn = float3(0, 1, 0); s.hc = int3(0); s.pickH = int3(0); s.pickW = int3(0); s.faceH = 13; s.faceW = 13;
  uint steps;
  if (SI.enabled == 0u) {
    Walk h = trace<false>(o, d, tLim, lodK, grid, tables, bricks, Cover{nullptr, nullptr, 0.0f, 0.0f, 0.0f});
    steps = h.steps;
    if (h.t >= 0) walkHit(s, h, d);
    if (h.tWater >= 0) { walkWater(s, h, o, d); s.wet = h.wet; }
  } else {
    raytracing::ray r(o, d, 0.0f, tLim);
    raytracing::intersector<raytracing::triangle_data, raytracing::instancing> X;
    X.assume_geometry_type(raytracing::geometry_type::triangle);
    X.force_opacity(raytracing::forced_opacity::opaque);
    X.assume_identity_transforms(true);
    X.accept_any_intersection(false);
    // 1. the nearest solid triangle
    auto qs = X.intersect(r, tlas, S_MASK_SOLID);
    float tS = qs.type == raytracing::intersection_type::triangle ? qs.distance : 1e30f;
    const device SurfPrim* sp = (const device SurfPrim*)qs.primitive_data;
    float2 sb = qs.triangle_barycentric_coord;
    // 2. the nearest liquid surface in front of it; its normal from the triangle's corners (Surface.h's layout)
    r.max_distance = min(tS, tLim);
    auto ql = X.intersect(r, tlas, S_MASK_LIQUID);
    float tL = 1e30f; float3 ln = float3(0, 1, 0); uint lk = WATER;
    if (ql.type == raytracing::intersection_type::triangle) {
      uint slot = ql.user_instance_id, pid = ql.primitive_id, col = pid >> 1;
      uint v = slot * uint(L_VERTS) + (col >> 5) * 33u + (col & 31u);
      float3 v0 = float3(lverts[v]), v1 = float3(lverts[v + ((pid & 1u) ? 34u : 33u)]), v2 = float3(lverts[v + ((pid & 1u) ? 1u : 34u)]);
      ln = normalize(cross(v1 - v0, v2 - v0));
      lk = lkinds[slot * 1024u + col];
      lk = lk < uint(NMAT) ? lk : uint(WATER);
      tL = ql.distance;
    }
    bool fromAbove = tL < 1e30f && dot(d, ln) < 0.0f;
    // What the meshes' liquid is to this ray (lk: its material, or the camera's liquid):
    //   0 nothing, 1 the camera is in clear liquid, 2 it enters clear liquid at tL, 3 it stops on opaque liquid at
    //   tL, 4 the camera is in opaque liquid. The walk can't see liquid the meshes stand for, so a camera in it is
    //   found here; a camera whose ray meets the surface from above is above it (in the top cell's empty part).
    uint mode = 0u;
    int3 co = int3(floor(o));
    if (!fromAbove && inWorld(co) && bitSet(lcover, mcIdx(co >> 5))) {
      uint cam = cellAt(co, grid, tables, bricks);
      if (isLiquid(cam)) { lk = matOf(cam); mode = isClear(cam) ? 1u : 4u; }
    } else if (fromAbove) mode = MATS[lk].clear != 0 ? 2u : 3u;
    float tOpq = mode == 4u ? 0.0f : (mode == 3u ? tL : tS);   // the meshes' opaque surface
    float tw0 = mode == 1u ? 0.0f : (mode == 2u ? tL : 1e30f);  // the meshes' clear liquid: [tw0, tw1)
    float tw1 = mode == 1u && tL < 1e30f ? tL : 1e30f;         // underwater, up to the surface above
    // 3. the voxels the meshes don't stand for, up to the meshes' opaque surface: a nearer voxel wins
    Walk h = trace<true>(o, d, min(tLim, tOpq), lodK, grid, tables, bricks, Cover{scover, lcover, R_TCOV, tw0, tw1});
    steps = h.steps;
    if (h.t >= 0) walkHit(s, h, d);
    else if (mode == 4u) {                            // inside opaque liquid: it is all the ray sees
      s.tHit = 0; s.mat = lk; s.n = -d; s.hc = co; s.pickH = co;
    } else if (tOpq < 1e30f) {
      float3 p = o + d * tOpq;
      if (mode == 3u) { s.mat = lk; s.n = ln; s.hc = int3(floor(p - float3(0, 0.5f, 0))); }   // the liquid cell under it
      else {
        SurfPrim q = *sp;                             // smooth normal: the corners' normals by barycentrics
        s.n = normalize((1.0f - sb.x - sb.y) * octDecode(q.n0) + sb.x * octDecode(q.n1) + sb.y * octDecode(q.n2));
        s.mat = min(q.mat & 255u, uint(NMAT) - 1u);
        s.hc = int3(floor(p - s.n * 0.5f));           // the solid cell just inside the surface
      }
      s.tHit = tOpq; s.smooth = true; s.pickH = s.hc; s.faceH = domFace(s.n);
    }
    // the meshes' clear liquid counts only in front of the surface the ray stops at
    float wetM = 0.0f;
    if (tw0 < 1e30f && (s.tHit < 0 || tw0 < s.tHit)) {
      wetM = min(tw1, s.tHit >= 0 ? s.tHit : 1e30f) - tw0;
      s.tWater = tw0;
      if (mode == 1u) { s.fres = false; s.pickW = co; s.faceW = axisFace(1, d); }
      else { s.wn = ln; s.fres = true; s.pickW = int3(floor(o + d * tL - float3(0, 0.5f, 0))); s.faceW = domFace(ln); }
    }
    if (h.tWater >= 0 && (s.tWater < 0 || h.tWater < s.tWater)) walkWater(s, h, o, d);
    s.wet = h.wet + wetM;
  }
  float3 col; float tt;
  if (s.tHit >= 0) {
    col = shade(s.mat, s.hc, s.n, s.smooth, s.tHit * 2.0f * C.tanHalf / float(H), L, sunCol);
    tt = s.tHit;
  } else {
    col = skyColor(d, L);
    tt = 1e30f;
    // the endless sea around the island
    if (seaOut && s.tWater < 0) {
      // over a sand floor 60 cells down, like the world's own sea far from the island
      s.tWater = tSea; s.wn = float3(0, 1, 0); s.fres = true; s.faceW = -1; s.wet = 60.0f / max(-d.y, 0.05f); tt = tSea;
      col = matColor(SAND) * (W_AMBIENT + sunCol * max(L.y, 0.0f));
    }
  }
  if (s.tWater >= 0) {
    // light through water: absorbed per cell travelled (red first), then the surface reflects the sky (Fresnel)
    float3 sigma = float3(0.11, 0.035, 0.018);
    float3 tr = exp(-sigma * s.wet);
    float3 deep = float3(0.02, 0.10, 0.16);
    col = col * tr + deep * (1.0f - tr);
    if (s.fres) {
      float cosv = saturate(-dot(d, s.wn));
      float F = 0.02f + 0.98f * pow(1.0f - cosv, 5.0f);
      col = mix(col, skyColor(reflect(d, s.wn), L), F);
    }
    if (s.tHit < 0) tt = s.tWater;
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
    bool wetFirst = s.tWater >= 0 && s.faceW >= 0 && (s.tHit < 0 || s.tWater < s.tHit);
    bool got = wetFirst || s.tHit >= 0;
    int3 pc = wetFirst ? s.pickW : s.pickH;
    atomic_store_explicit(&ctr[C_PICK], pc.x, memory_order_relaxed);
    atomic_store_explicit(&ctr[C_PICK + 1], pc.y, memory_order_relaxed);
    atomic_store_explicit(&ctr[C_PICK + 2], pc.z, memory_order_relaxed);
    atomic_store_explicit(&ctr[C_PICK + 3], got ? 1 + (wetFirst ? s.faceW : s.faceH) : 0, memory_order_relaxed);
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
