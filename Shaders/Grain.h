// Grain detail: the grains of a powder's surface. Included by World.metal before Render.h (one translation unit).
// ---------------------------------------------------------------------------------------------------------------
// CONTRACT (the swap point). Render.h calls grainDetail() once per surface hit, between finding the hit and shade():
// GrainIn is the hit, GrainOut is what shade() lights. This version shades grains from a tile baked at launch and never
// moves the hit (dt = 0); a geometric version (grains traced near the camera) may return dt, the change in hit
// distance, and Render.h already moves its depth and motion with it. The bindings (G_TEX_*, G_BIND_FRAME) and
// GrainFrame belong to Sources/BigSand/Grains.swift.
//
// The tile is a patch of real grains, stored as what the shading needs, with a mip chain (LEAN mapping: Olano & Baker
// 2010). Slopes and their squares average linearly, so the hardware's mip averaging is exact for them, and a pixel
// that covers many grains gets their mean normal plus their spread as roughness. One read per map per pixel; no
// branches but the material test.
//
// Every number comes from a measured property of the material or from the engine:
//   size      Wentworth (1922) medium sand, sieve sizes 0.25-0.5 mm (typical beach sand). Sizes are log-normal, so the
//             median is the class's geometric centre. A sieve measures a grain's intermediate axis.
//   sorting   Folk & Ward (1957) "well sorted", inclusive graphic standard deviation 0.35-0.50 phi (typical beach
//             sand); the class's centre.
//   form      Zingg (1935) spheroid class: b/a and c/b each in [2/3, 1]. A resting grain lies on its short axis.
//   packing   grains cover a surface like randomly packed discs: area fraction 0.82 (Berryman 1983, 2D random close
//             packing). One grain per jittered site, so the site spacing follows from the mean grain area.
//   texel     Nyquist: two texels across the class's smallest grain.
//   tile      enough grains that the tile's mean albedo (the top of its mip chain) is the material's to within one
//             8-bit output step: std(mean of k grains) = var / sqrt(3k) <= 1/255, var = MATS[].var (a grain's albedo
//             is uniform in [1 - var, 1 + var], as a cell's is). Rounded up to a power of two texels (a whole chain).
//   horizons  8 directions, 45 degrees apart (two RGBA8 textures). Each is searched out until the tallest grain could
//             no longer raise the horizon by one 8-bit step of its encoding ((pi/2) / 255).
//   occlusion the unoccluded share of the cosine-weighted sky: mean over directions of cos^2(horizon elevation).
//   sun       the sun's angular diameter, 0.53 degrees, sets the width of a grain's shadow edge.

#define G_TEX_A 3            // texture2d<float> rgba16f: slopes bx, by (dh/du, dh/dv) and bx^2, by^2
#define G_TEX_B 4            // texture2d<float> rgba16f: bx*by, ambient occlusion, albedo deviation [-1, 1], height (m)
#define G_TEX_H0 5           // texture2d<float> rgba8: horizon elevation / (pi/2) toward 0, 45, 90, 135 degrees
#define G_TEX_H1 6           // texture2d<float> rgba8: toward 180, 225, 270, 315 degrees
#define G_BIND_FRAME 12      // constant GrainFrame&
#define G_DIRS 8

constant float G_D_LO = 0.25e-3f, G_D_HI = 0.5e-3f;    // medium sand, m
constant float G_SORT_LO = 0.35f, G_SORT_HI = 0.50f;   // well sorted, phi
constant float G_ZINGG = 2.0f / 3.0f;                  // Zingg's class boundary
constant float G_PACK2D = 0.82f;                       // 2D random close packing
constant float G_STEP8 = 1.0f / 255.0f;                // one 8-bit step
constant float G_SUN_DIAM = 0.53f * M_PI_F / 180.0f;   // rad
constant uint G_SEED = 0x6772u;                        // the tile's hash stream

// What the bake works out (g_plan), and what Grains.swift allocates from.
struct GrainPlan {
  uint texels, sites, window, levels;   // tile side in texels; grain sites across; sites searched each way; mip levels
  float tileM, texelM, siteM, d50;      // tile side, texel, site spacing, median grain size (m)
  float sigLn, zMax, pad0, pad1;        // log-normal sigma of size; largest normal deviate the hashes can give
};

// What the render reads, per frame (Grains.swift fills it).
// org: the camera in each projection plane's lattice coordinates (plane 0: z, y; 1: x, z; 2: x, y), reduced modulo
// `period` in double precision on the CPU, so the shader's coordinates stay small and exact near the camera.
struct GrainFrame {
  float4 org01;                         // plane 0 (s, t), plane 1 (s, t)
  float4 org2;                          // plane 2 (s, t), output pixels per traced pixel, 0
  float tileM, texelM; uint levels, period;
};

struct GrainRes { texture2d<float> a, b, h0, h1; constant GrainFrame* f; };

struct GrainIn {
  float3 rel;       // hit relative to the camera, cells
  float3 n;         // surface normal
  float3 v;         // toward the eye
  float3 L;         // toward the sun
  float t;          // hit distance, cells
  float pixK;       // cells a traced pixel spans per cell of distance
  uint mat, rnd;    // material; 32 random bits for this pixel and frame
};
struct GrainOut {
  float3 n;         // shading normal
  float albedo;     // albedo deviation: the colour is matColor * (1 + albedo)
  float rough;      // slope spread under the pixel (Beckmann roughness it adds)
  float occl;       // sky light reaching the surface, 0..1
  float sun;        // sunlight past the grains, 0..1
  float dt;         // change in hit distance, cells
  bool on;          // false: not a grainy material; shade() uses its own colour variation
};

inline GrainOut grainNone(float3 n) {
  GrainOut o; o.n = n; o.albedo = 0.0f; o.rough = 0.0f; o.occl = 1.0f; o.sun = 1.0f; o.dt = 0.0f; o.on = false;
  return o;
}

inline GrainOut grainDetail(GrainIn gi, GrainRes R) {
  return grainNone(gi.n);
}

// --- bake (Grains.swift runs it once at launch: g_plan, then g_height, then g_derive, then mipmaps)

inline float gU(uint h) { return float(h >> 8) * (1.0f / 16777216.0f); }   // [0, 1), 24 bits

kernel void g_plan(device GrainPlan& P [[buffer(0)]], uint t [[thread_position_in_grid]]) {
  if (t != 0u) return;
  float d50 = sqrt(G_D_LO * G_D_HI);
  float sigLn = 0.5f * (G_SORT_LO + G_SORT_HI) * M_LN2_F;    // phi is log2 of size
  float ed2 = d50 * d50 * exp(2.0f * sigLn * sigLn);         // E[d^2] of a log-normal size
  float site = sqrt(M_PI_F * ed2 / (4.0f * G_PACK2D));       // one mean grain per site covers G_PACK2D of it
  float texel = 0.5f * G_D_LO;
  float sd = MATS[SAND].var / sqrt(3.0f);                    // std of one grain's albedo factor
  float k = ceil((sd / G_STEP8) * (sd / G_STEP8));           // grains whose mean is within one step
  float need = ceil(sqrt(k)) * site;                         // tile side holding them
  uint texels = 1u;
  while (float(texels) * texel < need) texels <<= 1;
  float tile = float(texels) * texel;
  uint sites = uint(round(tile / site));                     // whole sites across: the tile wraps
  float siteM = tile / float(sites);
  float zMax = sqrt(-2.0f * log(1.0f / 16777216.0f));        // Box-Muller from 24-bit uniforms
  float aMax = 0.5f * d50 * exp(sigLn * zMax) / G_ZINGG;     // longest semi-axis there can be
  P.texels = texels; P.sites = sites; P.window = uint(ceil(aMax / siteM)) + 1u; P.levels = ctz(texels) + 1u;
  P.tileM = tile; P.texelM = texel; P.siteM = siteM; P.d50 = d50;
  P.sigLn = sigLn; P.zMax = zMax; P.pad0 = 0.0f; P.pad1 = 0.0f;
}

// The tile's height field: the top of every grain over each texel centre (grains rest on a common bed, each centred
// one short semi-axis above it), and the albedo deviation of the grain on top. hmax: the tallest point (float bits:
// positive floats order as their bits do).
kernel void g_height(texture2d<float, access::write> HU [[texture(0)]], constant GrainPlan& P [[buffer(0)]],
                     device atomic_uint* hmax [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= P.texels || g.y >= P.texels) return;
  float2 p = (float2(g) + 0.5f) * P.texelM;
  int2 c = int2(floor(p / P.siteM));
  int W = int(P.window), n = int(P.sites);
  float top = 0.0f, alb = 0.0f;
  for (int dy = -W; dy <= W; dy++) {
    for (int dx = -W; dx <= W; dx++) {
      int2 s = c + int2(dx, dy), w = ((s % n) + n) % n;     // the site, wrapped
      float2 q = (float2(s) + float2(gU(hash3(w.x, w.y, 0, G_SEED)), gU(hash3(w.x, w.y, 1, G_SEED)))) * P.siteM - p;
      float u1 = (float(hash3(w.x, w.y, 2, G_SEED) >> 8) + 1.0f) * (1.0f / 16777216.0f);
      float z = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * gU(hash3(w.x, w.y, 3, G_SEED)));
      float b = 0.5f * P.d50 * exp(P.sigLn * z);            // intermediate semi-axis
      float a = b / mix(G_ZINGG, 1.0f, gU(hash3(w.x, w.y, 4, G_SEED)));
      float cz = b * mix(G_ZINGG, 1.0f, gU(hash3(w.x, w.y, 5, G_SEED)));
      float th = M_PI_F * gU(hash3(w.x, w.y, 6, G_SEED)), cs = cos(th), sn = sin(th);
      float2 r = float2(cs * q.x + sn * q.y, cs * q.y - sn * q.x);
      float e = (r.x * r.x) / (a * a) + (r.y * r.y) / (b * b);
      if (e < 1.0f) {
        float zt = cz * (1.0f + sqrt(1.0f - e));
        if (zt > top) { top = zt; alb = hval(hash3(w.x, w.y, 7, G_SEED)); }
      }
    }
  }
  HU.write(float4(top, alb, 0.0f, 0.0f), g);
  atomic_fetch_max_explicit(hmax, as_type<uint>(top), memory_order_relaxed);
}

inline float gH(texture2d<float, access::read> HU, int2 p, int n) { return HU.read(uint2(((p % n) + n) % n)).x; }

// From the height field: slopes (central differences over a texel), horizons, occlusion.
kernel void g_derive(texture2d<float, access::read> HU [[texture(0)]], texture2d<float, access::write> A [[texture(1)]],
                     texture2d<float, access::write> B [[texture(2)]], texture2d<float, access::write> H0 [[texture(3)]],
                     texture2d<float, access::write> H1 [[texture(4)]], constant GrainPlan& P [[buffer(0)]],
                     device const uint* hmax [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= P.texels || g.y >= P.texels) return;
  int n = int(P.texels); int2 c = int2(g);
  float2 hu = HU.read(g).xy;
  float h0 = hu.x;
  float bx = (gH(HU, c + int2(1, 0), n) - gH(HU, c - int2(1, 0), n)) / (2.0f * P.texelM);
  float by = (gH(HU, c + int2(0, 1), n) - gH(HU, c - int2(0, 1), n)) / (2.0f * P.texelM);
  float rise = as_type<float>(hmax[0]) - h0, tanStep = tan(0.5f * M_PI_F * G_STEP8);
  float hor[G_DIRS]; float ao = 0.0f;
  for (int j = 0; j < G_DIRS; j++) {
    float ang = float(j) * (2.0f * M_PI_F / float(G_DIRS));
    float2 dir = float2(cos(ang), sin(ang));
    float t = 0.0f;
    for (int s = 1; rise / (float(s) * P.texelM) > max(t, tanStep); s++) {
      // bilinear between the four texels around the sample point
      float2 q = float2(c) + dir * float(s);
      int2 q0 = int2(floor(q)); float2 f = q - float2(q0);
      float hs = mix(mix(gH(HU, q0, n), gH(HU, q0 + int2(1, 0), n), f.x),
                     mix(gH(HU, q0 + int2(0, 1), n), gH(HU, q0 + int2(1, 1), n), f.x), f.y);
      t = max(t, (hs - h0) / (float(s) * P.texelM));
    }
    float al = atan(t);
    hor[j] = al / (0.5f * M_PI_F);
    ao += cos(al) * cos(al);
  }
  A.write(float4(bx, by, bx * bx, by * by), g);
  B.write(float4(bx * by, ao / float(G_DIRS), hu.y, h0), g);
  H0.write(float4(hor[0], hor[1], hor[2], hor[3]), g);
  H1.write(float4(hor[4], hor[5], hor[6], hor[7]), g);
}
