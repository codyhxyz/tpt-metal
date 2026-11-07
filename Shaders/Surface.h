// Smooth surfaces: the CONTRACT between the four pieces that build them. Read-only for every piece; change it only
// by agreement. Included by World.metal after the storage and material helpers (one translation unit).
//
// The voxel bytes stay the truth (the simulation's data). Near the camera, the world is drawn from smooth meshes
// made from them (surface nets, NoCubes style: every material rounded) and ray traced in hardware. Far away, where a
// cell is about a pixel, the existing voxel ray walk draws it. Pieces and the files each one owns:
//   mesher    Shaders/Mesh.h, Sources/BigSand/Mesher.swift      solid surfaces of a list of mesh chunks -> staging
//   surfaces  Shaders/SurfaceJobs.h (new), Sources/BigSand/Surfaces.swift, the dirty marks in World.metal's physics
//             and edit kernels and their bindings in World.swift
//                                                             which chunks to mesh, BLAS per chunk, the TLAS, residency
//   liquids   Shaders/Liquid.h, Sources/BigSand/Liquids.swift   liquid surfaces as height fields, refit in place
//   render    Shaders/Render.h                                 primary rays through the TLAS + voxel walk, shading
//
// ---- mesh chunks ("mc"): 32^3 cells. mc m covers cells [32 m, 32 m + 32).
#define S_MC 32
#define S_MCX (NW / S_MC)            // mcs along x and along z
#define S_MCY (NH / S_MC)            // mcs along y
#define S_RADIUS 1024.0f             // surfaces are kept for mcs whose centre is within this distance (cells) of the camera
inline uint mcIdx(int3 m) { return (uint(m.z) * uint(S_MCX) + uint(m.x)) * uint(S_MCY) + uint(m.y); }
inline int3 mcOf(uint i) { uint y = i % uint(S_MCY), r = i / uint(S_MCY); return int3(r % uint(S_MCX), y, r / uint(S_MCX)); }
inline bool mcInWorld(int3 m) { return all(m >= 0) && m.x < S_MCX && m.y < S_MCY && m.z < S_MCX; }

// ---- which cells are surfaces of what
// Solid surface: between a solid cell (solid(m): not air, not liquid) and a non-solid one. Every solid material is
// rounded. An mc's solid mesh is the faces of the sample edges (between two neighbouring cells) whose lower cell lies
// in the mc; it may read cells up to S_HALO outside the mc on every side (the dual cubes around those edges, and
// smoothing), and nothing farther. Seams: a vertex both neighbours compute comes out bit-identical, so meshes meet
// without cracks. Dirty rule: a change to cell c dirties every mc whose box grown by S_HALO contains c.
// Liquid surface: the top of a liquid column (liquid cell with air above), per column of an mc (see Liquid.h). The
// same dirty rule, for the liquidDirty bits.
#define S_HALO 3
// Dirty bits (one per mc): solidDirty (set where a step or edit changed cells; cleared by the surfaces piece when it
// queues the mc) and liquidDirty (set where liquid moved or a step or edit changed cells; cleared by the liquids
// piece when it updates the mc's surface).

// ---- solid mesh triangles: per-triangle data stored in the acceleration structure as primitive data
// (MTLAccelerationStructureGeometryDescriptor.primitiveDataBuffer, stride and element size 16), read by the render
// through the intersection result's primitive data pointer.
struct SurfPrim {
  uint n0, n1, n2;   // smooth unit normals at the triangle's vertices 0, 1, 2 (octahedral, snorm 16+16)
  uint mat;          // bits 0-7: material id (MATS index, never a liquid); the rest reserved (0)
};
// Barycentrics from Metal (u, v) weight vertices 1 and 2; vertex 0 gets 1 - u - v.

// ---- mesher staging output (written by the mesher, read by the surfaces piece to build BLASes)
// Vertices: packed_float3 in world cell coordinates (cell c spans [c, c+1)). Indices: uint, relative to the chunk's
// first vertex. Prims: SurfPrim, one per triangle, in triangle order. Triangles wind counter-clockwise seen from
// outside the solid (the normal points out of the solid).
struct MeshOut {
  uint mc;           // mcIdx of the chunk
  uint triCount;     // triangles (0: no solid surface in this mc); S_RETRY: staging was full, mesh it again later
  uint vtxOffset;    // first vertex in the staging vertex buffer
  uint vtxCount;
  uint triOffset;    // first triangle: its indices start at 3 * triOffset, its prim at triOffset
  uint pad0, pad1, pad2;
};
#define S_RETRY 0xffffffffu
// Staging counters (one uint buffer, zeroed by the surfaces piece before each mesher run): [0] vertices handed out,
// [1] triangles handed out, [2] vertex capacity, [3] triangle capacity (capacities written by the surfaces piece).
#define S_SC_VTX 0
#define S_SC_TRI 1
#define S_SC_VCAP 2
#define S_SC_TCAP 3

// ---- acceleration structures
// One TLAS. Instances: one per mc with a non-empty solid mesh (mask S_MASK_SOLID, user id = mcIdx), one per liquid
// slot (mask S_MASK_LIQUID, user id = slot). Identity transforms: geometry is in world cell coordinates.
#define S_MASK_SOLID 1u
#define S_MASK_LIQUID 2u

// ---- coverage: what the meshes stand for. One bit per mc (bit i of word i / 32 is mcIdx i).
// solidCovered: the mc's solid surface is in the TLAS (or it has none): rays take solids there from the TLAS only.
// liquidCovered: the mc's liquid surfaces are in the TLAS (or it has none): rays take liquids there from the TLAS only.
// Where a bit is clear, the render walks the voxels for that kind of thing in that mc, as before.
inline bool bitSet(device const uint* bits, uint i) { return (bits[i >> 5] >> (i & 31u)) & 1u; }

// ---- liquid slots (liquids piece). Slot s is one mc's liquid surface: a 33 x 33 grid of vertices over its 32 x 32
// columns, refit in place as heights change.
#define L_SIDE 32
#define L_VERTS (33 * 33)
#define L_TRIS (2 * 32 * 32)
// Vertex (x, z) of slot s: lverts[s * L_VERTS + z * 33 + x], packed_float3, world coordinates.
// Column (x, z) (z * 32 + x) owns triangles 2c and 2c + 1:
//   2c:     v(x, z), v(x, z + 1), v(x + 1, z + 1)
//   2c + 1: v(x, z), v(x + 1, z + 1), v(x + 1, z)
// (counter-clockwise seen from above: normal up). Its liquid: lkinds[s * 1024 + c] = material id (MATS index,
// WATER..NMAT-1), or 0xff where the column has no liquid surface in this mc (its triangles are collapsed, never hit).

// ---- the render's bindings (set by Surfaces.bind; w_render's existing buffers 0..5 are unchanged)
#define S_BIND_TLAS 6         // instance_acceleration_structure
#define S_BIND_SCOVER 7       // device const uint*: solidCovered bits
#define S_BIND_LCOVER 8       // device const uint*: liquidCovered bits
#define S_BIND_LVERTS 9       // device const packed_float3*: liquid slot vertices
#define S_BIND_LKINDS 10      // device const uchar*: liquid slot kinds
#define S_BIND_INFO 11        // constant SurfInfo&
struct SurfInfo {
  uint enabled;      // 0: nothing is bound at 6..10 that may be touched; draw everything from voxels
  uint pad0, pad1, pad2;
};

// ---- octahedral unit normals (y up), snorm 16+16
inline uint octEncode(float3 n) {
  float3 v = n / (abs(n.x) + abs(n.y) + abs(n.z));
  float2 p = v.xz;
  if (v.y < 0.0f) p = (1.0f - abs(p.yx)) * select(float2(-1.0f), float2(1.0f), p >= 0.0f);
  return pack_float_to_snorm2x16(p);
}
inline float3 octDecode(uint u) {
  float2 p = unpack_snorm2x16_to_float(u);
  float3 v = float3(p.x, 1.0f - abs(p.x) - abs(p.y), p.y);
  float t = saturate(-v.y);
  v.x += v.x >= 0.0f ? -t : t;
  v.z += v.z >= 0.0f ? -t : t;
  return normalize(v);
}
