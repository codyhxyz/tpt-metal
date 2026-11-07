// C interface to the vendored binary greedy mesher (mesher.h), for Swift.
#ifndef TPT_MESHER_H
#define TPT_MESHER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// The mesher's fixed padded size: inputs are 64^3 cells (z fastest, then x, then y), meshes at most 62^3.
enum { BGM_PADDED = 64, BGM_MAX = 62 };

/// One mesher's scratch memory. Not thread-safe: one per thread.
typedef struct BGMContext BGMContext;

BGMContext* bgm_create(void);
void bgm_destroy(BGMContext* ctx);

/// Meshes one padded region.
/// voxels: 64^3 material bytes, index z + 64 x + 4096 y (padded coordinates; interior cells start at 1).
/// emit: 64 x 64 column masks (index 64 y + x, bit z) of the cells whose faces are drawn: interior cells only.
/// occluder: the same, for the cells that hide a neighbour's face, padding included.
/// bakeAO: 1 to bake corner ambient occlusion from `occluder`.
/// faceStart[6]: where each face's quads start (faces +y, -y, +x, -x, +z, -z). Returns the quad count.
/// The quads stay valid in bgm_quads until the next call on this context.
int bgm_mesh(BGMContext* ctx, const uint8_t* voxels, const uint64_t* emit, const uint64_t* occluder, int bakeAO,
             int faceStart[6]);
const uint64_t* bgm_quads(const BGMContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
