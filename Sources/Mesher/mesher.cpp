// The C interface (include/TPTMesher.h) around the vendored mesher.
#include <string.h>
#include <vector>
#define BM_IMPLEMENTATION
#include "mesher.h"
#include "TPTMesher.h"

struct BGMContext {
  std::vector<uint64_t> faceMasks = std::vector<uint64_t>(CS_2 * 6);
  std::vector<uint64_t> emit = std::vector<uint64_t>(CS_P2);
  std::vector<uint8_t> forwardMerged = std::vector<uint8_t>(CS_2);
  std::vector<uint8_t> rightMerged = std::vector<uint8_t>(CS);
  std::vector<uint64_t> quads = std::vector<uint64_t>(16384);
  MeshData data;
};

extern "C" BGMContext* bgm_create(void) {
  BGMContext* c = new BGMContext();
  c->data.faceMasks = c->faceMasks.data();
  c->data.forwardMerged = c->forwardMerged.data();
  c->data.rightMerged = c->rightMerged.data();
  c->data.vertices = &c->quads;
  c->data.maxVertices = (int)c->quads.size();
  return c;
}

extern "C" void bgm_destroy(BGMContext* c) { delete c; }

extern "C" int bgm_mesh(BGMContext* c, const uint8_t* voxels, const uint64_t* emit, const uint64_t* occluder, int bakeAO,
                        int faceStart[6]) {
  // mesh() only reads opaqueMask; the cast is for its non-const declaration.
  c->data.opaqueMask = const_cast<uint64_t*>(emit);
  c->data.occluderMask = occluder;
  c->data.bakeAO = bakeAO != 0;
  // The merge counters must start at zero; mesh() leaves them at zero when it returns.
  mesh(voxels, c->data);
  int n = 0;
  for (int f = 0; f < 6; f++) { faceStart[f] = c->data.faceVertexBegin[f]; n += c->data.faceVertexLength[f]; }
  return n;
}

extern "C" const uint64_t* bgm_quads(const BGMContext* c) { return c->quads.data(); }
