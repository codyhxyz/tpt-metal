/*
MIT License

Copyright (c) 2020 Erik Johansson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

//   Define BM_IMPLEMENTATION in exactly one source file to include the implementation
//   and include the mesher.h without that define as often as needed
//
//   it could look like this
//
//   #define BM_IMPLEMENTATION
//   #include "mesher.h"
//
//   Vendored into tpt-metal from https://github.com/cgerikj/binary-greedy-meshing (v2, commit af8c688), MIT.
//   tpt-metal changes, each marked "tpt-metal":
//   * MeshData.occluderMask: the cells that hide a neighbour's face, separate from the cells that emit faces
//     (opaqueMask). Padding cells then never emit, and water can be meshed against water and solids.
//   * MeshData.bakeAO: v1.0.0's corner ambient occlusion, 2 bits per corner, read from occluderMask. Faces merge
//     only when their corners match and stay constant along the merge, so merged quads shade like single faces.
//   * Quad bits 40-47: corner AO (corner k = wMod << 1 | hMod at bits 40 + 2k), 48-50: face, 51: which diagonal
//     to split the quad on (v1's anisotropy fix).
//
//   There are other defines to control the behaviour of the library.
//   * Define BM_VECTOR with your own vector implementation - otherwise it will use std::vector

#ifndef MESHER_H
#define MESHER_H

#ifndef BM_VECTOR
#include <vector>
#define BM_VECTOR std::vector
#endif

#include <stdint.h>

// CS = chunk size (max 62)
static constexpr int CS = 62;

// Padded chunk size
static constexpr int CS_P = CS + 2;
static constexpr int CS_2 = CS * CS;
static constexpr int CS_P2 = CS_P * CS_P;
static constexpr int CS_P3 = CS_P * CS_P * CS_P;

struct MeshData {
  uint64_t* faceMasks = nullptr; // CS_2 * 6
  uint64_t* opaqueMask = nullptr; //CS_P2
  const uint64_t* occluderMask = nullptr; // CS_P2, tpt-metal: nullptr means opaqueMask
  bool bakeAO = false; // tpt-metal
  uint8_t* forwardMerged = nullptr; // CS_2
  uint8_t* rightMerged = nullptr; // CS
  BM_VECTOR<uint64_t>* vertices = nullptr;
  int vertexCount = 0;
  int maxVertices = 0;
  int faceVertexBegin[6] = { 0 };
  int faceVertexLength[6] = { 0 };
};

// @param[in] voxels: The input data includes duplicate edge data from neighboring chunks which is used
// for visibility culling. For optimal performance, your world data should already be structured
// this way so that you can feed the data straight into this algorithm.
// Input data is ordered in ZXY and is 64^3 which results in a 62^3 mesh.
//
// @param[out] meshData The allocated vertices in MeshData with a length of meshData.vertexCount.
void mesh(const uint8_t* voxels, MeshData& meshData);

#endif // MESHER_H

#ifdef BM_IMPLEMENTATION

#ifndef BM_MEMSET
#define BM_MEMSET memset
#include <string.h> // memset
#endif

static inline const int getAxisIndex(const int axis, const int a, const int b, const int c) {
  if (axis == 0) return b + (a * CS_P) + (c * CS_P2);
  else if (axis == 1) return b + (c * CS_P) + (a * CS_P2);
  else return c + (a * CS_P) + (b * CS_P2);
}

static inline const void insertQuad(BM_VECTOR<uint64_t>& vertices, uint64_t quad, int& vertexI, int& maxVertices) {
  if (vertexI >= maxVertices - 6) {
    vertices.resize(maxVertices * 2, 0);
    maxVertices *= 2;
  }

  vertices[vertexI] = quad;

  vertexI++;
}

static inline const uint64_t getQuad(uint64_t x, uint64_t y, uint64_t z, uint64_t w, uint64_t h, uint64_t type) {
  return (type << 32) | (h << 24) | (w << 18) | (z << 12) | (y << 6) | x;
}

constexpr uint64_t P_MASK = ~(1ull << 63 | 1);

// tpt-metal: corner ambient occlusion (v1.0.0's rule), for the face of padded cell (x, y, z) facing `face`.
// Corner k = wMod << 1 | hMod, in the directions the vertex shader expands a quad: w along (face & 2) >> 1,
// h along 2 - (face >> 2), w mirrored on faces whose flipLookup is -1.
static constexpr int BM_FLIP[6] = { 1, -1, -1, 1, -1, 1 };
static constexpr int BM_NORMAL_AXIS[6] = { 1, 1, 0, 0, 2, 2 };

static inline int bmSolid(const uint64_t* occ, int x, int y, int z) {
  return (int)(occ[(y * CS_P) + x] >> z & 1);
}

static inline uint8_t bmVertexAO(int side1, int side2, int corner) {
  return (side1 && side2) ? 0 : (uint8_t)(3 - (side1 + side2 + corner));
}

static inline uint8_t bmFaceAO(const uint64_t* occ, int x, int y, int z, int face) {
  int c[3] = { x, y, z };
  c[BM_NORMAL_AXIS[face]] += (face & 1) ? -1 : 1;
  const int wDir = (face & 2) >> 1, hDir = 2 - (face >> 2);
  uint8_t key = 0;
  for (int k = 0; k < 4; k++) {
    const int sw = (((k >> 1) != 0) != (BM_FLIP[face] < 0)) ? 1 : -1, sh = (k & 1) ? 1 : -1;
    int p1[3] = { c[0], c[1], c[2] }, p2[3] = { c[0], c[1], c[2] }, p3[3] = { c[0], c[1], c[2] };
    p1[wDir] += sw; p2[hDir] += sh; p3[wDir] += sw; p3[hDir] += sh;
    const int s1 = bmSolid(occ, p1[0], p1[1], p1[2]), s2 = bmSolid(occ, p2[0], p2[1], p2[2]);
    const int cr = !s1 && !s2 && bmSolid(occ, p3[0], p3[1], p3[2]);
    key |= bmVertexAO(s1, s2, cr) << (2 * k);
  }
  return key;
}

// Padded cell coordinates of getAxisIndex(axis, a, b, c).
static inline void bmAxisCoords(const int axis, const int a, const int b, const int c, int& x, int& y, int& z) {
  if (axis == 0) { z = b; x = a; y = c; }
  else if (axis == 1) { z = b; x = c; y = a; }
  else { z = c; x = a; y = b; }
}

static inline uint8_t bmAO(const MeshData& m, const int axis, const int face, const int a, const int b, const int c) {
  if (!m.bakeAO) return 0xff;
  int x, y, z;
  bmAxisCoords(axis, a, b, c, x, y, z);
  return bmFaceAO(m.occluderMask ? m.occluderMask : m.opaqueMask, x, y, z, face);
}

// Same corners, and constant along the merge direction (alongW: the quad's w axis, else its h axis).
static inline bool bmAOMerges(uint8_t k1, uint8_t k2, bool alongW) {
  if (k1 != k2) return false;
  if (alongW) return (k1 & 3) == (k1 >> 4 & 3) && (k1 >> 2 & 3) == (k1 >> 6 & 3);
  return (k1 & 3) == (k1 >> 2 & 3) && (k1 >> 4 & 3) == (k1 >> 6 & 3);
}

static inline uint64_t bmQuadExtra(uint8_t ao, int face) {
  const int a0 = ao & 3, a1 = ao >> 2 & 3, a2 = ao >> 4 & 3, a3 = ao >> 6 & 3;
  const uint64_t flip = a0 + a3 > a1 + a2 ? 1 : 0;
  return ((uint64_t)ao << 40) | ((uint64_t)face << 48) | (flip << 51);
}

void mesh(const uint8_t* voxels, MeshData& meshData) {
  meshData.vertexCount = 0;
  int vertexI = 0;

  uint64_t* opaqueMask = meshData.opaqueMask;
  uint64_t* faceMasks = meshData.faceMasks;
  uint8_t* forwardMerged = meshData.forwardMerged;
  uint8_t* rightMerged = meshData.rightMerged;
  const uint64_t* occ = meshData.occluderMask ? meshData.occluderMask : opaqueMask; // tpt-metal

  // Hidden face culling
  for (int a = 1; a < CS_P - 1; a++) {
    const int aCS_P = a * CS_P;

    for (int b = 1; b < CS_P - 1; b++) {
      const uint64_t columnBits = opaqueMask[(a * CS_P) + b] & P_MASK;
      const int baIndex = (b - 1) + (a - 1) * CS;
      const int abIndex = (a - 1) + (b - 1) * CS;

      faceMasks[baIndex + 0 * CS_2] = (columnBits & ~occ[aCS_P + CS_P + b]) >> 1;
      faceMasks[baIndex + 1 * CS_2] = (columnBits & ~occ[aCS_P - CS_P + b]) >> 1;

      faceMasks[abIndex + 2 * CS_2] = (columnBits & ~occ[aCS_P + (b + 1)]) >> 1;
      faceMasks[abIndex + 3 * CS_2] = (columnBits & ~occ[aCS_P + (b - 1)]) >> 1;

      faceMasks[baIndex + 4 * CS_2] = columnBits & ~(occ[aCS_P + b] >> 1);
      faceMasks[baIndex + 5 * CS_2] = columnBits & ~(occ[aCS_P + b] << 1);
    }
  }

  // Greedy meshing faces 0-3
  for (int face = 0; face < 4; face++) {
    const int axis = face / 2;

    const int faceVertexBegin = vertexI;

    for (int layer = 0; layer < CS; layer++) {
      const int bitsLocation = layer * CS + face * CS_2;

      for (int forward = 0; forward < CS; forward++) {
        uint64_t bitsHere = faceMasks[forward + bitsLocation];
        if (bitsHere == 0) continue;

        const uint64_t bitsNext = forward + 1 < CS ? faceMasks[(forward + 1) + bitsLocation] : 0;

        uint8_t rightMerged = 1;
        while (bitsHere) {
          unsigned long bitPos;
          #ifdef _MSC_VER
            _BitScanForward64(&bitPos, bitsHere);
          #else
            bitPos = __builtin_ctzll(bitsHere);
          #endif

          const uint8_t type = voxels[getAxisIndex(axis, forward + 1, bitPos + 1, layer + 1)];
          const uint8_t ao = bmAO(meshData, axis, face, forward + 1, bitPos + 1, layer + 1); // tpt-metal
          uint8_t& forwardMergedRef = forwardMerged[bitPos];

          if ((bitsNext >> bitPos & 1) && type == voxels[getAxisIndex(axis, forward + 2, bitPos + 1, layer + 1)]
              && bmAOMerges(ao, bmAO(meshData, axis, face, forward + 2, bitPos + 1, layer + 1), true)) {
            forwardMergedRef++;
            bitsHere &= ~(1ull << bitPos);
            continue;
          }

          for (int right = bitPos + 1; right < CS; right++) {
            if (!(bitsHere >> right & 1) || forwardMergedRef != forwardMerged[right] || type != voxels[getAxisIndex(axis, forward + 1, right + 1, layer + 1)]
                || !bmAOMerges(ao, bmAO(meshData, axis, face, forward + 1, right + 1, layer + 1), false)) break;
            forwardMerged[right] = 0;
            rightMerged++;
          }
          bitsHere &= ~((1ull << (bitPos + rightMerged)) - 1);

          const uint8_t meshFront = forward - forwardMergedRef;
          const uint8_t meshLeft = bitPos;
          const uint8_t meshUp = layer + (~face & 1);

          const uint8_t meshWidth = rightMerged;
          const uint8_t meshLength = forwardMergedRef + 1;

          forwardMergedRef = 0;
          rightMerged = 1;

          uint64_t quad;
          switch (face) {
          case 0:
          case 1:
            quad = getQuad(meshFront + (face == 1 ? meshLength : 0), meshUp, meshLeft, meshLength, meshWidth, type);
            break;
          case 2:
          case 3:
            quad = getQuad(meshUp, meshFront + (face == 2 ? meshLength : 0), meshLeft, meshLength, meshWidth, type);
            break;
          }
          quad |= bmQuadExtra(ao, face); // tpt-metal

          insertQuad(*meshData.vertices, quad, vertexI, meshData.maxVertices);
        }
      }
    }

    const int faceVertexLength = vertexI - faceVertexBegin;
    meshData.faceVertexBegin[face] = faceVertexBegin;
    meshData.faceVertexLength[face] =faceVertexLength;
  }

  // Greedy meshing faces 4-5
  for (int face = 4; face < 6; face++) {
    const int axis = face / 2;

    const int faceVertexBegin = vertexI;

    for (int forward = 0; forward < CS; forward++) {
      const int bitsLocation = forward * CS + face * CS_2;
      const int bitsForwardLocation = (forward + 1) * CS + face * CS_2;

      for (int right = 0; right < CS; right++) {
        uint64_t bitsHere = faceMasks[right + bitsLocation];
        if (bitsHere == 0) continue;

        const uint64_t bitsForward = forward < CS - 1 ? faceMasks[right + bitsForwardLocation] : 0;
        const uint64_t bitsRight = right < CS - 1 ? faceMasks[right + 1 + bitsLocation] : 0;
        const int rightCS = right * CS;

        while (bitsHere) {
          unsigned long bitPos;
          #ifdef _MSC_VER
            _BitScanForward64(&bitPos, bitsHere);
          #else
            bitPos = __builtin_ctzll(bitsHere);
          #endif

          bitsHere &= ~(1ull << bitPos);

          const uint8_t type = voxels[getAxisIndex(axis, right + 1, forward + 1, bitPos)];
          const uint8_t ao = bmAO(meshData, axis, face, right + 1, forward + 1, bitPos); // tpt-metal
          uint8_t& forwardMergedRef = forwardMerged[rightCS + (bitPos - 1)];
          uint8_t& rightMergedRef = rightMerged[bitPos - 1];
          
          if (rightMergedRef == 0 && (bitsForward >> bitPos & 1) && type == voxels[getAxisIndex(axis, right + 1, forward + 2, bitPos)]
              && bmAOMerges(ao, bmAO(meshData, axis, face, right + 1, forward + 2, bitPos), false)) {
            forwardMergedRef++;
            continue;
          }
          
          if ((bitsRight >> bitPos & 1) && forwardMergedRef == forwardMerged[(rightCS + CS) + (bitPos - 1)] && type == voxels[getAxisIndex(axis, right + 2, forward + 1, bitPos)]
              && bmAOMerges(ao, bmAO(meshData, axis, face, right + 2, forward + 1, bitPos), true)) {
            forwardMergedRef = 0;
            rightMergedRef++;
            continue;
          }

          const uint8_t meshLeft = right - rightMergedRef;
          const uint8_t meshFront = forward - forwardMergedRef;
          const uint8_t meshUp = bitPos - 1 + (~face & 1);

          const uint8_t meshWidth = 1 + rightMergedRef;
          const uint8_t meshLength = 1 + forwardMergedRef;

          forwardMergedRef = 0;
          rightMergedRef = 0;
          
          const uint64_t quad = getQuad(meshLeft + (face == 4 ? meshWidth : 0), meshFront, meshUp, meshWidth, meshLength, type)
                                | bmQuadExtra(ao, face); // tpt-metal

          insertQuad(*meshData.vertices, quad, vertexI, meshData.maxVertices);
        }
      }
    }
  
    const int faceVertexLength = vertexI - faceVertexBegin;
    meshData.faceVertexBegin[face] = faceVertexBegin;
    meshData.faceVertexLength[face] =faceVertexLength;
  }

  meshData.vertexCount = vertexI + 1;
}

#endif // BM_IMPLEMENTATION
