// Roofline: what this GPU can actually move and compute. Every later budget divides by these.
// Build + run: xcrun swiftc -O bench/roofline.swift -o .build/roofline && .build/roofline
// Times come from the command buffer's own GPU start/end (no CPU timing, no frame pacing).
import Metal
import Foundation

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void copy4(device const uint4* a [[buffer(0)]], device uint4* b [[buffer(1)]], uint i [[thread_position_in_grid]]) { b[i] = a[i]; }
kernel void read4(device const uint4* a [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
                  uint i [[thread_position_in_grid]], uint g [[threads_per_grid]]) {
  uint4 s = 0; for (uint k = i; k < n; k += g) s ^= a[k];
  if ((s.x ^ s.y ^ s.z ^ s.w) == 0x9e3779b9u) out[0] = 1; }   // keeps the loads alive
kernel void write4(device uint4* b [[buffer(0)]], uint i [[thread_position_in_grid]]) { b[i] = uint4(i); }
// Math: 8 independent FMA chains x ITER per thread.
kernel void fma32(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  float a0=i,a1=i+1,a2=i+2,a3=i+3,a4=i+4,a5=i+5,a6=i+6,a7=i+7; const float m=0.9999f, c=0.0001f;
  for (uint k=0;k<iters;k++){ a0=fma(a0,m,c);a1=fma(a1,m,c);a2=fma(a2,m,c);a3=fma(a3,m,c);a4=fma(a4,m,c);a5=fma(a5,m,c);a6=fma(a6,m,c);a7=fma(a7,m,c);}
  float s=a0+a1+a2+a3+a4+a5+a6+a7; if (s == 12345.678f) out[0]=s; }
// Bit ops, the sand sim's core work: 8 independent chains of (shift, and, or) per iteration.
kernel void bits32(device uint* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  uint a0=i,a1=i*3,a2=i*5,a3=i*7,a4=i*11,a5=i*13,a6=i*17,a7=i*19;
  for (uint k=0;k<iters;k++){ a0=((a0>>1)&a1)|a2; a1=((a1>>1)&a2)|a3; a2=((a2>>1)&a3)|a4; a3=((a3>>1)&a4)|a5;
                              a4=((a4>>1)&a5)|a6; a5=((a5>>1)&a6)|a7; a6=((a6>>1)&a7)|a0; a7=((a7>>1)&a0)|a1; }
  uint s=a0^a1^a2^a3^a4^a5^a6^a7; if (s == 0x12345678u) out[0]=s; }
kernel void bits64(device ulong* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  ulong a0=i,a1=ulong(i)*3,a2=ulong(i)*5,a3=ulong(i)*7,a4=ulong(i)*11,a5=ulong(i)*13,a6=ulong(i)*17,a7=ulong(i)*19;
  for (uint k=0;k<iters;k++){ a0=((a0>>16)&a1)|a2; a1=((a1>>16)&a2)|a3; a2=((a2>>16)&a3)|a4; a3=((a3>>16)&a4)|a5;
                              a4=((a4>>16)&a5)|a6; a5=((a5>>16)&a6)|a7; a6=((a6>>16)&a7)|a0; a7=((a7>>16)&a0)|a1; }
  ulong s=a0^a1^a2^a3^a4^a5^a6^a7; if (s == 0x123456789ul) out[0]=s; }
"""

let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let q = dev.makeCommandQueue()!
func pipe(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }

/// Median GPU seconds of `reps` command buffers, after one warmup.
func time(_ reps: Int = 7, _ encode: (MTLComputeCommandEncoder) -> Void) -> Double {
  var ts: [Double] = []
  for r in 0...reps {
    let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
    encode(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if r > 0 { ts.append(cb.gpuEndTime - cb.gpuStartTime) }
  }
  return ts.sorted()[ts.count / 2]
}
func dispatch(_ e: MTLComputeCommandEncoder, _ p: MTLComputePipelineState, _ n: Int) {
  e.setComputePipelineState(p)
  e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}

let GiB = 1 << 30
let pCopy = pipe("copy4"), pRead = pipe("read4"), pWrite = pipe("write4")
let pF = pipe("fma32"), pB32 = pipe("bits32"), pB64 = pipe("bits64")
let small = dev.makeBuffer(length: 64, options: .storageModePrivate)!

print("Memory (GB/s = 1e9 bytes/s; bytes counted = read + write)")
for size in [4 << 20, 32 << 20, 256 << 20, GiB] {
  let a = dev.makeBuffer(length: size, options: .storageModePrivate)!
  let b = dev.makeBuffer(length: size, options: .storageModePrivate)!
  let n16 = size / 16
  // many reps inside one command buffer for small sizes so launch overhead doesn't dominate
  let inner = max(1, (256 << 20) / size)
  let tc = time { e in for _ in 0..<inner { e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1); dispatch(e, pCopy, n16) } }
  let tw = time { e in for _ in 0..<inner { e.setBuffer(b, offset: 0, index: 0); dispatch(e, pWrite, n16) } }
  var n = UInt32(n16)
  let tr = time { e in for _ in 0..<inner { e.setBuffer(a, offset: 0, index: 0); e.setBuffer(small, offset: 0, index: 1); e.setBytes(&n, length: 4, index: 2); dispatch(e, pRead, 10 * 1024 * 32) } }
  let mb = size >> 20
  print(String(format: "  %5d MiB  copy %6.1f   read %6.1f   write %6.1f", mb,
               2 * Double(size * inner) / tc / 1e9, Double(size * inner) / tr / 1e9, Double(size * inner) / tw / 1e9))
}

print("Math (G ops/s)")
var iters = UInt32(4096)
let threads = 10 * 1024 * 64
let tf = time { e in e.setBuffer(small, offset: 0, index: 0); e.setBytes(&iters, length: 4, index: 1); dispatch(e, pF, threads) }
let t32 = time { e in e.setBuffer(small, offset: 0, index: 0); e.setBytes(&iters, length: 4, index: 1); dispatch(e, pB32, threads) }
let t64 = time { e in e.setBuffer(small, offset: 0, index: 0); e.setBytes(&iters, length: 4, index: 1); dispatch(e, pB64, threads) }
let ops = Double(threads) * Double(iters) * 8
print(String(format: "  fp32 FMA        %7.0f G FMA/s  (= %.2f TFLOPS)", ops / tf / 1e9, 2 * ops / tf / 1e12))
print(String(format: "  u32 shift+and+or %6.0f G triples/s (%.0f G bit-ops/s)", ops / t32 / 1e9, ops * 3 * 32 / t32 / 1e9))
print(String(format: "  u64 shift+and+or %6.0f G triples/s (%.0f G bit-ops/s)", ops / t64 / 1e9, ops * 3 * 64 / t64 / 1e9))
