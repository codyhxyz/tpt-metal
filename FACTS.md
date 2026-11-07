# The machine (facts every budget divides by)

Measured 2026-11-07 on Cody's MacBook Pro (Mac17,2), macOS 27.0, on AC power, GPU claimed exclusively.
Re-measure with `xcrun swiftc -O bench/roofline.swift -o .build/roofline && .build/roofline` (about 3 s).

## Chip
- Apple M5: 10-core CPU (4 performance + 6 efficiency), **10-core GPU**, Metal 4, hardware ray tracing.
- **32 GB unified memory**: CPU and GPU share it. Nothing is copied to "the GPU".
- Metal says it will keep **25.0 GB** resident for us (`recommendedMaxWorkingSetSize`). The largest single buffer is **18.7 GB**.
- **32 KB of threadgroup memory** (fast on-chip scratch shared by one group of GPU threads); up to 1024 threads per group, 32 per SIMD group.
- GPU timers: timestamps at stage boundaries (start or end of a pass), not inside a dispatch. Pass times come from command-buffer and encoder boundaries.
- Screen: built-in Liquid Retina XDR, **3024 × 1964**, ProMotion (up to 120 Hz).

## Speeds (measured; Apple's spec in brackets)
| What | Measured | Note |
|---|---|---|
| Main memory, read | **123 GB/s** | [Apple: 153.6 GB/s theoretical peak] |
| Main memory, write | **142 GB/s** | |
| Main memory, copy (read + write) | **120 GB/s** | ~78% of peak: the realistic ceiling for streaming kernels |
| Small working set (4 MiB), copy | **247 GB/s** | Fits in the on-chip cache: data reused while it's still in cache is ~2× cheaper |
| fp32 math | **3.93 TFLOPS** (1965 G fused multiply-adds/s) | |
| 32-bit bit ops | **806 G** (shift, and, or) triples/s | ~2.4 T simple integer ops/s |
| 64-bit bit ops | **417 G** triples/s | Half the rate, but twice the bits per op: same ~80 T bit-ops/s |

## Ratios that decide designs
- **Math is cheap; memory is expensive.** The GPU does ~16 multiply-adds, or ~20 integer ops, in the time it moves one byte from main memory.
  A kernel that does fewer than ~20 integer ops per byte it moves is **memory-bound**: it runs at the speed of its bytes, and fewer bytes is the only lever.
- One byte read + written per cell → **at most 60 G cell updates/s**. One bit per cell → **8× that**.
- Data reused while it's still in cache moves at ~2× main-memory speed: tile the work so each byte comes in from main memory once.
- 16.7 ms frame (60 fps) × 120 GB/s = **2.0 GB of memory traffic per frame, total**, for physics and drawing together. Our steady-60 rule (≤12 ms GPU) makes that **1.4 GB**.

## Toolchain
- Swift 6.4.2 (`xcrun swiftc`). Shaders are precompiled by `scripts/build-shaders.sh` (Metal Toolchain installed); the app compiles `Shaders/BigSand.metal` at launch only when no `.metallib` is found.

Sources: [Apple newsroom, M5](https://www.apple.com/jo/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/), [Wikipedia, Apple M5](https://en.wikipedia.org/wiki/Apple_M5) (153.6 GB/s, LPDDR5X-9600).
