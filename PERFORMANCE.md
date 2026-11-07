# Massive falling sand, from first principles

All numbers divide by the measured machine in FACTS.md: **120 GB/s** main-memory traffic, **~2.4 T** simple integer ops/s,
**25 GB** we may keep resident, a 3024 × 1964 screen.

## The test scene
A cube of sand **4096 cells on a side, 69 billion grains**, released 2000 cells above the ground. It falls, hits the ground,
and spreads into a pile at sand's angle of repose (45° per cell step: The Powder Toy's and Noita's rule). The world is
**16384 × 8192 × 16384 = 2.2 trillion cells**. It has to run smoothly: ≤12 ms of GPU per frame, every frame.
(If a cell is 1 cm, the cube is 41 m on a side and lands at about 20 m/s, about 33 cells per tick at 60 ticks/s.)

## Three different limits, not one
"How big" is three separate questions with three separate costs:

| Question | What it costs | Ceiling here |
|---|---|---|
| How much world **exists** | memory, proportional to what's *interesting* (surfaces, mixtures), not volume | trillions of cells resident; beyond that, stream from the SSD |
| How much can **move at once** | memory traffic per tick, proportional to what's *changing in a non-trivial way* | ~2.5 billion individually colliding grains per tick at 60 ticks/s; far more moving together |
| How much you can **see** | rays per pixel × steps per ray: proportional to the screen, not the world | ~2–4 ms per frame at any world size, with a skip hierarchy |

The design rule that follows: **work must scale with what's happening, never with the world's volume.**

## 1. Storing 2.2 trillion cells
- One byte per cell, dense: 2.2 TB. Impossible. One bit per cell, dense: 275 GB. Still impossible.
- But almost all of the world is air or solid ground in long uniform runs. Store each **column** of the world as a list of
  vertical runs ("from height 0 to 1830: stone; 1830 to 1840: sand; above: air"). This is Voxlap's layout (Ken Silverman's
  voxel engine, which stored worlds as vertical spans for exactly this reason).
- 16384 × 16384 = 268 M columns × a few runs × 4–8 bytes per run ≈ **2–4 GB**. The falling cube adds 16.7 M columns × 1 run
  = **134 MB**. All of it fits in the 25 GB with room to spare.

## 2. Moving 69 billion grains
- Per grain, dense: 69 G grains × 3 bits of traffic (read sand, read ground, write sand) = 26 GB per tick → **215 ms per tick**.
  Thirteen times over the whole frame budget. Per-grain simulation of the whole cube is impossible, and no tuning fixes it.
- But a falling cube isn't 69 G independent events. Each column of it is **one run falling together**: moving it means
  adding the fall speed to the run's start height. 16.7 M runs × ~16 bytes of traffic = **270 MB per tick ≈ 2.2 ms**.
  Speed is free: falling 33 cells costs the same as falling 1.
- Grains need individual work only where they **collide**: the impact zone at the bottom, and the avalanching surface of
  the pile. Those cells are unpacked into dense **bit tiles** (1 bit per cell, 64 cells per 64-bit word, so one integer op
  handles 64 grains) and simulated per cell: fall if the cell below is empty; else slide diagonally; else rest.
  Conflict-free: straight falls first, then slides in one direction per pass (+x, −x, +z, −z rotating), so every empty
  cell has only one possible source. A cheap hash per grain decides who slides, so piles don't grow grid-shaped.
- Per-cell budget: 8 ms of the frame × 120 GB/s = 0.96 GB ÷ 3 bits per cell = **~2.5 billion colliding cells per tick**.
  The math is ~40 integer ops per 64-bit word, about 1.3 ms for those 2.5 G cells, so this work is **memory-bound** and
  bytes are the only lever. That's why 1 bit, not 1 byte.
- At impact, each column feeds up to ~33 grains per tick into the impact zone: 16.7 M × 33 = **550 M cells per tick**,
  under a quarter of the budget.
- Once a part of the pile stops moving, it packs back into runs and goes to sleep: **zero cost**.

## 3. Seeing it
- 3024 × 1964 = 5.9 M pixels. Trace rays at half resolution (1.5 M) and upscale with Apple's MetalFX.
- Each ray walks the column runs, skipping empty space with a coarse "is anything here" hierarchy (one bit per 4³, 16³,
  64³ block). That's ~50–100 steps per ray at ~30 integer ops each, ≈ 4.5 G ops ≈ **2 ms**, the same for a world 10×
  bigger.
- Why not triangle meshes: meshing costs time for every *changed* surface. Here a billion grains change every tick, so
  meshes would cost seconds per tick. Rays cost the same however much moves.

## Frame budget (60 fps, ≤12 ms GPU)
| Pass | Estimate |
|---|---|
| Falling runs | ~2.2 ms |
| Colliding grains (impact + avalanches) | ~1.5–5 ms, depending on the moment |
| Rays + shading + upscale | ~3–4 ms |
| **Total** | **~7–11 ms** |

If a moment ever needs more colliding cells than the budget, physics slows down (fewer ticks per second) while drawing
stays smooth at full rate, never the other way around.

## What the estimates assume, and the first tests that check them
1. A tiled bit-sim kernel reaches ≥70% of the 120 GB/s. Test: dense bit-tile sim on a 1 G-cell block; GB/s achieved.
2. The run-falling kernel costs ~16 bytes of traffic per column per tick. Test: 16.7 M falling runs, ms per tick.
3. Rays take ≤100 steps through the hierarchy. Test: count steps per pixel on the landed pile; ms per frame.
Each test prints **expected vs measured** next to each other. A gap means the model is wrong somewhere, and we find out where.
