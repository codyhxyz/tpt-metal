#!/usr/bin/env bash
# Compiles Shaders/BigSand.metal ahead of time into .build/BigSand.metallib, so BigSand skips the source compile
# at launch. Needs the Metal Toolchain: xcodebuild -downloadComponent MetalToolchain
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/Shaders/BigSand.metal"
out="$root/.build"
mkdir -p "$out"

xcrun -sdk macosx metal -O3 -c "$src" -o "$out/BigSand.air"
xcrun -sdk macosx metallib "$out/BigSand.air" -o "$out/BigSand.metallib"
rm -f "$out/BigSand.air"
echo "built $out/BigSand.metallib"
