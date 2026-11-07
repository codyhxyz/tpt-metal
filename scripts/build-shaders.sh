#!/usr/bin/env bash
# Compiles Shaders/*.metal ahead of time into one .build/BigSand.metallib, so the app skips the source compile
# at launch. Needs the Metal Toolchain: xcodebuild -downloadComponent MetalToolchain
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/.build"
mkdir -p "$out"
rm -f "$out/BigSand.metallib"

airs=()
for src in "$root"/Shaders/*.metal; do
  air="$out/$(basename "${src%.metal}").air"
  xcrun -sdk macosx metal -O3 -c "$src" -o "$air"
  airs+=("$air")
done
xcrun -sdk macosx metallib "${airs[@]}" -o "$out/BigSand.metallib"
rm -f "${airs[@]}"
echo "built $out/BigSand.metallib"
