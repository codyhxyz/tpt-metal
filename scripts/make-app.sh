#!/usr/bin/env bash
# Build BigSand and package it as build/BigSand.app, ad-hoc signed.
# The games category and LSSupportsGameMode in Resources/Info.plist let macOS
# turn on Game Mode when the app runs full screen.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

swift build -c release

if [[ -f scripts/build-shaders.sh ]]; then
  bash scripts/build-shaders.sh
fi

APP="$ROOT/build/BigSand.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$(swift build -c release --show-bin-path)/BigSand" "$APP/Contents/MacOS/BigSand"
cp Resources/Info.plist "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"

if [[ -f .build/BigSand.metallib ]]; then
  cp .build/BigSand.metallib "$APP/Contents/Resources/BigSand.metallib"
fi
# Source fallback: the app compiles this at runtime when no metallib is bundled.
cp Shaders/*.metal "$APP/Contents/Resources/"

plutil -lint "$APP/Contents/Info.plist" >/dev/null
codesign --force --sign - --deep "$APP"
codesign -v "$APP"

echo "$APP"
