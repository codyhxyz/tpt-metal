#!/usr/bin/env bash
# Install /Applications/BigSand.app: a launcher that, every time it is opened, rebuilds BigSand from this repo
# (the shaders only if a file in Shaders/ changed, the Swift code incrementally) and then runs it in its own process,
# so the game is always the code as it is now. Build output goes to ~/Library/Logs/BigSand-build.log; a failed
# build shows an alert instead of starting a stale game. Run this once (again only if the repo moves).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${1:-/Applications/BigSand.app}"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
sed -e 's|<string>xyz.codyh.bigsand</string>|<string>xyz.codyh.bigsand.launcher</string>|' \
  "$ROOT/Resources/Info.plist" > "$APP/Contents/Info.plist"
printf 'APPL????' > "$APP/Contents/PkgInfo"

cat > "$APP/Contents/MacOS/BigSand" <<EOF
#!/bin/bash
# BigSand launcher (made by scripts/make-launcher.sh): rebuild what changed, then run the game.
ROOT="$ROOT"
LOG="\$HOME/Library/Logs/BigSand-build.log"
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:\$PATH"
cd "\$ROOT" || exit 1

build() {
  local lib=.build/BigSand.metallib
  if [[ ! -f \$lib ]] || [[ -n \$(find Shaders -newer "\$lib" -print -quit) ]]; then
    bash scripts/build-shaders.sh || return 2
  fi
  swift build -c release || return 3
}
build > "\$LOG" 2>&1
case \$? in
  0) ;;
  2) osascript -e 'display alert "BigSand didn’t build" message "The shaders failed to compile. See ~/Library/Logs/BigSand-build.log." as critical' > /dev/null; exit 1 ;;
  *) osascript -e 'display alert "BigSand didn’t build" message "The Swift code failed to compile. See ~/Library/Logs/BigSand-build.log." as critical' > /dev/null; exit 1 ;;
esac
exec "\$ROOT/.build/release/BigSand" "\$@"
EOF
chmod +x "$APP/Contents/MacOS/BigSand"

plutil -lint "$APP/Contents/Info.plist" > /dev/null
codesign --force --sign - "$APP" > /dev/null 2>&1 || true
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" 2>/dev/null || true
echo "$APP"
