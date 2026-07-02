#!/bin/sh
# compute-vp9 — system installer
#
# Builds in Release mode and installs into the correct system locations:
#   libcompute_vp9.so        → <prefix>/lib            (default /usr/lib)
#   compute_vp9_drv_video.so → libva driverdir         (e.g. /usr/lib/dri)
#   SPIR-V shaders           → <prefix>/share/compute-vp9/shaders
#   headers                  → <prefix>/include/compute_vp9
#
# Usage:
#   ./install.sh                 build + install (asks for sudo if needed)
#   ./install.sh --prefix /opt/x custom install prefix
#   ./install.sh --uninstall     remove a previous install
set -eu

SRC_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUILD_DIR="$SRC_DIR/build-release"
PREFIX="/usr"
UNINSTALL=0

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix)    PREFIX="$2"; shift 2 ;;
        --uninstall) UNINSTALL=1; shift ;;
        -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done

# Run a command as root only when necessary
as_root() {
    if [ "$(id -u)" = "0" ]; then "$@"; else sudo "$@"; fi
}

if [ "$UNINSTALL" = "1" ]; then
    MANIFEST="$BUILD_DIR/install_manifest.txt"
    if [ ! -f "$MANIFEST" ]; then
        echo "No install manifest found at $MANIFEST — nothing to uninstall." >&2
        exit 1
    fi
    echo "Removing installed files listed in $MANIFEST ..."
    as_root sh -c "xargs -d '\n' rm -f -- < '$MANIFEST' && ldconfig"
    echo "Uninstalled."
    exit 0
fi

# ─── Dependency checks ───────────────────────────────────────────────────────
for tool in cmake ninja pkg-config glslc; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "ERROR: '$tool' not found — install it first." >&2; exit 1; }
done
pkg-config --exists libva libva-drm || {
    echo "ERROR: libva development files not found (install libva/libva-devel)." >&2; exit 1; }

# ─── Configure + build ───────────────────────────────────────────────────────
echo "==> Configuring (prefix: $PREFIX)"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DENABLE_VULKAN=ON \
    -DENABLE_VAAPI=ON \
    -DENABLE_TESTS=OFF

echo "==> Building"
cmake --build "$BUILD_DIR" --parallel

# ─── Install ─────────────────────────────────────────────────────────────────
echo "==> Installing (may ask for sudo password)"
as_root cmake --install "$BUILD_DIR"
as_root ldconfig

DRIVER_DIR=$(pkg-config --variable=driverdir libva 2>/dev/null || true)
[ -n "$DRIVER_DIR" ] || DRIVER_DIR="$PREFIX/lib/dri"

cat <<EOF

==> Installed successfully.

  Library:  $PREFIX/lib/libcompute_vp9.so
  Driver:   $DRIVER_DIR/compute_vp9_drv_video.so
  Shaders:  $PREFIX/share/compute-vp9/shaders/

Verify the driver is picked up:

  LIBVA_DRIVER_NAME=compute_vp9 vainfo
  # expected: VAProfileVP9Profile0 : VAEntrypointVLD

Try it with Chrome:

  LIBVA_DRIVER_NAME=compute_vp9 google-chrome-stable \\
      --enable-features=VaapiVideoDecoder,VaapiVideoDecodeLinuxGL \\
      --disable-features=UseChromeOSDirectVideoDecoder

Then check chrome://gpu ("Video Decode: Hardware accelerated") and
chrome://media-internals while playing a VP9 video (YouTube).

To remove:  ./install.sh --uninstall
EOF
