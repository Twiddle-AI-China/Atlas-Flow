#!/usr/bin/env bash
# Fetch ONNX Runtime and, on macOS, fuse it into a universal library.
#
# Why universal is not optional: a host running under Rosetta is x86_64 and
# silently skips an arm64-only plugin. It does not appear in the browser, it
# does not fail, and nothing is logged -- which is exactly what happened here
# against an Ableton Live that had Rosetta turned on.
#
# Why 1.23.2 rather than the newest: Microsoft stopped publishing macOS x86_64
# builds after it. The arm64 half comes from the GitHub release (which also
# carries the C++ headers); the x86_64 half is taken from the Python wheel and
# the two slices are fused with lipo. (v1.23.2 does also publish an
# osx-universal2 tarball, which would make the fuse unnecessary -- worth
# switching to, but the lipo path is what the shipped bundles were built and
# signed with, so it is left alone until someone can re-verify a build.)
#
# Windows needs none of that: one zip, two file names, no signing.
#
# The re-sign at the end matters. install_name_tool and lipo both invalidate a
# Mach-O signature, and macOS answers an invalid signature with SIGKILL and no
# message at all -- the plugin simply dies on load.
set -euo pipefail

VERSION="${ORT_VERSION:-1.23.2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/third_party/onnxruntime"

if [ -f "$DEST/VERSION_NUMBER" ] && [ "$(cat "$DEST/VERSION_NUMBER")" = "$VERSION" ]; then
  echo "onnxruntime $VERSION already present"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

case "$(uname -s)" in
  Darwin)
    PACKAGE="onnxruntime-osx-arm64-${VERSION}"
    echo "fetching ${PACKAGE} (headers + arm64 slice)"
    curl -sSL -o "$TMP/arm.tgz" \
      "https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PACKAGE}.tgz"
    tar xzf "$TMP/arm.tgz" -C "$TMP"

    echo "fetching the x86_64 slice from the python wheel"
    WHEEL_DIR="$TMP/wheel"
    mkdir -p "$WHEEL_DIR"
    python3 -m pip download --quiet --only-binary=:all: \
      --platform macosx_13_0_x86_64 --python-version 3.13 --no-deps \
      "onnxruntime==${VERSION}" -d "$WHEEL_DIR"
    unzip -qo "$WHEEL_DIR"/onnxruntime-*x86_64.whl -d "$WHEEL_DIR/x64"

    ARM="$TMP/${PACKAGE}/lib/libonnxruntime.${VERSION}.dylib"
    X64="$WHEEL_DIR/x64/onnxruntime/capi/libonnxruntime.${VERSION}.dylib"
    [ -f "$ARM" ] && [ -f "$X64" ] || { echo "missing a slice" >&2; exit 1; }

    cp "$ARM" "$TMP/arm.dylib"; cp "$X64" "$TMP/x64.dylib"
    chmod u+w "$TMP/arm.dylib" "$TMP/x64.dylib"
    # lipo requires one install name, and the plugin asks for the major-version
    # form, which is what the bundle embeds.
    install_name_tool -id "@rpath/libonnxruntime.1.dylib" "$TMP/arm.dylib"
    install_name_tool -id "@rpath/libonnxruntime.1.dylib" "$TMP/x64.dylib"
    lipo -create "$TMP/arm.dylib" "$TMP/x64.dylib" -output "$TMP/universal.dylib"

    rm -rf "$DEST"
    mv "$TMP/${PACKAGE}" "$DEST"
    cp "$TMP/universal.dylib" "$DEST/lib/libonnxruntime.${VERSION}.dylib"
    rm -f "$DEST/lib/libonnxruntime.dylib" "$DEST/lib/libonnxruntime.1.dylib"
    ln -s "libonnxruntime.${VERSION}.dylib" "$DEST/lib/libonnxruntime.dylib"
    ln -s "libonnxruntime.${VERSION}.dylib" "$DEST/lib/libonnxruntime.1.dylib"

    codesign --force --sign - --timestamp=none "$DEST/lib/libonnxruntime.${VERSION}.dylib"
    codesign --verify --strict "$DEST/lib/libonnxruntime.${VERSION}.dylib"

    echo "onnxruntime $VERSION -> $(lipo -archs "$DEST/lib/libonnxruntime.${VERSION}.dylib")"
    ;;

  MINGW*|MSYS*|CYGWIN*|Windows_NT)
    # Windows ships a plain zip with the same shape as the other platforms --
    # include/, lib/, VERSION_NUMBER -- so nothing downstream has to special
    # case the layout, only the library names: onnxruntime.lib to link against
    # and onnxruntime.dll to ship. There is no fusing to do and no signature to
    # invalidate, which makes this the shortest branch of the three.
    ARCH="${ORT_ARCH:-x64}"
    PACKAGE="onnxruntime-win-${ARCH}-${VERSION}"
    echo "fetching ${PACKAGE}"
    curl -sSL -o "$TMP/ort.zip" \
      "https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PACKAGE}.zip"
    unzip -q "$TMP/ort.zip" -d "$TMP"
    [ -f "$TMP/${PACKAGE}/lib/onnxruntime.dll" ] || { echo "no dll in ${PACKAGE}" >&2; exit 1; }
    rm -rf "$DEST"
    mv "$TMP/${PACKAGE}" "$DEST"
    echo "onnxruntime $VERSION (${ARCH}) -> ${DEST#"$HERE/"}"
    ;;

  Linux)
    PACKAGE="onnxruntime-linux-x64-${VERSION}"
    curl -sSL -o "$TMP/ort.tgz" \
      "https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PACKAGE}.tgz"
    tar xzf "$TMP/ort.tgz" -C "$TMP"
    rm -rf "$DEST"
    mv "$TMP/${PACKAGE}" "$DEST"
    echo "onnxruntime $VERSION -> ${DEST#"$HERE/"}"
    ;;

  *)
    echo "unsupported platform: $(uname -s)" >&2
    exit 1
    ;;
esac
