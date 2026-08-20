#!/usr/bin/env bash
# Build the demo image ONCE, then distribute the identical image to both
# machines (Phase 2). Never rebuild separately per-machine.
#
#   scripts/build_image.sh                 # build + tag hetero-demo:latest
#   scripts/build_image.sh --save          # also write hetero-demo.tar
#   scripts/build_image.sh --load FILE.tar # load an image from a tarball
set -euo pipefail

IMG="${IMG:-hetero-demo:latest}"
TAR="${TAR:-hetero-demo.tar}"

case "${1:-}" in
    --save)
        docker build -t "$IMG" .
        echo "saving to $TAR (compress with -C zstd,0 for smaller transfer)..."
        docker save -o "$TAR" "$IMG"
        ls -lh "$TAR"
        ;;
    --load)
        [ -n "${2:-}" ] || { echo "usage: build_image.sh --load FILE.tar" >&2; exit 1; }
        docker load -i "$2"
        ;;
    "")
        docker build -t "$IMG" .
        ;;
    *)
        echo "usage: build_image.sh [--save|--load FILE.tar]" >&2
        exit 1
        ;;
esac

echo "image: $IMG"
docker images "$IMG" --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'