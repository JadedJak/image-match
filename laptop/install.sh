#!/bin/bash
# Install or update image-match for the user running it (not root).
#
#   bash <path-to-repo>/laptop/install.sh
#
# Copies the project to ~/image-match (code only -- an existing laptop/.env
# is never touched), copies the CLIP weights there if they aren't already,
# creates laptop/.env from .env.example on first run, adds the commands to
# ~/.bashrc, and builds the image. Safe to re-run after code updates: it
# replaces app/ and laptop/ files, then rebuilds.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/image-match"
# Optional: a folder that already holds the weights (an old hf-cache/ copy).
# If none is found, they are downloaded once (~3.9 GB) after the build.
WEIGHTS_SRC="${WEIGHTS_SRC:-}"

[[ $(id -u) -ne 0 ]] || { echo "Run this as your normal user, not root/sudo."; exit 1; }

echo "== copying project from $SRC to $DEST"
mkdir -p "$DEST"
rsync -a --exclude .env --exclude __pycache__ "$SRC"/ "$DEST"/

if [[ -d $DEST/hf-cache/hub ]]; then
    echo "== CLIP weights already in $DEST/hf-cache"
elif [[ -d $WEIGHTS_SRC/hub ]]; then
    echo "== copying CLIP weights (3.7 GB)"
    rsync -a "$WEIGHTS_SRC"/ "$DEST/hf-cache"/
else
    echo "== no CLIP weights at $WEIGHTS_SRC -- will download them after the build"
fi

if [[ -f $DEST/laptop/.env ]]; then
    echo "== keeping existing $DEST/laptop/.env"
else
    sed "s#/home/YOURUSER/image-match#$DEST#" "$DEST/laptop/.env.example" > "$DEST/laptop/.env"
    echo "== created $DEST/laptop/.env -- edit the SOURCE/INDEX/PROXY paths before indexing"
fi

line="source $DEST/laptop/image-match.sh"
if grep -qxF "$line" "$HOME/.bashrc"; then
    echo "== commands already in ~/.bashrc"
else
    echo "$line" >> "$HOME/.bashrc"
    echo "== added image-match commands to ~/.bashrc (open a new terminal)"
fi

echo "== building image (first time downloads ~3 GB)"
cd "$DEST/laptop" && podman compose build imgmatch-general

if [[ ! -d $DEST/hf-cache/hub ]]; then
    # the only step allowed to use the network for weights -- containers
    # run with HF_HUB_OFFLINE=1 and fail loudly if this folder is empty
    echo "== downloading CLIP weights (~3.9 GB, one time)"
    mkdir -p "$DEST/hf-cache"
    podman run --rm -v "$DEST/hf-cache":/opt/hf-cache:z -e HF_HUB_OFFLINE=0 \
        localhost/image-match-xpu:latest python -c \
        "import open_clip; open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')"
fi

cat <<EOF

Done. Next:
  1. Edit $DEST/laptop/.env -- point SOURCE/INDEX/PROXY at your folders
     (mount VeraCrypt first; inside it, use ext4, not exFAT/NTFS).
  2. Open a new terminal, then:  indexer all   then:  image-match up
  3. Browse http://127.0.0.1:7860 (general) and :7861 (personal).
EOF
