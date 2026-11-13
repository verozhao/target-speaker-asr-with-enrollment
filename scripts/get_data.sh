#!/usr/bin/env bash
# Download LibriSpeech, WHAM! noise and OpenSLR 28 RIRs, then generate
# Libri2Mix (train-100, dev, test) and Libri3Mix (test) at 16 kHz, max mode,
# with the official LibriMix metadata and script.
# WITH_960=1 (default) also downloads train-clean-360 and train-other-500 so that
# training mixtures can be made on the fly from all of LibriSpeech 960.
#
# Usage: bash scripts/get_data.sh /path/to/storage
#        STEP=download bash scripts/get_data.sh ...   only download/extract (e.g. on a login node)
#        STEP=generate bash scripts/get_data.sh ...   only generate LibriMix (as a job)
# Needs ~120 GB free (+ ~60 GB for the 960 h subsets). LibriMix deps: soundfile pandas numpy scipy tqdm.
set -euo pipefail

STORAGE=$(mkdir -p "${1:?usage: get_data.sh STORAGE_DIR}" && cd "$1" && pwd)
PY=${PYTHON:-python}
cd "$STORAGE"

fetch() {  # url
  wget -c --tries=0 --read-timeout=20 "$1" -P "$STORAGE"
}

STEP=${STEP:-all}
if [ "$STEP" != "generate" ]; then
SUBSETS="train-clean-100 dev-clean test-clean"
[ "${WITH_960:-1}" = "1" ] && SUBSETS="$SUBSETS train-clean-360 train-other-500"
for s in $SUBSETS; do
  if [ ! -d "LibriSpeech/$s" ]; then
    fetch "http://www.openslr.org/resources/12/$s.tar.gz"
    tar -xzf "$s.tar.gz" && rm "$s.tar.gz"
  fi
done

if [ ! -d wham_noise/tt ]; then
  fetch "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
  unzip -q wham_noise.zip && rm wham_noise.zip
fi

if [ ! -d RIRS_NOISES/simulated_rirs ]; then
  fetch "https://www.openslr.org/resources/28/rirs_noises.zip"
  unzip -q rirs_noises.zip && rm rirs_noises.zip
fi

[ -d LibriMix ] || git clone https://github.com/JorisCos/LibriMix.git
git -C LibriMix rev-parse HEAD > librimix_commit.txt
fi
[ "$STEP" = "download" ] && { echo "downloads done in $STORAGE"; exit 0; }

# The LibriMix script processes every CSV in --metadata_dir whose name lacks
# "info", so pass a copy that holds only the splits we use (no train-360).
# "--types mix_clean" alone fails (noise/ is never created), so mix_both is
# generated too and simply not used.
gen() {  # n_src, splits...
  local n=$1; shift
  local md="$STORAGE/librimix_metadata_subset/Libri${n}Mix"
  mkdir -p "$md"
  for s in "$@"; do
    cp "LibriMix/metadata/Libri${n}Mix/libri${n}mix_${s}.csv" "$md/"
  done
  "$PY" LibriMix/scripts/create_librimix_from_metadata.py \
    --librispeech_dir "$STORAGE/LibriSpeech" \
    --wham_dir "$STORAGE/wham_noise" \
    --metadata_dir "$md" \
    --librimix_outdir "$STORAGE" \
    --n_src "$n" --freqs 16k --modes max --types mix_clean mix_both
}

[ -d Libri2Mix/wav16k/max/test ] || gen 2 train-clean-100 dev-clean test-clean
[ -d Libri3Mix/wav16k/max/test ] || gen 3 test-clean

echo "done. Libri2Mix at $STORAGE/Libri2Mix/wav16k/max, Libri3Mix at $STORAGE/Libri3Mix/wav16k/max"
