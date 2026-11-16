#!/usr/bin/env bash
# Download the AMI manual annotations and the single distant microphone audio
# (Array1-01, 16 kHz) of the full-corpus ASR partition sessions.
# Usage: bash scripts/get_ami.sh STORAGE_DIR [test|dev|train|all]
#   test: 16 sessions, ~1 GB    train: every other annotated session, ~10 GB
set -euo pipefail
STORAGE=$(mkdir -p "${1:?usage: get_ami.sh STORAGE_DIR [test|dev|train|all]}" && cd "$1" && pwd)
SPLIT=${2:-test}
AMI=$STORAGE/ami
mkdir -p "$AMI/audio"

TEST="EN2002a EN2002b EN2002c EN2002d ES2004a ES2004b ES2004c ES2004d IS1009a IS1009b IS1009c IS1009d TS3003a TS3003b TS3003c TS3003d"
DEV="ES2011a ES2011b ES2011c ES2011d IS1008a IS1008b IS1008c IS1008d TS3004a TS3004b TS3004c TS3004d IB4001 IB4002 IB4003 IB4004 IB4005 IB4010 IB4011"
if [ ! -d "$AMI/annotations/words" ]; then
  wget -c --tries=0 --read-timeout=20 -O "$AMI/ami_public_manual_1.6.2.zip" \
    https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
  unzip -q -o "$AMI/ami_public_manual_1.6.2.zip" -d "$AMI/annotations"
fi

# training partition: every annotated session outside dev and test
ALL=$(ls "$AMI/annotations/words" | sed -n 's/^\([A-Z]*[0-9]*[a-z]*\)\..*\.words\.xml$/\1/p' | sort -u)
TRAIN=$(printf '%s\n' $ALL | grep -vxF -f <(printf '%s\n' $TEST $DEV) || true)
case "$SPLIT" in
  test) SESSIONS=$TEST ;;
  dev) SESSIONS=$DEV ;;
  train) SESSIONS=$TRAIN ;;
  all) SESSIONS="$TEST $DEV $TRAIN" ;;
  *) echo "split must be test, dev, train or all"; exit 1 ;;
esac

for s in $SESSIONS; do
  f="$AMI/audio/$s.Array1-01.wav"
  if [ ! -s "$f" ]; then
    wget -c --tries=3 --read-timeout=20 -O "$f" \
      "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$s/audio/$s.Array1-01.wav" \
      || { echo "WARNING: no Array1-01 audio for $s"; rm -f "$f"; }
  fi
done
echo "AMI annotations: $AMI/annotations   audio: $AMI/audio"
