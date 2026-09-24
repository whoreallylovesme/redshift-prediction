#!/usr/bin/env bash
# Чанковая загрузка 4 мозаик тайла (VIS,Y,J,H) с IRSA. Аргумент: tile id.
set -u
TILE="$1"
BASE="https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/MER/$TILE"
OUT="/root/euclid_q1/mosaics/$TILE"
mkdir -p "$OUT"

CHUNK=32   # МБ
NCHUNK=16  # параллельных потоков

dl_file() {
  local url="$1" out="$2"
  local want=$(curl -sI --max-time 30 "$url" | grep -i content-length | tr -dc '0-9')
  [ -z "$want" ] || [ "$want" -lt 1000 ] && { echo "нет $url"; return 1; }
  local have=0
  [ -f "$out" ] && have=$(stat -c%s "$out")
  [ "$have" = "$want" ] && return 0
  local nchunks=$(( (want + CHUNK*1048576 - 1) / (CHUNK*1048576) ))
  local i=0
  while [ $i -lt $nchunks ]; do
    local j=$i
    local pids=()
    while [ $j -lt $nchunks ] && [ ${#pids[@]} -lt $NCHUNK ]; do
      local A=$(( j * CHUNK * 1048576 ))
      local B=$(( A + CHUNK * 1048576 - 1 ))
      [ $B -ge $want ] && B=$(( want - 1 ))
      local part="${out}.part$j"
      local partsize=$(( B - A + 1 ))
      local havesize=0
      [ -f "$part" ] && havesize=$(stat -c%s "$part")
      if [ "$havesize" != "$partsize" ]; then
        curl -s --max-time 3600 -r $A-$B -o "$part" "$url" &
        pids+=($!)
      fi
      j=$(( j + 1 ))
    done
    wait
    i=$j
  done
  # сборка
  cat $(for k in $(seq 0 $((nchunks-1))); do echo "${out}.part$k"; done) > "$out"
  local got=$(stat -c%s "$out")
  rm -f "$out".part*
  if [ "$got" = "$want" ]; then
    echo "OK $(basename $out) $got"
    return 0
  else
    echo "FAIL $out $got/$want"
    rm -f "$out"
    return 1
  fi
}

for BAND in VIS NIR-Y NIR-J NIR-H; do
  if [ "$BAND" = "VIS" ]; then DIR=VIS; PAT="BGSUB-MOSAIC-VIS"; else DIR=NISP; PAT="BGSUB-MOSAIC-$BAND"; fi
  F=$(curl -s --max-time 30 "$BASE/$DIR/" | grep -oE "href=\"[^\"]+$PAT[^\"]+\.fits\"" | sed 's/href="//; s/"//' | head -1)
  [ -z "$F" ] && { echo "нет мозаики $BAND у тайла $TILE"; exit 1; }
  dl_file "$BASE/$DIR/$F" "$OUT/$(echo $BAND | tr 'A-Z-' 'a-z_').fits" || exit 1
done
echo "МОЗАИКИ $TILE ГОТОВЫ"
