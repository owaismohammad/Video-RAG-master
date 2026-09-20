#!/usr/bin/env bash
# ===========================================================================
# Normalise dataset videos into a format decord/OpenCV/ffmpeg decode reliably:
#
#   container  MP4 (+faststart)        video  H.264, yuv420p, even width/height
#   frame rate constant (CFR)          audio  AAC (if the file has audio)
#
# By default only files that need it are converted. A file "needs it" if it is
# not already MP4/H.264/yuv420p/even-sized/constant-fps, OR if decord (what the
# benchmark actually uses) fails to read 64 evenly spaced frames from it.
#
# Files are converted in place: <DIR>/Video_ID_7.mkv -> <DIR>/Video_ID_7.mp4,
# and the original is MOVED to <DIR>/originals/ -- never deleted. The output is
# verified (duration + decord read) before the original is touched.
#
#   bash convert_videos.sh                         # /workspace/dataset/videos
#   bash convert_videos.sh -n                      # dry run: just report
#   bash convert_videos.sh -f                      # re-encode everything
#   bash convert_videos.sh -H 720 /path/to/videos  # also cap height at 720p
#
# Options:
#   -n        dry run (report what would be converted, change nothing)
#   -f        force: convert every file, even ones that already look fine
#   -r        go from the highest name down (Video_ID_20 first) instead of up
#   -m N      skip videos whose trailing number is below N (e.g. ones a running
#             benchmark has already processed)
#   -x N      skip videos whose trailing number is above N (split a big job
#             across several runs on disjoint ID ranges)
#   -H N      downscale to at most N pixels tall (default: keep resolution)
#   -c N      x264 CRF, lower = better/bigger (default 18, ~visually lossless)
#   -p NAME   x264 preset (default veryfast; slower = smaller file, more CPU)
#   -h        this help
# ===========================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="/workspace/dataset/videos"
FORCE=0; DRY=0; REV=0; MIN_ID=""; MAX_ID=""; MAX_H=""; CRF=18; PRESET="veryfast"

usage() { sed -n '2,/^# =====*$/p' "${BASH_SOURCE[0]}" | sed -n '2,$p' | sed 's/^# \{0,1\}//' | head -n -1; }

while getopts "fnrm:x:H:c:p:h" opt; do
    case "$opt" in
        f) FORCE=1 ;;
        n) DRY=1 ;;
        r) REV=1 ;;
        m) MIN_ID="$OPTARG" ;;
        x) MAX_ID="$OPTARG" ;;
        H) MAX_H="$OPTARG" ;;
        c) CRF="$OPTARG" ;;
        p) PRESET="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] && DIR="$1"

for t in ffmpeg ffprobe; do
    command -v "$t" >/dev/null 2>&1 || { echo "!! $t not found (apt-get install -y ffmpeg)"; exit 1; }
done
[ -d "$DIR" ] || { echo "!! not a directory: $DIR"; exit 1; }
for pair in "H:$MAX_H" "m:$MIN_ID" "x:$MAX_ID"; do
    if [ -n "${pair#*:}" ] && ! [[ "${pair#*:}" =~ ^[0-9]+$ ]]; then
        echo "!! -${pair%%:*} needs a number, got '${pair#*:}'"; exit 2
    fi
done

# decord check uses the project venv -- that is the decoder that really fails.
PYBIN="$REPO_ROOT/.venv/bin/python"
if [ -x "$PYBIN" ] && "$PYBIN" -c "import decord" >/dev/null 2>&1; then
    HAVE_DECORD=1
else
    HAVE_DECORD=0
    echo "(decord not importable from $PYBIN -- skipping the decord read test)"
fi

# --- helpers ---------------------------------------------------------------

# Read 64 evenly spaced frames the way the pipeline does. 0 = OK.
decord_ok() {
    [ "$HAVE_DECORD" = 1 ] || return 0
    "$PYBIN" - "$1" <<'PY' >/dev/null 2>&1
import sys
import numpy as np
from decord import VideoReader, cpu
vr = VideoReader(sys.argv[1], ctx=cpu(), num_threads=1)
n = len(vr)
assert n > 0, "no frames"
idx = np.linspace(0, n - 1, min(64, n)).astype(int).tolist()
vr.get_batch(idx).asnumpy()
PY
}

frac() { awk -F/ '{ if ($2 + 0) print $1 / $2; else print $1 + 0 }' <<<"$1"; }

duration() {
    ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$1" 2>/dev/null \
        | awk 'NR==1 && $1+0>0 {print $1+0; exit} END{if(NR==0||$1+0<=0) print 0}'
}

# Prints the reason and returns 0 if the file should be converted.
needs_conversion() {
    local f="$1" k v vcodec="" pix="" w=0 h=0 rfr="" afr="" fmt ext
    ext="${f##*.}"; ext="${ext,,}"
    while IFS='=' read -r k v; do
        case "$k" in
            codec_name) vcodec="$v" ;;  pix_fmt) pix="$v" ;;
            width) w="$v" ;;            height) h="$v" ;;
            r_frame_rate) rfr="$v" ;;   avg_frame_rate) afr="$v" ;;
        esac
    done < <(ffprobe -v error -select_streams v:0 \
        -show_entries stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate \
        -of default=nw=1 "$f" 2>/dev/null)
    fmt="$(ffprobe -v error -show_entries format=format_name -of default=nw=1:nk=1 "$f" 2>/dev/null | head -1)"

    [ -z "$vcodec" ]                && { echo "no readable video stream"; return 0; }
    [ "$ext" != "mp4" ]             && { echo "container .$ext"; return 0; }
    [[ "$fmt" != *mp4* ]]           && { echo "container '$fmt' inside .mp4"; return 0; }
    [ "$vcodec" != "h264" ]         && { echo "codec $vcodec"; return 0; }
    [ "$pix" != "yuv420p" ]         && { echo "pix_fmt $pix"; return 0; }
    if [ $((w % 2)) -ne 0 ] || [ $((h % 2)) -ne 0 ]; then
        echo "odd size ${w}x${h}"; return 0
    fi
    local r a
    r="$(frac "$rfr")"; a="$(frac "$afr")"
    if awk -v r="$r" -v a="$a" 'BEGIN{ d=r-a; if(d<0)d=-d; exit !(a>0 && d/a>0.02) }'; then
        echo "variable frame rate (r=$rfr avg=$afr)"; return 0
    fi
    decord_ok "$f" || { echo "decord cannot read it"; return 0; }
    return 1
}

CUR_TMP=""
trap '[ -n "$CUR_TMP" ] && rm -f -- "$CUR_TMP"; echo; echo "interrupted"; exit 130' INT TERM

convert_one() {
    local src="$1" stem final tmp vf sdur ddur
    stem="$(basename "${src%.*}")"
    final="$DIR/$stem.mp4"
    tmp="$DIR/.$stem.converting.mp4"

    if [ "$final" != "$src" ] && [ -e "$final" ]; then
        echo "  !! $final already exists; not overwriting. Rename one of them."
        return 1
    fi

    if [ -n "$MAX_H" ]; then
        vf="scale=-2:'trunc(min(ih,${MAX_H})/2)*2',format=yuv420p"
    else
        vf="scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"
    fi

    CUR_TMP="$tmp"
    if ! ffmpeg -nostdin -hide_banner -loglevel error -stats -y \
            -fflags +genpts+discardcorrupt -err_detect ignore_err \
            -i "$src" -map 0:v:0 -map '0:a:0?' -sn -dn \
            -vf "$vf" -fps_mode cfr \
            -c:v libx264 -preset "$PRESET" -crf "$CRF" -profile:v high \
            -c:a aac -b:a 128k \
            -movflags +faststart "$tmp"; then
        echo; echo "  !! ffmpeg failed"; rm -f -- "$tmp"; CUR_TMP=""; return 1
    fi
    echo

    sdur="$(duration "$src")"; ddur="$(duration "$tmp")"
    if awk -v d="$ddur" 'BEGIN{ exit !(d <= 0) }'; then
        echo "  !! output has no duration"; rm -f -- "$tmp"; CUR_TMP=""; return 1
    fi
    if ! decord_ok "$tmp"; then
        echo "  !! decord still cannot read the converted file"; rm -f -- "$tmp"; CUR_TMP=""; return 1
    fi
    # Corrupt segments can be dropped on the way through; say so, don't hide it.
    if awk -v s="$sdur" -v d="$ddur" 'BEGIN{ exit !(s > 0 && d < s * 0.95 - 1) }'; then
        echo "  ?? WARNING: output is ${ddur}s but source reported ${sdur}s -- part of the source was undecodable"
    fi

    mkdir -p "$DIR/originals"
    local keep="$DIR/originals/$(basename "$src")"
    [ -e "$keep" ] && keep="$keep.$(date +%s)"
    # Never leave a moment where the file is missing (a benchmark may be running):
    # same path -> hard-link the original away, then atomically rename over it;
    # new extension -> put the .mp4 in place first, then move the old file away.
    if [ "$final" = "$src" ]; then
        { ln -- "$src" "$keep" 2>/dev/null || cp -p -- "$src" "$keep"; } && mv -f -- "$tmp" "$final"
    else
        mv -f -- "$tmp" "$final" && mv -- "$src" "$keep"
    fi || { echo "  !! could not move files into place"; return 1; }
    CUR_TMP=""
    echo "  -> $final   (original kept: $keep)"
    return 0
}

# --- main ------------------------------------------------------------------
echo "dir: $DIR   crf=$CRF preset=$PRESET${MAX_H:+ max-height=$MAX_H}$([ "$FORCE" = 1 ] && echo ' FORCE')$([ "$DRY" = 1 ] && echo ' DRY-RUN')"

n_total=0; n_ok=0; n_conv=0; n_fail=0; n_skip=0; failed=()
while IFS= read -r -d '' f; do
    name="$(basename "$f")"
    if [ -n "$MIN_ID$MAX_ID" ] && [[ "${name%.*}" =~ ([0-9]+)$ ]]; then
        id=$((10#${BASH_REMATCH[1]}))
        if { [ -n "$MIN_ID" ] && [ "$id" -lt "$MIN_ID" ]; } || { [ -n "$MAX_ID" ] && [ "$id" -gt "$MAX_ID" ]; }; then
            n_skip=$((n_skip + 1)); continue
        fi
    fi
    n_total=$((n_total + 1))
    if [ "$FORCE" = 1 ]; then
        reason="forced"
    elif reason="$(needs_conversion "$f")"; then
        :
    else
        n_ok=$((n_ok + 1)); echo "[ok]      $name"; continue
    fi

    if [ "$DRY" = 1 ]; then
        echo "[convert] $name   ($reason)"; n_conv=$((n_conv + 1)); continue
    fi
    echo "[convert] $name   ($reason)"
    if convert_one "$f"; then
        n_conv=$((n_conv + 1))
    else
        n_fail=$((n_fail + 1)); failed+=("$name")
    fi
done < <(find "$DIR" -maxdepth 1 -type f \
            \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.avi' -o -iname '*.mov' \
               -o -iname '*.webm' -o -iname '*.m4v' -o -iname '*.flv' -o -iname '*.wmv' \
               -o -iname '*.mpg' -o -iname '*.mpeg' -o -iname '*.ts' \) \
            ! -name '.*' -print0 | sort -zV$([ "$REV" = 1 ] && echo r))

echo
echo "=============================================================="
range_note=""
[ -z "$MIN_ID$MAX_ID" ] || range_note=" (skipped $n_skip outside ID range ${MIN_ID:-0}..${MAX_ID:-inf})"
echo " videos: $n_total$range_note   already fine: $n_ok   $([ "$DRY" = 1 ] && echo 'would convert' || echo converted): $n_conv   failed: $n_fail"
if [ "$n_fail" -gt 0 ]; then
    printf '  failed: %s\n' "${failed[@]}"
fi
[ "$DRY" = 1 ] || [ "$n_conv" -eq 0 ] || echo " originals are in $DIR/originals/ (delete them once you are happy)"
echo "=============================================================="
[ "$n_fail" -eq 0 ]
