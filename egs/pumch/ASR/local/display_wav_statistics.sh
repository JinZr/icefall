#!/usr/bin/env bash
# calc_wav_stats.sh  <folder>
# Prints max, min, and mean duration (seconds) of all *.wav in the folder (recursively).

set -euo pipefail

dir=${1:-.}

# Track min/max durations and their corresponding file paths
min_dur=""
max_dur=0
min_file=""
max_file=""

# Requires ffprobe (FFmpeg). 
# Collect durations in seconds with millisecond precision.
durations=()
while IFS= read -r -d '' f; do
    dur=$(ffprobe -v error -select_streams a:0 -show_entries stream=duration \
                  -of default=noprint_wrappers=1:nokey=1 "$f")
    # Handle possible N/A
    if [[ $dur != "N/A" && -n $dur ]]; then
        durations+=("$dur")
        # Update minimum
        if [[ -z "$min_dur" ]] || awk -v d1="$dur" -v d2="$min_dur" 'BEGIN{exit(d1<d2?0:1)}'; then
            min_dur="$dur"
            min_file="$f"
        fi
        # Update maximum
        if awk -v d1="$dur" -v d2="$max_dur" 'BEGIN{exit(d1>d2?0:1)}'; then
            max_dur="$dur"
            max_file="$f"
        fi
    fi
done < <(find "$dir" -type f -iname '*.wav' -print0)

if ((${#durations[@]} == 0)); then
    echo "No WAV files found."
    exit 1
fi

# Compute mean duration (to three‑decimal precision)
mean=$(printf '%s\n' "${durations[@]}" | awk '{sum+=$1; n++} END { if (n>0) printf "%.3f", sum/n }')

printf "Files analysed : %d\n" "${#durations[@]}"
printf "Min duration  : %.3f s  ( %s )\n" "$min_dur" "$min_file"
printf "Max duration  : %.3f s  ( %s )\n" "$max_dur" "$max_file"
printf "Mean duration : %s s\n" "$mean"