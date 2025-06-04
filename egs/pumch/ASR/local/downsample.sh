#!/usr/bin/env bash

# downsample.sh: Downsample a video to a given height (keeping aspect ratio) and save to another directory.
# Usage: ./downsample.sh /path/to/input_video.mp4 /path/to/output_dir 480

set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_dir> <output_dir> <target_height>"
    echo "Example: $0 /home/user/input_videos /home/user/downsampled 480"
    exit 1
fi

INPUT="$1"
OUTPUT_DIR="$2"
TARGET_HEIGHT="$3"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check that input directory exists
if [ ! -d "$INPUT" ]; then
    echo "Error: Input directory '$INPUT' not found."
    exit 1
fi

# Iterate over all .mp4 and .mov files in the input directory
for INPUT_FILE in "$INPUT"/*.mp4 "$INPUT"/*.mov; do
    [ -f "$INPUT_FILE" ] || continue

    # Extract base name and extension
    FILENAME="$(basename "$INPUT_FILE")"
    BASENAME="${FILENAME%.*}"
    EXTENSION="${FILENAME##*.}"

    # Build output path
    OUTPUT_PATH="$OUTPUT_DIR/${BASENAME}.${EXTENSION}"

    # Run ffmpeg to scale video to target height (width auto-calculated to preserve aspect ratio), copy audio
    ffmpeg -i "$INPUT_FILE" \
        -vf "scale=-2:${TARGET_HEIGHT}" \
        -c:v libx264 -crf 23 -c:a copy \
        "$OUTPUT_PATH"

    echo "Downsampled $INPUT_FILE saved to: $OUTPUT_PATH"
done
