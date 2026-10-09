#!/usr/bin/env bash
# Builds test clips in testdata/ from OpenCV's pedestrian sample (vtest.avi).
# Each variant isolates one behaviour the analyzer should get right.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p testdata
cd testdata

if [[ ! -f vtest.avi ]]; then
  curl -sSfL -o vtest.avi https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi
fi

ff() { ffmpeg -v error -y "$@"; }

ff -i vtest.avi -c:v libx264 -pix_fmt yuv420p people.mp4

# Captions that change every 2.5 s for the whole clip.
lines=("Welcome to the square" "People are walking past the camera" "Some of them stop for a moment"
       "Others keep moving toward the exit" "The afternoon light is changing" "Nobody seems to be in a hurry")
srt_time() { printf '00:%02d:%02d,%03d' $(($1 / 60000)) $(($1 / 1000 % 60)) $(($1 % 1000)); }
: > captions.srt
for i in $(seq 0 31); do
  printf '%d\n%s --> %s\n%s\n\n' $((i + 1)) "$(srt_time $((i * 2500)))" "$(srt_time $(((i + 1) * 2500)))" \
    "${lines[$((i % ${#lines[@]}))]}" >> captions.srt
done

ff -i vtest.avi -vf "subtitles=captions.srt:force_style='FontSize=22,Outline=2'" -c:v libx264 -pix_fmt yuv420p people_burned.mp4
ff -i people.mp4 -i captions.srt -c:v copy -c:s mov_text -metadata:s:s:0 language=eng people_softsub.mp4
ff -i people.mp4 -i captions.srt -c:v copy -c:s srt -metadata:s:s:0 language=eng people_softsub.mkv
ff -i vtest.avi -vf "drawtext=text='SAMPLE CHANNEL':fontcolor=white:fontsize=28:borderw=2:x=(w-tw)/2:y=h-60" \
   -c:v libx264 -pix_fmt yuv420p people_watermark.mp4
ff -f lavfi -i "mandelbrot=size=640x360:rate=25" -t 15 -c:v libx264 -pix_fmt yuv420p no_people.mp4

ls -la
