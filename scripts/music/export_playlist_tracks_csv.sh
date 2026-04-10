#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat <<'USAGE' >&2
Usage: scripts/music/export_playlist_tracks_csv.sh "Playlist Name" [output_file]

Examples:
  scripts/music/export_playlist_tracks_csv.sh "ARA MIX"
  scripts/music/export_playlist_tracks_csv.sh "ARA MIX" data/exports/ara_mix_tracks.csv
USAGE
  exit 1
fi

playlist_name="$1"
slug="$(printf '%s' "$playlist_name" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_')"
output_file="${2:-data/exports/${slug}_tracks.csv}"

mkdir -p "$(dirname "$output_file")"
tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

osascript - "$playlist_name" <<'APPLESCRIPT' > "$tmp_file"
on esc(v)
  set s to v as text

  set AppleScript's text item delimiters to "\""
  set parts to text items of s
  set AppleScript's text item delimiters to "\"\""
  set s to parts as text

  set AppleScript's text item delimiters to linefeed
  set parts to text items of s
  set AppleScript's text item delimiters to " "
  set s to parts as text

  set AppleScript's text item delimiters to return
  set parts to text items of s
  set AppleScript's text item delimiters to " "
  set s to parts as text

  set AppleScript's text item delimiters to ""
  return "\"" & s & "\""
end esc

on run argv
  set playlistName to item 1 of argv

  tell application "Music"
    try
      set p to playlist playlistName
    on error
      error ("Playlist not found: " & playlistName) number 1
    end try

    set rows to {"name,artist,album,genre,bpm,filepath"}

    repeat with t in tracks of p
      set nm to ""
      set ar to ""
      set al to ""
      set ge to ""
      set bp to ""
      set fp to ""

      try
        set nm to name of t
      end try
      try
        set ar to artist of t
      end try
      try
        set al to album of t
      end try
      try
        set ge to genre of t
      end try
      try
        set bp to (bpm of t) as text
      end try
      try
        set loc to location of t
        if loc is not missing value then
          set fp to POSIX path of loc
        end if
      end try

      set end of rows to (my esc(nm) & "," & my esc(ar) & "," & my esc(al) & "," & my esc(ge) & "," & my esc(bp) & "," & my esc(fp))
    end repeat

    set AppleScript's text item delimiters to linefeed
    return rows as text
  end tell
end run
APPLESCRIPT

mv "$tmp_file" "$output_file"
trap - EXIT

total_lines="$(wc -l < "$output_file" | tr -d ' ')"
if [[ "$total_lines" -gt 0 ]]; then
  track_count=$((total_lines - 1))
else
  track_count=0
fi

echo "Wrote ${track_count} tracks to ${output_file}"
