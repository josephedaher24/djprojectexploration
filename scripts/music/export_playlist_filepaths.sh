#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat <<'USAGE' >&2
Usage: scripts/music/export_playlist_filepaths.sh "Playlist Name" [output_file]

Examples:
  scripts/music/export_playlist_filepaths.sh "ARA MIX"
  scripts/music/export_playlist_filepaths.sh "ARA MIX" data/exports/ara_mix_filepaths.txt
USAGE
  exit 1
fi

playlist_name="$1"
slug="$(printf '%s' "$playlist_name" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '_')"
output_file="${2:-data/exports/${slug}_filepaths.txt}"

mkdir -p "$(dirname "$output_file")"
tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

osascript - "$playlist_name" <<'APPLESCRIPT' > "$tmp_file"
on run argv
  set playlistName to item 1 of argv

  tell application "Music"
    try
      set p to playlist playlistName
    on error
      error ("Playlist not found: " & playlistName) number 1
    end try

    set outLines to {}

    repeat with t in tracks of p
      try
        set loc to location of t
        if loc is not missing value then
          set end of outLines to (POSIX path of loc)
        end if
      end try
    end repeat

    set AppleScript's text item delimiters to linefeed
    return outLines as text
  end tell
end run
APPLESCRIPT

mv "$tmp_file" "$output_file"
trap - EXIT

line_count="$(wc -l < "$output_file" | tr -d ' ')"
echo "Wrote ${line_count} file paths to ${output_file}"
