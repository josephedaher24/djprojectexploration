# djprojectexploration

Minimal Python project scaffold using `uv`, with audio and signal-processing libraries installed.

```bash
uv sync
```

## Requirements

- `numpy` and `scipy` for core numerical and DSP work
- `librosa` for higher-level audio analysis
- `essentia` for music/audio feature extraction
- `soundfile` for reading and writing audio files
- `matplotlib` for waveform and spectrogram visualization

## Model Downloads

Raw model files must be downloaded from [Essentia](https://essentia.upf.edu/models.html), including:
- deam-msd-musicnn-2.pb
- discogs-maest-30s-pw-519l-2.pb
- msd-musicnn-1.pb

## Development

```bash
uv add <package>
uv run python -m djprojectexploration
```

## Apple Music playlist exports

Create metadata and file-path exports from a local Apple Music playlist without moving files into a new folder.

Export tracks to CSV (`name,artist,album,genre,bpm,filepath`):

```bash
scripts/music/export_playlist_tracks_csv.sh "My Playlist Name"
```

Export just file paths:

```bash
scripts/music/export_playlist_filepaths.sh "My Playlist Name"
```

Both commands accept an optional second argument for output path:

```bash
scripts/music/export_playlist_tracks_csv.sh "My Playlist Name" data/exports/my_playlist_tracks.csv
scripts/music/export_playlist_filepaths.sh "My Playlist Name" data/exports/my_playlist_filepaths.txt
```

Compatibility wrappers also exist:
- `scripts/playlist_csv.sh`
- `scripts/playlist_files.sh`

Notes:
- Run these from macOS with the Music app library available.
- Streaming-only tracks with no local file location are included in CSV metadata but have an empty `filepath`.
- Path-only export skips tracks that have no local file location.

## Playlist embedding pipelines (NPZ only)

Build one compressed embedding file per playlist (no per-track embedding JSON files):

```bash
uv run djprojectexploration-maest-playlist data/exports/dataset_tracks.csv
uv run djprojectexploration-chroma-playlist data/exports/dataset_tracks.csv
uv run djprojectexploration-tempo-playlist data/exports/dataset_tracks.csv
uv run djprojectexploration-deam-playlist data/exports/dataset_tracks.csv
```

Defaults:
- MAEST output: `data/maest_embeddings/<playlist>_tracks.npz`
- Chroma output: `data/chroma_embeddings/<playlist>_tracks.npz`
- Tempo output: `data/tempo_embeddings/<playlist>_tracks.npz`
- DEAM output: `data/deam_embeddings/<playlist>_tracks.npz`
- `<playlist>_tracks.npz` comes from the CSV filename stem (for example, `ara_mix_tracks.csv` -> `ara_mix_tracks.npz`)

When your CSV does not include `filepath`, pass `--music-dir` to resolve `mp3_name`:

```bash
uv run djprojectexploration-maest-playlist music/ara-mix/ara_mix_tracks.csv --music-dir music/ara-mix
uv run djprojectexploration-chroma-playlist music/ara-mix/ara_mix_tracks.csv --music-dir music/ara-mix
uv run djprojectexploration-tempo-playlist music/ara-mix/ara_mix_tracks.csv --music-dir music/ara-mix
uv run djprojectexploration-deam-playlist music/ara-mix/ara_mix_tracks.csv --music-dir music/ara-mix
```

Useful options:
- `--output-file <path>` to override the exact NPZ output path
- `--skip-missing-audio` to skip tracks whose audio files are missing
- Chroma-specific: `--exclude-key-features`, `--sample-rate`, `--frame-size`, `--hop-size`, `--chroma-bins`
- Tempo-specific: `--model-file`, `--auto-download-model`, `--sample-rate`, `--snippet-length-sec`, `--window-sec`, `--hop-sec`, `--rms-percentile`
- DEAM-specific: `--embedding-backend`, `--embedding-model-file`, `--regression-model-file`, `--embedding-output`, `--regression-output`, `--sample-rate`

## Snippet cache (for sharing + interactive playback)

Build a lightweight snippet cache (WAV files + manifest) from a playlist CSV:

```bash
uv run djprojectexploration-snippet-cache data/exports/dataset_tracks.csv
```

Defaults:
- Output dir: `data/snippets/<csv-stem>/`
- Manifest: `data/snippets/<csv-stem>/snippets_manifest.csv`
- Snippet length: `8.0s`
- Snippet sample rate: `22050 Hz` (smaller cache files)
- Selection strategy: highest RMS window within the middle `66%` of each track

Useful options:
- `--output-dir <path>` to override snippet cache location
- `--music-dir <path>` when CSV does not include `filepath`
- `--snippet-seconds <float>`
- `--middle-fraction <float>`
- `--hop-seconds <float>`
- `--target-sample-rate <int>`
- `--overwrite` to recompute existing snippet files
- `--skip-missing-audio` to skip tracks with missing files

## MAEST embeddings (Essentia)

```bash
uv run djprojectexploration-maest \
  --model-file models/discogs-maest-30s-pw-519l-2.pb \
  --output-file music/maest_embedding_discogs-maest-30s-pw.json
```

Notes:
- `--audio-file` is optional; if omitted, the first `.mp3` in `music/` is used.
- If you do not have MAEST support in your Essentia install, use a TensorFlow-enabled build such as `essentia-tensorflow`.

## Chroma embeddings (Essentia HPCP)

```bash
uv run djprojectexploration-chroma \
  --output-file music/chroma_embedding.json
```

Notes:
- `--audio-file` is optional; if omitted, the first `.mp3` in `music/` is used.
- Default behavior is unit-sum normalization without centering (`center_baseline=None`).
- Use `--center-baseline 0.0833333333` to enable 1/12 centering.
- Use `--exclude-key-features` to return only the base beat-synchronous mean/std chroma embedding.

## Tempo embeddings (Essentia TempoCNN)

```bash
uv run djprojectexploration-tempo \
  --output-file music/tempo_embedding_tempocnn.json
```

Notes:
- `--audio-file` is optional; if omitted, the first `.mp3` in `music/` is used.
- The extractor returns single-track JSON with `tempo_bpm`, break-aware `confidence`, and local-window tempo/probability diagnostics.
- By default it uses `models/deeptemp-k16-3.pb`; use `--auto-download-model` if the model is missing.
- Useful controls: `--snippet-length-sec`, `--window-sec`, `--hop-sec`, `--rms-percentile`, `--sample-rate`.

## DEAM valence/arousal embeddings (Essentia)

```bash
uv run djprojectexploration-deam \
  --embedding-model-file models/msd-musicnn-1.pb \
  --regression-model-file models/deam-msd-musicnn-2.pb \
  --output-file music/deam_embedding_musicnn.npz
```

Notes:
- `--audio-file` is optional; if omitted, the first `.mp3` in `music/` is used.
- Output is a single-track NPZ with one-row `embeddings`, summary stats (`mean/min/max/std` for valence and arousal), and flattened raw segment series (`deam_valence_series_flat`, `deam_arousal_series_flat`).
- Playlist NPZ output uses the same schema plus per-track `deam_series_start_index`/`deam_series_count` so variable-length series can be reconstructed per track.

## Quick import check

```bash
.venv/bin/python -c "import essentia, librosa, numpy, scipy, soundfile, matplotlib; print('imports ok')"
```
