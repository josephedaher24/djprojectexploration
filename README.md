# djprojectexploration

Exploration repo for DJ/music compatibility tooling. The project analyzes local audio collections, writes playlist-level
feature bundles, and generates interactive tools for exploring track similarity, building energy-aware DJ sequences, and
auditioning transitions.

```bash
uv sync
```

## Current Focus

The most active workflows are:

- **DJ Sequence Builder**: a PaCMAP-based track browser, recommendation surface, sequence editor, library view, diagnostics
  panel, and transition preview workflow.
- **Interactive DJ PaCMAP / UMAP**: a standalone embedding visualization with controls for balancing style/genre against
  tempo, groove, and key compatibility.
- **Transition tools**: local transition rendering, waveform/automation visualization, and a small web workbench for
  testing cue points, pitch shifts, crossfades, EQ, and filters.
- **Playlist feature pipelines**: MAEST, chroma, tempo, groove, waveform, energy, and optional DEAM feature extraction into
  compact NPZ files.

Generated HTML, snippets, artwork, waveforms, transition renders, and local audio are treated as disposable artifacts.
Commit source code, docs, small CSV metadata, intentionally shared feature bundles, and the packaged frontend assets under
`src/djprojectexploration/templates/` and `src/djprojectexploration/static/`; avoid committing generated exports unless
they are explicitly part of a handoff.

## Requirements

The project is managed with `uv` and targets Python 3.13+. Key dependencies include:

- `numpy`, `scipy`: numeric work, distance matrices, DSP
- `librosa`, `soundfile`, `pyloudnorm`: audio loading, rendering, waveform and loudness features
- `essentia`, `essentia-tensorflow`, `tensorflow`: MAEST, TempoCNN, DEAM, and related model inference
- `plotly`, `pacmap`, `umap-learn`: interactive plots and dimensionality reduction
- `mutagen`: MP3 duration/tag/artwork metadata
- `matplotlib`, `kaleido`, Jupyter packages: exploratory analysis and export support

Some code paths import `PIL.Image` for artwork extraction. If your environment does not already provide Pillow
transitively, add it explicitly.

## Model Files

Essentia model files are expected under `models/` when needed:

```text
models/discogs-maest-30s-pw-519l-2.pb
models/deeptemp-k16-3.pb
models/msd-musicnn-1.pb
models/deam-msd-musicnn-2.pb
```

Raw models can be downloaded from Essentia's model catalog. Tempo tooling can also use `--auto-download-model` for the
default TempoCNN model.

## Data Layout

The tools are organized around playlist CSVs plus playlist-level NPZ feature bundles.

```text
music/<mix-slug>/<mix_slug>_tracks.csv
music/<mix-slug>/<mix_slug>_cues.csv
music/<mix-slug>/<mix_slug>_preview_sections.csv

data/maest_embeddings/<csv-stem>.npz
data/chroma_embeddings/<csv-stem>.npz
data/tempo_embeddings/<csv-stem>.npz
data/groove_embeddings/<csv-stem>.npz
data/waveform_features/<csv-stem>.npz
data/energy_features/<csv-stem>_energy_features.csv
data/energy_embeddings/aries_ara_energy_features.npz

data/snippets/<csv-stem>/
data/artwork/
data/transitions/
data/exports/
```

The default demo still centers on `aries-mix` and `ara-mix`, but most playlist commands accept either a tracklist path or
repeatable `--mix` arguments.

To start from a folder of audio files, generate a normalized tracklist first:

```bash
uv run djprojectexploration-build-tracklist path/to/music-folder --name my-set
```

Default output is written inside the music folder:

```text
path/to/music-folder/my_set_tracks.csv
```

The generated CSV includes `track_number`, `title`, `artists`, `mp3_name`, `filepath`, `key`, `bpm`, `onset-time`,
`genre`, `key shift`, and `energy`. Tag values are filled from audio metadata when available; cue/onset, key-shift, and
energy can be added later by importers or manual labeling.

For a fuller app-ready build, use the dataset orchestrator:

```bash
uv run djprojectexploration-build-dataset path/to/music-folder --name my-set
```

This creates `music/my-set/my_set_tracks.csv`, then runs snippet caching, waveform extraction, playlist embeddings,
energy feature/NPZ export, and sequence-builder HTML export. Use `--skip-embeddings`, `--skip-energy`, or
`--skip-app-export` while iterating.

## Source Layout

Most production code lives under `src/djprojectexploration/`. The main app/export modules are:

- `energy_sequence_builder.py`: builds the sequence-builder data payload, PaCMAP layouts, recommendations, and standalone
  HTML export.
- `interactive_pacmap_knn_simplex.py`: canonical standalone PaCMAP/UMAP visualization exporter.
- `transition_workbench.py`: local transition-rendering workbench and API.
- `sequence_builder_app.py`: local served sequence-builder app that combines the static sequence UI with transition
  rendering endpoints.

Frontend bundles are no longer embedded directly in the Python modules:

- `templates/*.html`: reusable HTML shells and control fragments.
- `static/*.css`: packaged CSS for standalone exports and local apps.
- `static/*.js`: packaged browser logic for the sequence builder, PaCMAP/UMAP visualizer, and workbench.
- `frontend_assets.py`: loads packaged templates/static files and renders self-contained HTML documents.
- `local_http.py`: shared helpers for JSON/text responses, byte-range file serving, and artifact routes used by the local
  apps.

The exporters still write single-file standalone HTML by inlining the packaged CSS and JS at export time. The local served
apps use the same generated HTML surface, plus HTTP endpoints for audio/artifact access.

For details on modifying browser behavior, see `docs/frontend_architecture.md`.

## Main Apps

### Sequence Builder

Generate a standalone HTML snapshot:

```bash
uv run djprojectexploration-dj-sequence
```

Default output:

```text
data/exports/dj_sequence_builder.html
```

This static export is self-contained and is useful for layout/debugging work. It can open from disk or be served from
`data/exports/`, but local transition rendering and some audio/artifact routes require the served app.

Run the served app when you want full local audio, artwork, waveform, and transition-render support:

```bash
uv run djprojectexploration-sequence-builder-app
```

Default URL:

```text
http://127.0.0.1:8770/
```

Useful options:

```bash
uv run djprojectexploration-sequence-builder-app \
  --mix aries-mix \
  --mix ara-mix \
  --sequence-length 12
```

The sequence builder uses MAEST/style, tempo, groove, and chroma/key compatibility. In dynamic layout mode it precomputes a
grid of PaCMAP layouts and interpolates between them in the browser:

```bash
uv run djprojectexploration-sequence-builder-app --dynamic-layout --step 0.1
```

### Interactive DJ PaCMAP / UMAP

Generate the standalone PaCMAP visualization:

```bash
uv run djprojectexploration-dj-pacmap
```

Default output:

```text
data/exports/aries_mix_tracks__ara_mix_tracks_interactive_dj_pacmap.html
```

Useful options:

```bash
uv run djprojectexploration-dj-pacmap \
  --reducer pacmap \
  --control-mode genre-mixability \
  --n-neighbors 10 \
  --mn-ratio 0.5 \
  --fp-ratio 1.5 \
  --step 0.1
```

Generate the same interactive HTML shell with UMAP layouts instead of PaCMAP:

```bash
uv run djprojectexploration-dj-pacmap \
  --reducer umap \
  --umap-min-dist 0.1 \
  --step 0.1
```

The older experimental UMAP and PaCMAP scripts have been retired. Use `djprojectexploration-dj-pacmap` for standalone
HTML exports and reducer comparisons.

To serve a generated static export for quick browser checks:

```bash
python3 -m http.server 8788 --directory data/exports
```

Then open the generated file, for example:

```text
http://127.0.0.1:8788/dj_sequence_builder.html
```

## Transition Tools

Render a transition from a tracklist and cue table:

```bash
uv run djprojectexploration-transition-preview \
  1 \
  2 \
  --tracklist music/aries-mix/aries_mix_tracks.csv \
  --overlap-bars 16 \
  --front-padding-bars 2 \
  --back-padding-bars 2 \
  --preset auto
```

Render output is written under `data/transitions/<transition-id>/`:

```text
preview.wav
transition.json
transition_waveforms.json
transition_visualizer.html
```

Export a standalone visualizer for an existing render:

```bash
uv run djprojectexploration-transition-visualizer data/transitions/<transition-id>
```

Run the transition workbench:

```bash
uv run djprojectexploration-transition-workbench
```

The sequence builder app reuses this backend through `POST /api/render-transition`.

## Feature Pipelines

Build one compressed NPZ per playlist:

```bash
uv run djprojectexploration-build-tracklist music/my-set --name my-set
uv run djprojectexploration-maest-playlist music/ara-mix/ara_mix_tracks.csv --music-dir music/ara-mix
uv run djprojectexploration-chroma-playlist music/ara-mix/ara_mix_tracks.csv --music-dir music/ara-mix
uv run djprojectexploration-tempo-playlist music/ara-mix/ara_mix_tracks.csv --music-dir music/ara-mix
uv run djprojectexploration-groove-playlist music/ara-mix/ara_mix_tracks.csv --music-dir music/ara-mix
uv run djprojectexploration-deam-playlist music/ara-mix/ara_mix_tracks.csv --music-dir music/ara-mix
```

Generate waveform features used by the served sequence builder:

```bash
uv run djprojectexploration-waveforms music/ara-mix/ara_mix_tracks.csv --music-dir music/ara-mix
```

Generate energy feature CSVs:

```bash
uv run djprojectexploration-energy-features music/ara-mix/ara_mix_tracks.csv --music-dir music/ara-mix
```

Create the canonical app-consumable energy NPZ from one or more energy feature CSVs:

```bash
uv run djprojectexploration-energy-npz \
  data/energy_features/aries_mix_tracks_energy_features.csv \
  data/energy_features/ara_mix_tracks_energy_features.csv \
  --name aries_ara_energy_features
```

The NPZ writer implements the replaceable full energy model from `notebooks/energy_analysis.ipynb`: BPM plus simple,
advanced, and loudness/RMS audio features with a Gaussian Tweedie/ridge pipeline. If labeled `energy` values are present,
it fits the model and writes `glm_energy_pred`; otherwise it still writes the canonical NPZ with unavailable predictions.

Useful common options:

- `--output-file <path>` to override the exact output path
- `--output-dir <path>` to override the default feature directory
- `--skip-missing-audio` to skip rows whose audio files are unavailable
- `--music-dir <path>` when the CSV has `mp3_name` but no absolute `filepath`

## Cue, Preview, and Playlist Utilities

Import Rekordbox cue points:

```bash
uv run djprojectexploration-rekordbox-cues \
  path/to/rekordbox.xml \
  --tracklist music/aries-mix/aries_mix_tracks.csv
```

Generate preview-section CSVs from snippet metadata:

```bash
uv run djprojectexploration-preview-sections
```

Create snippet caches for lightweight playback:

```bash
uv run djprojectexploration-snippet-cache music/ara-mix/ara_mix_tracks.csv --music-dir music/ara-mix
```

Export tracks from a local Apple Music playlist on macOS:

```bash
scripts/music/export_playlist_tracks_csv.sh "My Playlist Name" data/exports/my_playlist_tracks.csv
scripts/music/export_playlist_filepaths.sh "My Playlist Name" data/exports/my_playlist_filepaths.txt
```

Compatibility wrappers:

```bash
scripts/playlist_csv.sh
scripts/playlist_files.sh
```

## Single-Track Extractors

These are useful for debugging individual audio files:

```bash
uv run djprojectexploration-maest --audio-file path/to/file.mp3
uv run djprojectexploration-chroma --audio-file path/to/file.mp3
uv run djprojectexploration-tempo --audio-file path/to/file.mp3
uv run djprojectexploration-groove --audio-file path/to/file.mp3
uv run djprojectexploration-deam --audio-file path/to/file.mp3
```

## Cleaning Generated Artifacts

```bash
make clean-snippets
make clean-transitions
make clean-interactive-html
make clean-generated
```

Generated paths such as `data/exports/`, `data/snippets/`, `data/transitions/`, `data/artwork/`, waveform caches, and local
MP3 files are ignored for new files. Some historical artifacts may still be tracked; treat those intentionally when
preparing commits.

## Development Notes

```bash
uv run python -m djprojectexploration
uv run python -c "import essentia, librosa, numpy, scipy, soundfile, matplotlib; print('imports ok')"
```

When preparing a commit, usually include:

- source files under `src/djprojectexploration/`
- scripts under `scripts/`
- `pyproject.toml` and `uv.lock`
- small tracklist/cue CSV metadata
- documentation

Usually leave out:

- generated HTML and ZIP exports
- audio snippets, waveform caches, artwork, transition renders
- local MP3 files and metadata rewrites
- large exploratory notebooks unless intentionally part of the change
