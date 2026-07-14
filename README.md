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

Generated HTML, artwork, waveform caches, transition renders, optional snippets, and local audio are treated as
disposable artifacts. Commit source code, docs, small CSV metadata, intentionally shared feature/model bundles, and the
packaged frontend assets under `src/djprojectexploration/templates/` and `src/djprojectexploration/static/`; avoid
committing generated exports unless they are explicitly part of a handoff.

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

data/maest_embeddings/<csv-stem>.npz
data/maest_embeddings/<csv-stem>_peak30.npz
data/chroma_embeddings/<csv-stem>.npz
data/tempo_embeddings/<csv-stem>.npz
data/groove_embeddings/<csv-stem>.npz
data/waveform_features/<csv-stem>.npz
data/energy_features/<csv-stem>_energy_features.csv
data/energy_embeddings/<dataset>_energy_features.npz
data/energy_models/<model-name>.joblib
data/energy_models/<model-name>.json

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

This creates `music/my-set/my_set_tracks.csv`, validates it, then runs waveform extraction, MAEST, chroma, tempo,
groove, energy feature/NPZ export, sequence-builder HTML export, standalone PaCMAP export, and a build
manifest. Each stage prints start/done timing so slow extraction steps are easier to spot. For an existing CSV, pass it
explicitly:

```bash
uv run djprojectexploration-build-dataset path/to/music-folder \
  --name my-set \
  --tracklist path/to/my_tracks.csv
```

The build writes:

```text
data/exports/<name>_validation.json
data/exports/<name>_build_manifest.json
data/exports/<name>_sequence_builder.html
data/exports/<name>_pacmap.html
```

By default, existing feature bundles are reused. Use `--force` to regenerate, `--skip-waveforms`, `--skip-embeddings`,
`--skip-energy`, `--skip-sequence-export`, or `--skip-pacmap-export` while iterating. In a full build, tempo is extracted
once and then reused by groove and energy when those stages need BPM estimates. Energy NPZ generation applies the frozen
`data/energy_models/energycurvedataset_maest_full_plus_peak30_pca64_ridge.joblib` model by default; the builder also
creates the required peak-RMS 30-second MAEST bundle when that model is active. Pass `--energy-model-file` to use a
different frozen model, or `--refit-energy-model` when you intentionally want to fit from the dataset being built. The
dataset builder no longer creates snippet caches; use `djprojectexploration-snippet-cache` directly for older notebooks
or preview-section utilities.

### Energy Features and Models

Energy feature extraction reads the source audio files listed by the tracklist, not cached snippets. By default it
analyzes the full decoded song and also finds the highest-RMS 30-second window using a sliding RMS window. Extracted
columns are namespaced as `full_*` and `peak30_*`, with peak-window metadata such as `peak30_start_sec`,
`peak30_end_sec`, and `peak30_window_rms_db`. Use `--analysis-seconds` only when you intentionally want to analyze a
prefix instead of the full song.

Extract audio-derived energy features once:

```bash
uv run djprojectexploration-energy-features music/dj-dataset-1/dj_dataset_1_tracks.csv
```

Compare handcrafted feature sets from labeled feature CSVs:

```bash
uv run djprojectexploration-energy-compare-models \
  data/energy_features/dj_dataset_1_tracks_energy_features.csv
```

Available handcrafted feature sets include `full_only`, `peak30_only`, `full_plus_peak30`, and no-BPM variants such as
`full_plus_peak30_no_bpm`. The default handcrafted set is `full_plus_peak30_no_bpm`.

Train and freeze a reusable handcrafted model from labeled feature CSVs:

```bash
uv run djprojectexploration-energy-train-model \
  data/energy_features/aries_mix_tracks_energy_features.csv \
  data/energy_features/ara_mix_tracks_energy_features.csv \
  --name full_tweedie_ridge_aries_ara
```

This writes `data/energy_models/full_tweedie_ridge_aries_ara.joblib` plus a JSON metadata sidecar. Apply that frozen
model to any already-extracted feature CSV without re-reading audio. Predictions from a frozen model are clipped to the
1-9 energy scale by default:

```bash
uv run djprojectexploration-energy-npz \
  data/energy_features/dj_dataset_1_tracks_energy_features.csv \
  --name dj_dataset_1_energy_features \
  --model-file data/energy_models/full_tweedie_ridge_aries_ara.joblib
```

Omit `--model-file` when you want to refit directly from the provided feature CSV rows. That mode requires labeled
`energy` values in the input CSVs.

The default reusable app model artifact is
`data/energy_models/energycurvedataset_maest_full_plus_peak30_pca64_ridge.joblib`. It stores a ridge model trained on
PCA-64 components from full-track MAEST embeddings plus PCA-64 components from peak30 MAEST embeddings. Applying that
artifact writes predicted energy values for new datasets without fitting on the target dataset. This requires matching
MAEST files in `data/maest_embeddings/`:

```text
data/maest_embeddings/<tracklist-stem>.npz
data/maest_embeddings/<tracklist-stem>_peak30.npz
```

Create an app-consumable energy NPZ for a new dataset with the frozen model:

```bash
uv run djprojectexploration-energy-npz \
  data/energy_features/dj_dataset_1_tracks_energy_features.csv \
  --name dj_dataset_1_energy_features \
  --model-file data/energy_models/energycurvedataset_maest_full_plus_peak30_pca64_ridge.joblib
```

Retrain the canonical MAEST full+peak30 model only when you have labeled energy rows and the corresponding full/peak30
MAEST bundles:

```bash
uv run djprojectexploration-energy-train-model \
  data/energy_features/energycurvedataset_tracks_energy_features.csv \
  --model maest_full_plus_peak30_pca64_ridge \
  --name energycurvedataset_maest_full_plus_peak30_pca64_ridge
```

Use `--model full` with `djprojectexploration-energy-npz` or `djprojectexploration-energy-train-model` when you want the
handcrafted-only ridge path instead of the MAEST-backed model.

## Source Layout

Most production code lives under `src/djprojectexploration/`. The main app/export modules are:

- `energy_sequence_builder.py`: builds the sequence-builder data payload, PaCMAP layouts, recommendations, and standalone
  HTML export.
- `interactive_pacmap_knn_simplex.py`: canonical standalone PaCMAP/UMAP visualization exporter.
- `pacmap_settings.py`: shared PaCMAP/reducer settings, defaults, validation, and CLI flags.
- `tracklist_validation.py`: reusable tracklist validation summary used by dataset builds.
- `dataset_builder.py`: full dataset orchestration for tracklists/folders, feature bundles, exports, and manifests.
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

The sequence builder uses MAEST/style, tempo, groove, and chroma/key compatibility. By default, the served app precomputes
a dynamic grid of PaCMAP layouts and interpolates between them in the browser:

```bash
uv run djprojectexploration-sequence-builder-app --step 0.1
```

For a faster fixed-layout startup:

```bash
uv run djprojectexploration-sequence-builder-app --static-layout
```

Reusable PaCMAP settings can also be loaded from JSON presets. Explicit CLI flags override preset values:

```bash
uv run djprojectexploration-sequence-builder-app \
  --mix aries-mix \
  --mix ara-mix \
  --pacmap-preset presets/pacmap/aries_ara_dynamic_pca.json
```

Sequence-builder presets can also include frontend visualization defaults under `ui` and, optionally, the dataset inputs
used by the sequence builder. These values are embedded into the generated app as `app_settings`, shown in the settings
popup, and used to initialize the matching controls:

```json
{
  "n_neighbors": 15,
  "MN_ratio": 1.0,
  "FP_ratio": 1.0,
  "pair_source": "combined-all",
  "distance_combine": "l1",
  "layout_init": "pca",
  "static_layout": false,
  "mix_slugs": ["aries-mix", "ara-mix", "bootes-mix"],
  "energy_npz": "data/energy_embeddings/aries_ara_bootes_energy_features.npz",
  "output_file": "data/exports/aries_ara_bootes_sequence_builder.html",
  "sequence_length": 10,
  "ui": {
    "latent_links_per_track": 3,
    "recommended_links_highlight": 25,
    "point_color": "genre",
    "map_renderer": "webgl",
    "map_fx": true
  }
}
```

In the app, latent links are top-K weighted candidate links per track. Recommended links are the current selected track's
ranked next-track recommendations highlighted on the map. Point coloring supports `genre`, `energy`, `tempo`, `key`, and
`target`; `target` colors each track by selected energy minus the active target energy slot. The map renderer can be
switched between the original Plotly renderer and an experimental WebGL renderer from the app controls or by setting
`ui.map_renderer` to `webgl` in a preset.

Recent sequence-builder interaction conventions:

- Double-click a map point to set or clear Track 1, the locked source for recommendations.
- Single-click another point while Track 1 is set to select Track 2, the candidate target.
- Adding a candidate to the sequence promotes it to Track 1.
- Recommendation rows can be searched and pinned; multiple pins are allowed and are reset when the recommendation source
  changes.
- The main map supports Plotly zoom/pan plus keyboard controls: `+` or `i` to zoom in, `-` to zoom out, arrow keys to pan,
  and `R` to reset. Press `?` in the app for the full shortcut panel.
- The diagnostics view shows the focused transition, score bars, chroma/groove comparisons, sequence transition rows,
  and full-track waveform scrubbers when waveform features are available.

For the current energy curve dataset preset:

```bash
uv run djprojectexploration-sequence-builder-app \
  --pacmap-preset presets/pacmap/energycurvedataset_dynamic.json
```

Presets can also include the sequence-builder dataset inputs, so a combined Aries/Ara/Bootes export can be built with
one argument:

```bash
uv run djprojectexploration-energy-sequence-builder \
  --pacmap-preset presets/pacmap/aries_ara_bootes_dynamic.json
```

Serve the same preset locally:

```bash
uv run djprojectexploration-sequence-builder-app \
  --pacmap-preset presets/pacmap/aries_ara_bootes_dynamic.json
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
  --pair-source combined-all \
  --distance-combine l1 \
  --layout-init neighbor \
  --layout-selection-mode interpolated \
  --step 0.1
```

The same PaCMAP knobs are shared by the standalone PaCMAP exporter, the sequence-builder exporter, the served sequence
builder app, and the dataset builder:

- `--pacmap-preset presets/pacmap/<preset>.json`
- `--n-neighbors`
- `--mn-ratio`
- `--fp-ratio`
- `--pair-source {neighbors-only,combined-all}`
- `--distance-combine {l2,l1}`
- `--layout-init {neighbor,pca,random}`
- `--layout-selection-mode {interpolated,discrete}`
- `--step`
- `--static-layout` where fixed-layout export is supported

The standalone PaCMAP exporter also accepts existing tracklists directly:

```bash
uv run djprojectexploration-dj-pacmap \
  --tracklist path/to/my_tracks.csv \
  --dataset-name my-set \
  --control-mode genre-mixability
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

Generate peak-RMS 30-second MAEST embeddings for the same playlist when using the default MAEST-backed energy model:

```bash
uv run djprojectexploration-maest-playlist \
  music/ara-mix/ara_mix_tracks.csv \
  --music-dir music/ara-mix \
  --section peak30
```

When running standalone commands, pass a previously generated tempo NPZ to groove to avoid re-running TempoCNN for BPM
seeding:

```bash
uv run djprojectexploration-groove-playlist music/ara-mix/ara_mix_tracks.csv \
  --music-dir music/ara-mix \
  --tempo-embeddings data/tempo_embeddings/ara_mix_tracks.npz
```

Generate waveform features used by the served sequence builder. This also stores the full-song peak-RMS preview window
used as the default playback start:

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
  --name aries_ara_energy_features \
  --model-file data/energy_models/energycurvedataset_maest_full_plus_peak30_pca64_ridge.joblib
```

For new app datasets, use the saved model so the target dataset is scored rather than refit. If `--model-file` is
omitted, the NPZ writer fits from the provided rows: the default fit is MAEST full+peak30 PCA64 ridge, and `--model full`
selects the handcrafted ridge model using one of the `--feature-set` options. Refit modes need labeled `energy` values
for meaningful predictions and diagnostics.

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

Create snippet caches for older notebooks or lightweight playback experiments:

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
- intentionally shared energy model artifacts under `data/energy_models/`
- documentation

Usually leave out:

- generated HTML and ZIP exports
- audio snippets, waveform caches, artwork, transition renders
- local MP3 files and metadata rewrites
- large exploratory notebooks unless intentionally part of the change
