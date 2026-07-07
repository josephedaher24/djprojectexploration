# Frontend Architecture
## Asset Layout

Frontend assets live under `src/djprojectexploration/`:

- `templates/standalone_document.html`: shared outer HTML document used by standalone exports.
- `templates/energy_sequence_builder_*.html`: sequence-builder body and control fragments.
- `templates/interactive_pacmap_*.html`: PaCMAP/UMAP body, detail panel, and control fragments.
- `templates/transition_workbench_body.html`: transition workbench body.
- `static/energy_sequence_builder.css` and `.js`: sequence-builder UI behavior.
- `static/interactive_pacmap_simplex.css` and `.js`: three-way PaCMAP/UMAP visualizer behavior.
- `static/interactive_pacmap_genre_mixability.css` and `.js`: four-way genre/mixability visualizer behavior.
- `static/transition_workbench.css` and `.js`: transition workbench behavior.
- `frontend_assets.py`: loads template/static files and renders single-file HTML documents.
- `local_http.py`: shared local HTTP helpers for JSON/text responses, range requests, static file responses, and artifact
  route parsing.

The generated HTML remains self-contained. CSS and JS are loaded from packaged files at export time, then inlined into the
output document.

## Exporters And Assets

`energy_sequence_builder.py` builds the sequence-builder data payload and calls `render_standalone_document()` with:

- `templates/energy_sequence_builder_body.html`
- `templates/energy_sequence_builder_legacy_weights.html` or `templates/energy_sequence_builder_mixability_weights.html`
- `static/energy_sequence_builder.css`
- `static/energy_sequence_builder.js`

`interactive_pacmap_knn_simplex.py` is the canonical PaCMAP/UMAP exporter. It uses:

- `templates/interactive_pacmap_body.html`
- `templates/interactive_pacmap_detail_panel.html`
- `templates/interactive_pacmap_simplex_controls.html`
- `templates/interactive_pacmap_genre_mixability_controls.html`
- `static/interactive_pacmap_simplex.css`
- `static/interactive_pacmap_simplex.js`
- `static/interactive_pacmap_genre_mixability.css`
- `static/interactive_pacmap_genre_mixability.js`

`transition_workbench.py` renders the local workbench with:

- `templates/transition_workbench_body.html`
- `static/transition_workbench.css`
- `static/transition_workbench.js`

## Data Injection

Python is responsible for computing data and injecting it into inert JSON script tags. Browser code reads those tags on
load.

Sequence builder payloads:

- `seq-records-json`: track metadata, URIs, waveform metadata, cue data, and energy fields.
- `seq-sim-json`: transition/similarity scores.
- `seq-points-json`: active layout points.
- `seq-layout-entries-json`: dynamic layout grid entries.
- `seq-config-json`: UI defaults, weighting mode, static/dynamic layout mode, and `app_mode`.

PaCMAP/UMAP payloads:

- `simplex-records-json`: track metadata.
- `simplex-layouts-json`: named layout point arrays.
- `simplex-layout-entries-json`: four-way layout grid entries when using genre/mixability controls.
- `simplex-sim-json`: recommendation payloads.
- `simplex-config-json`: reducer/control settings.

The browser scripts should treat these JSON tags as the source of truth for initial state.

## Static Export Vs Served App

The sequence builder has two modes:

- Static export: `uv run djprojectexploration-dj-sequence`
- Served app: `uv run djprojectexploration-sequence-builder-app`

Both render the same core UI, but only the served app sets `app_mode: true`.

When `app_mode` is false, the transition preview tab can align, swap, clear, and inspect selected tracks, but it does not
show the render button. There is no backend endpoint in a static file.

When `app_mode` is true, `static/energy_sequence_builder.js` loads runtime options from:

- `GET /api/tracks`
- `GET /api/options`

It renders the transition button and submits transition render requests to:

- `POST /api/render-transition`

Generated artifacts are then served through:

- `GET /artifacts/<transition-id>/<filename>`

The standalone transition workbench has a similar backend API:

- `GET /api/tracks`
- `GET /api/options`
- `POST /api/render`
- `GET /artifacts/<transition-id>/<filename>`

## Modifying Frontend Behavior

Use this rough routing when making changes:

- Layout/markup for an initial page surface: edit `templates/*.html`.
- Visual styling: edit the matching `static/*.css`.
- Browser state, controls, interactions, plotting, playback, and API calls: edit the matching `static/*.js`.
- Data shape or extra fields: edit the Python exporter that creates the JSON payload, then update the matching JS reader.
- Local app endpoint behavior: edit `sequence_builder_app.py`, `transition_workbench.py`, or shared helpers in
  `local_http.py`.

Avoid adding large HTML/CSS/JS strings back into Python. If a Python module needs a browser fragment, add a template or
static asset and load it through `frontend_assets.py`.

## Debug Checklist

For syntax and packaging checks:

```bash
python3 -m py_compile \
  src/djprojectexploration/frontend_assets.py \
  src/djprojectexploration/local_http.py \
  src/djprojectexploration/energy_sequence_builder.py \
  src/djprojectexploration/interactive_pacmap_knn_simplex.py \
  src/djprojectexploration/transition_workbench.py \
  src/djprojectexploration/sequence_builder_app.py
```

Generate and serve a static sequence-builder export:

```bash
uv run djprojectexploration-dj-sequence
python3 -m http.server 8788 --directory data/exports
```

Open:

```text
http://127.0.0.1:8788/dj_sequence_builder.html
```

Run the full served sequence builder when testing transition rendering:

```bash
uv run djprojectexploration-sequence-builder-app
```

Open:

```text
http://127.0.0.1:8770/
```

If a control does not appear, inspect the relevant config JSON tag first. For example, the sequence-builder render button
depends on `seq-config-json.app_mode`.
