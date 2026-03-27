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

## MAEST embeddings (Essentia)

```bash
uv run djprojectexploration-maest \
  --model-file models/discogs-maest-30s-pw-519l-2.pb \
  --output-file music/maest_embedding_discogs-maest-30s-pw.json
```

Notes:
- `--audio-file` is optional; if omitted, the first `.mp3` in `music/` is used.
- If you do not have MAEST support in your Essentia install, use a TensorFlow-enabled build such as `essentia-tensorflow`.

## Chromagram plotting

```bash
uv run djprojectexploration-chromagram music/genix-giantsteps.mp3
```

Zoomable view:

```bash
uv run djprojectexploration-chromagram music/genix-giantsteps.mp3 --interactive
```

Show beat-pooled chromagram panel:

```bash
uv run djprojectexploration-chromagram music/genix-giantsteps.mp3 --show-beat-pooled
```

Use a manual BPM with first-onset phase anchoring:

```bash
uv run djprojectexploration-chromagram music/genix-giantsteps.mp3 --bpm 128
```

Override phase anchor manually (milliseconds):

```bash
uv run djprojectexploration-chromagram music/genix-giantsteps.mp3 --bpm 128 --onset-time-ms 3870
```

Controls in interactive mode:
- Mouse wheel: zoom in/out around cursor
- `+` / `-`: zoom in/out
- `<` / `>`: pan left/right
- `r`: reset zoom

## Quick import check

```bash
.venv/bin/python -c "import essentia, librosa, numpy, scipy, soundfile, matplotlib; print('imports ok')"
```
