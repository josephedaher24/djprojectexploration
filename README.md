# djprojectexploration

Minimal Python project scaffold using `uv`, with audio and signal-processing libraries installed.

## Getting started

```bash
uv sync
uv run djprojectexploration
```

## Audio stack

- `numpy` and `scipy` for core numerical and DSP work
- `librosa` for higher-level audio analysis
- `essentia` for music/audio feature extraction
- `soundfile` for reading and writing audio files
- `matplotlib` for waveform and spectrogram visualization

## Development

```bash
uv add <package>
uv run python -m djprojectexploration
```

## Quick import check

```bash
.venv/bin/python -c "import essentia, librosa, numpy, scipy, soundfile, matplotlib; print('imports ok')"
```
