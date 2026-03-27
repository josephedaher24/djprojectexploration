"""Compute and plot a full-song chromagram with Essentia."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from essentia.standard import (
    FrameGenerator,
    HPCP,
    Key,
    MonoLoader,
    OnsetRate,
    RhythmExtractor2013,
    SpectralPeaks,
    Spectrum,
    Windowing,
)

# Essentia's HPCP bins are A..G# by default.
PITCH_CLASS_LABELS = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


def compute_chromagram(audio: np.ndarray, sample_rate: int, frame_size: int, hop_size: int, chroma_bins: int) -> np.ndarray:
    windowing = Windowing(type="hann")
    spectrum = Spectrum(size=frame_size)
    spectral_peaks = SpectralPeaks(
        sampleRate=sample_rate,
        minFrequency=40,
        maxFrequency=5000,
        maxPeaks=80,
        magnitudeThreshold=1e-5,
        orderBy="magnitude",
    )
    hpcp = HPCP(
        size=chroma_bins,
        sampleRate=sample_rate,
        referenceFrequency=440,
        minFrequency=40,
        maxFrequency=5000,
        harmonics=8,
        nonLinear=False,
        weightType="cosine",
    )

    frames: list[np.ndarray] = []
    for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        spec = spectrum(windowing(frame))
        freqs, mags = spectral_peaks(spec)
        if len(freqs) == 0:
            frames.append(np.zeros(chroma_bins, dtype=np.float32))
            continue
        frames.append(np.asarray(hpcp(freqs, mags), dtype=np.float32))

    if not frames:
        raise ValueError("No frames were generated from the input audio.")

    # Shape: [chroma_bins, time_frames]
    return np.vstack(frames).T


def estimate_key(audio: np.ndarray, sample_rate: int, frame_size: int, hop_size: int) -> tuple[str, str, float]:
    """Estimate key/scale/strength using Essentia-style HPCP+Key configuration."""
    windowing = Windowing(type="blackmanharris62")
    spectrum = Spectrum(size=frame_size)
    spectral_peaks = SpectralPeaks(
        orderBy="magnitude",
        magnitudeThreshold=1e-5,
        minFrequency=20,
        maxFrequency=3500,
        maxPeaks=60,
    )
    hpcp_key = HPCP(
        size=36,
        referenceFrequency=440,
        sampleRate=sample_rate,
        bandPreset=False,
        minFrequency=20,
        maxFrequency=3500,
        weightType="cosine",
        nonLinear=False,
        windowSize=1.0,
    )
    key_detector = Key(
        profileType="edma",
        numHarmonics=4,
        pcpSize=36,
        slope=0.6,
        usePolyphony=True,
        useThreeChords=True,
    )

    hpcp_frames: list[np.ndarray] = []
    for frame in FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        spec = spectrum(windowing(frame))
        freqs, mags = spectral_peaks(spec)
        if len(freqs) == 0:
            continue
        hpcp_frames.append(np.asarray(hpcp_key(freqs, mags), dtype=np.float32))

    if not hpcp_frames:
        return "N", "unknown", 0.0

    # Aggregate track-level tonal profile before running key estimation.
    mean_hpcp = np.vstack(hpcp_frames).mean(axis=0).astype(np.float32)
    key, scale, strength, _relative_strength = key_detector(mean_hpcp)
    return str(key), str(scale), float(strength)


def detect_beats(audio: np.ndarray) -> tuple[float, np.ndarray]:
    bpm, beat_times, *_ = RhythmExtractor2013(method="multifeature")(audio)
    return float(bpm), np.asarray(beat_times, dtype=np.float32)


def pool_chromagram_by_beats(
    chromagram: np.ndarray,
    sample_rate: int,
    hop_size: int,
    beat_times: np.ndarray,
    chroma_duration: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Pool frame chroma by beat intervals. Returns (pooled_chroma, interval_edges)."""
    if beat_times.size == 0:
        return None

    num_bins, num_frames = chromagram.shape
    if num_frames == 0:
        return None

    frame_times = (np.arange(num_frames, dtype=np.float32) * hop_size) / sample_rate
    internal_beats = beat_times[(beat_times > 0.0) & (beat_times < chroma_duration)]
    interval_edges = np.concatenate(([0.0], internal_beats, [chroma_duration])).astype(np.float32)
    interval_edges = np.unique(interval_edges)
    if interval_edges.size < 2:
        return None

    pooled_frames: list[np.ndarray] = []
    for i in range(interval_edges.size - 1):
        start = float(interval_edges[i])
        end = float(interval_edges[i + 1])
        mask = (frame_times >= start) & (frame_times < end)
        if np.any(mask):
            pooled_frames.append(chromagram[:, mask].mean(axis=1))
        else:
            midpoint = (start + end) * 0.5
            nearest_idx = int(np.argmin(np.abs(frame_times - midpoint)))
            pooled_frames.append(chromagram[:, nearest_idx])

    pooled_chromagram = np.stack(pooled_frames, axis=1).astype(np.float32)
    return pooled_chromagram, interval_edges


def detect_first_onset(audio: np.ndarray) -> float | None:
    """Return first onset time in seconds, or None if no onset is detected."""
    _, onset_times = OnsetRate()(audio)
    onset_times = np.atleast_1d(np.asarray(onset_times, dtype=np.float32))
    if onset_times.size == 0:
        return None
    return float(onset_times[0])


def build_beat_grid_from_bpm(audio_duration: float, bpm: float, phase_anchor: float) -> np.ndarray:
    """Build beat timestamps from a fixed BPM and phase anchor."""
    beat_period = 60.0 / bpm
    # Shift anchor backward to the first beat on/after 0 s.
    first_beat = phase_anchor
    while first_beat - beat_period >= 0.0:
        first_beat -= beat_period

    # Guard against tiny negative float drift.
    first_beat = max(0.0, first_beat)
    beat_times = np.arange(first_beat, audio_duration + beat_period, beat_period, dtype=np.float32)
    return beat_times[beat_times <= audio_duration]


def prepare_waveform_for_plot(
    audio: np.ndarray, sample_rate: int, samples_per_second: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return high-resolution waveform data for plotting."""
    if samples_per_second <= 0:
        raise ValueError("samples_per_second must be a positive integer.")

    total_duration = audio.size / sample_rate
    target_points = max(2000, int(np.ceil(total_duration * samples_per_second)))

    if audio.size <= target_points:
        times = np.arange(audio.size, dtype=np.float32) / sample_rate
        values = audio.astype(np.float32)
        return times, values, values, "raw"

    bin_size = int(np.ceil(audio.size / target_points))
    padded_size = int(np.ceil(audio.size / bin_size) * bin_size)
    if padded_size > audio.size:
        padded_audio = np.pad(audio, (0, padded_size - audio.size), mode="edge")
    else:
        padded_audio = audio

    reshaped = padded_audio.reshape(-1, bin_size)
    minima = reshaped.min(axis=1).astype(np.float32)
    maxima = reshaped.max(axis=1).astype(np.float32)
    times = (np.arange(reshaped.shape[0], dtype=np.float32) * bin_size + (bin_size / 2)) / sample_rate
    return times, minima, maxima, "envelope"


def attach_zoom_controls(
    fig: plt.Figure,
    primary_axis: plt.Axes,
    axes: list[plt.Axes],
    x_bounds: tuple[float, float],
) -> None:
    """Add mouse-wheel zoom and keyboard reset for interactive inspection."""
    if not axes:
        return

    home_xlim = x_bounds
    home_ylims = {axis: axis.get_ylim() for axis in axes}
    zoom_step = 1.2
    pan_fraction = 0.2

    def clamp_xlim(left: float, right: float) -> tuple[float, float]:
        min_x, max_x = x_bounds
        if left < min_x:
            right += min_x - left
            left = min_x
        if right > max_x:
            left -= right - max_x
            right = max_x
        if (right - left) >= (max_x - min_x):
            return min_x, max_x
        return left, right

    def apply_zoom(scale: float, center_x: float) -> None:
        current_left, current_right = primary_axis.get_xlim()
        x_range = current_right - current_left
        if x_range <= 0:
            return
        rel_x = (center_x - current_left) / x_range
        new_x_range = x_range * scale
        new_left = center_x - new_x_range * rel_x
        new_right = center_x + new_x_range * (1 - rel_x)
        new_left, new_right = clamp_xlim(new_left, new_right)
        primary_axis.set_xlim(new_left, new_right)
        fig.canvas.draw_idle()

    def on_scroll(event) -> None:
        if event.inaxes not in axes or event.xdata is None:
            return

        scale = 1 / zoom_step if event.button == "up" else zoom_step
        apply_zoom(scale=scale, center_x=event.xdata)

    def on_key_press(event) -> None:
        if event.key == "r":
            primary_axis.set_xlim(home_xlim)
            for axis in axes:
                axis.set_ylim(home_ylims[axis])
            fig.canvas.draw_idle()
            return

        if event.key in {"<", ",", "left", "shift+<"}:
            current_left, current_right = primary_axis.get_xlim()
            shift = (current_right - current_left) * pan_fraction
            new_left, new_right = clamp_xlim(current_left - shift, current_right - shift)
            primary_axis.set_xlim(new_left, new_right)
            fig.canvas.draw_idle()
            return

        if event.key in {">", ".", "right", "shift+>"}:
            current_left, current_right = primary_axis.get_xlim()
            shift = (current_right - current_left) * pan_fraction
            new_left, new_right = clamp_xlim(current_left + shift, current_right + shift)
            primary_axis.set_xlim(new_left, new_right)
            fig.canvas.draw_idle()
            return

        if event.key in {"+", "=", "plus"}:
            current_left, current_right = primary_axis.get_xlim()
            center_x = (current_left + current_right) / 2
            apply_zoom(scale=1 / zoom_step, center_x=center_x)
            return

        if event.key in {"-", "_", "minus"}:
            current_left, current_right = primary_axis.get_xlim()
            center_x = (current_left + current_right) / 2
            apply_zoom(scale=zoom_step, center_x=center_x)

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key_press)


def plot_chromagram(
    chromagram: np.ndarray,
    audio: np.ndarray,
    sample_rate: int,
    hop_size: int,
    beat_times: np.ndarray,
    show_beat_pooled: bool,
    waveform_samples_per_second: int,
    title: str,
    key_label: str | None,
    output_file: Path,
    show: bool,
    interactive: bool,
) -> None:
    num_bins, num_frames = chromagram.shape
    audio_duration = audio.size / sample_rate
    chroma_duration = (num_frames * hop_size) / sample_rate
    x_max = max(audio_duration, chroma_duration)

    beat_pooled_data = None
    if show_beat_pooled:
        beat_pooled_data = pool_chromagram_by_beats(
            chromagram=chromagram,
            sample_rate=sample_rate,
            hop_size=hop_size,
            beat_times=beat_times,
            chroma_duration=chroma_duration,
        )

    has_beat_pooled_panel = beat_pooled_data is not None
    if has_beat_pooled_panel:
        fig, (chroma_ax, pooled_ax, waveform_ax) = plt.subplots(
            3,
            1,
            figsize=(14, 9.2),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 2.2, 1.3], "hspace": 0.08},
        )
        colorbar_axes = [chroma_ax, pooled_ax, waveform_ax]
    else:
        fig, (chroma_ax, waveform_ax) = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3.4, 1.3], "hspace": 0.08},
        )
        pooled_ax = None
        colorbar_axes = [chroma_ax, waveform_ax]

    image = chroma_ax.imshow(
        chromagram,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
        extent=[0, chroma_duration, 0, num_bins],
    )
    fig.colorbar(image, ax=colorbar_axes, label="Chroma Energy", pad=0.02)
    chroma_ax.set_ylabel("Pitch Class")
    if key_label is None:
        chroma_ax.set_title(title)
    else:
        chroma_ax.set_title(f"{title}  |  Key: {key_label}")
    chroma_ax.tick_params(axis="x", labelbottom=False)

    if num_bins % 12 == 0:
        bins_per_pitch_class = num_bins // 12
        tick_positions = [i * bins_per_pitch_class + (bins_per_pitch_class / 2) for i in range(12)]
        chroma_ax.set_yticks(tick_positions)
        chroma_ax.set_yticklabels(PITCH_CLASS_LABELS)
    else:
        chroma_ax.set_yticks(np.arange(0, num_bins, max(1, num_bins // 12)))

    if has_beat_pooled_panel and pooled_ax is not None:
        pooled_chromagram, interval_edges = beat_pooled_data
        y_edges = np.arange(0, num_bins + 1, dtype=np.float32)
        pooled_ax.pcolormesh(
            interval_edges,
            y_edges,
            pooled_chromagram,
            shading="auto",
            cmap="magma",
            vmin=float(np.min(chromagram)),
            vmax=float(np.max(chromagram)),
        )
        pooled_ax.set_ylabel("Pitch Class")
        pooled_ax.set_title("Beat-Pooled Chromagram")
        pooled_ax.tick_params(axis="x", labelbottom=False)
        if num_bins % 12 == 0:
            bins_per_pitch_class = num_bins // 12
            tick_positions = [i * bins_per_pitch_class + (bins_per_pitch_class / 2) for i in range(12)]
            pooled_ax.set_yticks(tick_positions)
            pooled_ax.set_yticklabels(PITCH_CLASS_LABELS)
        else:
            pooled_ax.set_yticks(np.arange(0, num_bins, max(1, num_bins // 12)))

    wave_times, wave_min, wave_max, waveform_mode = prepare_waveform_for_plot(
        audio=audio,
        sample_rate=sample_rate,
        samples_per_second=waveform_samples_per_second,
    )
    if waveform_mode == "raw":
        waveform_ax.plot(wave_times, wave_max, color="steelblue", linewidth=0.25)
    else:
        waveform_ax.fill_between(wave_times, wave_min, wave_max, color="steelblue", linewidth=0)

    waveform_peak = float(max(np.max(np.abs(wave_min)), np.max(np.abs(wave_max)), 1e-6))
    waveform_ax.set_ylim(-waveform_peak, waveform_peak)
    waveform_ax.set_ylabel("Amplitude")
    waveform_ax.set_xlabel("Time (s)")
    waveform_ax.axhline(0.0, color="black", linewidth=0.35, alpha=0.4)
    waveform_ax.margins(x=0)
    chroma_ax.set_xlim(0.0, x_max)

    if beat_times.size > 0:
        waveform_ax.vlines(beat_times, -waveform_peak, waveform_peak, color="crimson", linewidth=0.35, alpha=0.2)
        downbeat_times = beat_times[::4]
        if downbeat_times.size > 0:
            waveform_ax.vlines(
                downbeat_times,
                -waveform_peak,
                waveform_peak,
                color="red",
                linewidth=0.6,
                alpha=0.35,
            )

    # Add a second bottom axis for beat labels beneath the time axis.
    beat_axis = waveform_ax.secondary_xaxis("bottom", functions=(lambda x: x, lambda x: x))
    beat_axis.spines["bottom"].set_position(("outward", 34))
    beat_axis.set_xlabel("Beat")
    if beat_times.size > 0:
        step = 32
        tick_positions = beat_times[::step]
        tick_labels = [str(i) for i in range(0, beat_times.size, step)]
        beat_axis.set_xticks(tick_positions)
        beat_axis.set_xticklabels(tick_labels)

    if interactive:
        controlled_axes = [chroma_ax, waveform_ax] if pooled_ax is None else [chroma_ax, pooled_ax, waveform_ax]
        attach_zoom_controls(
            fig=fig,
            primary_axis=chroma_ax,
            axes=controlled_axes,
            x_bounds=(0.0, x_max),
        )

    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200)
    if show:
        if interactive:
            print("Interactive controls: mouse wheel or +/- to zoom, '<'/'>' to pan, 'r' to reset.")
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and plot a full chromagram for an audio file.")
    parser.add_argument("audio_file", type=Path, help="Path to input audio file (mp3/wav/etc.).")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Path to save the chromagram plot PNG. Defaults to <audio_stem>_chromagram.png.",
    )
    parser.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate for analysis.")
    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Optional manual BPM. If set, beat grid is generated from BPM and first onset phase anchor.",
    )
    parser.add_argument(
        "--onset-time-ms",
        type=float,
        default=None,
        help="Optional manual onset time in milliseconds used as phase anchor (requires --bpm).",
    )
    parser.add_argument("--frame-size", type=int, default=4096, help="Analysis frame size.")
    parser.add_argument("--hop-size", type=int, default=1024, help="Hop size between frames.")
    parser.add_argument(
        "--chroma-bins",
        type=int,
        default=12,
        help="Number of chroma bins (commonly 12 or 36).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving (optional).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable zoom controls in the displayed plot window (requires --show or will auto-enable it).",
    )
    parser.add_argument(
        "--waveform-samples-per-second",
        type=int,
        default=800,
        help="Waveform plotting resolution in samples per second of audio duration.",
    )
    parser.add_argument(
        "--show-beat-pooled",
        action="store_true",
        help="Show an additional beat-pooled chromagram panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_file = args.audio_file.resolve()
    if not audio_file.exists():
        raise SystemExit(f"Audio file not found: {audio_file}")
    if args.chroma_bins <= 0:
        raise SystemExit("--chroma-bins must be a positive integer.")
    if args.bpm is not None and args.bpm <= 0:
        raise SystemExit("--bpm must be a positive number.")
    if args.onset_time_ms is not None and args.onset_time_ms < 0:
        raise SystemExit("--onset-time-ms must be >= 0.")
    if args.onset_time_ms is not None and args.bpm is None:
        raise SystemExit("--onset-time-ms requires --bpm.")
    if args.waveform_samples_per_second <= 0:
        raise SystemExit("--waveform-samples-per-second must be a positive integer.")
    if args.interactive and not args.show:
        args.show = True

    output_file = (
        args.output_file.resolve()
        if args.output_file is not None
        else audio_file.with_name(f"{audio_file.stem}_chromagram.png")
    )

    audio = MonoLoader(filename=str(audio_file), sampleRate=args.sample_rate, resampleQuality=4)()
    chromagram = compute_chromagram(
        audio=audio,
        sample_rate=args.sample_rate,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        chroma_bins=args.chroma_bins,
    )
    if args.bpm is not None:
        audio_duration = audio.size / args.sample_rate
        if args.onset_time_ms is not None:
            onset_anchor = float(args.onset_time_ms) / 1000.0
            onset_anchor = min(max(0.0, onset_anchor), audio_duration)
            beat_source = "manual_bpm_manual_onset_anchor"
        else:
            onset_anchor = detect_first_onset(audio)
            if onset_anchor is None:
                onset_anchor = 0.0
            beat_source = "manual_bpm_first_onset_anchor"
        bpm = float(args.bpm)
        beat_times = build_beat_grid_from_bpm(audio_duration=audio_duration, bpm=bpm, phase_anchor=onset_anchor)
    else:
        bpm, beat_times = detect_beats(audio)
        onset_anchor = None
        beat_source = "detected_rhythmextractor2013_multifeature"

    detected_key, detected_scale, key_strength = estimate_key(
        audio=audio,
        sample_rate=args.sample_rate,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
    )

    plot_chromagram(
        chromagram=chromagram,
        audio=audio,
        sample_rate=args.sample_rate,
        hop_size=args.hop_size,
        beat_times=beat_times,
        show_beat_pooled=args.show_beat_pooled,
        waveform_samples_per_second=args.waveform_samples_per_second,
        title=f"Chromagram: {audio_file.name}",
        key_label=f"{detected_key} {detected_scale} ({key_strength:.2f})",
        output_file=output_file,
        show=args.show,
        interactive=args.interactive,
    )

    print(f"Audio file: {audio_file}")
    print(f"Estimated key: {detected_key} {detected_scale} (strength={key_strength:.3f})")
    print(f"Detected BPM: {bpm:.2f}")
    print(f"Beat source: {beat_source}")
    if onset_anchor is not None:
        print(f"Phase anchor (first onset): {onset_anchor:.3f} s")
    print(f"Detected beats: {beat_times.size}")
    print(f"Chromagram shape: {chromagram.shape} [bins x frames]")
    print(f"Saved chromagram plot: {output_file}")


if __name__ == "__main__":
    main()
