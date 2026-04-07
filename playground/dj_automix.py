"""
DJ Automix Engine — Professional Transition Framework

A full-featured DJ transition engine that can describe, configure, and execute
complex song-to-song transitions with fine-grained control over every parameter.

Key capabilities:
  1. BPM Sync — 4 modes: A→B, B→A, Meet-in-Middle, None
  2. Key Sync — Auto-detect keys via Essentia, pitch-shift to match
  3. Crossfade Curves — Linear, Equal-Power, S-Curve, Exponential
  4. DJ EQ Profiles — Bass Swap, Full EQ Fade, Filter Sweep
  5. Transition FX — Reverb Tail, Echo Out, Backspin
  6. Multi-Track Chaining — Build a full DJ set from N tracks
  7. Presets — smooth_blend, energy_drop, bass_swap, echo_out, classic_cut

────────────────────────────────────────────────────────────────────

Example usage (CLI):

  # Quick 12-bar transition (4 bars overlap)
  uv run python playground/dj_automix.py \ 
    --track-a "playground/Above & Beyond - Far From In Love (Original Mix).mp3" \
    --a-start-bar 4 --a-out-bar 12 \
    --track-b "playground/Bloodstream (Cubicore Extended Remix).mp3" \
    --b-in-bar 4 --b-end-bar 12 \
    --overlap 4 \
    --crossfade equal_power \
    --eq-profile bass_swap \
    --output playground/output/quick_12bar.wav


  # Using a preset
  uv run python playground/dj_automix.py \
      --track-a "playground/Above & Beyond - Far From In Love (Original Mix).mp3" \
      --track-b "playground/Bloodstream (Cubicore Extended Remix).mp3" \
      --preset smooth_blend \
      --output playground/output/preset_mix.wav

  # Full manual control
  uv run python playground/dj_automix.py \
      --track-a "playground/Above & Beyond - Far From In Love (Original Mix).mp3" --a-out-bar 64 \
      --track-b "playground/Bloodstream (Cubicore Extended Remix).mp3" --b-in-bar 16 \
      --overlap 16 \
      --bpm-sync b_to_a \
      --key-sync b_to_a \
      --crossfade equal_power \
      --eq-profile bass_swap \
      --fx reverb_tail \
      --output playground/output/manual_mix.wav

────────────────────────────────────────────────────────────────────

Example usage (Python API):

  from playground.dj_automix import AutomixEngine, TransitionConfig, PRESETS
  from playground.dj_automix import BpmSyncMode, KeySyncMode, CrossfadeCurve, EQProfile, TransitionFX

  # ── Preset-based (simplest) ──
  engine = AutomixEngine()
  engine.add_track("track_a.mp3")
  engine.add_track("track_b.mp3")
  result = engine.mix_transition(0, 1, PRESETS["smooth_blend"])
  engine.export(result, "mix.wav")

  # ── Fully custom ──
  config = TransitionConfig(
      a_out_bar=64,              # Start mixing out of Track A at bar 64
      b_in_bar=16,               # Start mixing into Track B at bar 16
      overlap_bars=16,           # 16-bar crossfade overlap
      bpm_sync=BpmSyncMode.B_TO_A,         # Stretch B to A's tempo
      target_bpm=None,                       # (or set absolute e.g. 128.0)
      key_sync=KeySyncMode.B_TO_A,          # Pitch-shift B to A's key
      pitch_shift_semitones=None,            # (or manual override e.g. -2.0)
      crossfade_curve=CrossfadeCurve.EQUAL_POWER,
      eq_profile=EQProfile.BASS_SWAP,
      transition_fx=[TransitionFX.REVERB_TAIL],
  )
  result = engine.mix_transition(0, 1, config)
  engine.export(result, "custom_mix.wav")

  # ── Multi-track chaining ──
  engine = AutomixEngine()
  engine.add_track("track_a.mp3")
  engine.add_track("track_b.mp3")
  engine.add_track("track_c.mp3")
  transitions = [
      PRESETS["smooth_blend"],
      TransitionConfig(a_out_bar=48, b_in_bar=8, overlap_bars=8,
                       bpm_sync=BpmSyncMode.MEET_IN_MIDDLE,
                       crossfade_curve=CrossfadeCurve.S_CURVE,
                       eq_profile=EQProfile.FILTER_SWEEP,
                       transition_fx=[TransitionFX.ECHO_OUT]),
  ]
  result = engine.chain_mix(transitions)
  engine.export(result, "dj_set.wav")
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, fftconvolve


# ═══════════════════════════════════════════════════════════════════
#  Enums — Strategy Types
# ═══════════════════════════════════════════════════════════════════

class BpmSyncMode(Enum):
    """How to match tempos between two tracks."""
    A_TO_B = "a_to_b"          # Stretch A to match B's BPM
    B_TO_A = "b_to_a"          # Stretch B to match A's BPM  (classic DJ default)
    MEET_IN_MIDDLE = "middle"  # Both stretch to average BPM
    NONE = "none"              # No sync — raw BPMs


class KeySyncMode(Enum):
    """How to match musical keys between two tracks."""
    A_TO_B = "a_to_b"         # Pitch-shift A to match B's key
    B_TO_A = "b_to_a"         # Pitch-shift B to match A's key
    NONE = "none"             # No key matching


class CrossfadeCurve(Enum):
    """Shape of the crossfade volume envelope."""
    LINEAR = "linear"                # Simple straight line
    EQUAL_POWER = "equal_power"      # sqrt-based, constant perceived loudness
    S_CURVE = "s_curve"              # Smooth sigmoid — holds levels longer, fast swap
    EXPONENTIAL = "exponential"      # Aggressive fade — quick drop-off / build-up


class EQProfile(Enum):
    """DJ EQ behaviour during the overlap region."""
    NONE = "none"                    # No EQ processing
    BASS_SWAP = "bass_swap"          # HPF on A tail / clean B head (swap the bass)
    FULL_EQ_FADE = "full_eq"         # Gradual 3-band EQ crossover (low, mid, high)
    FILTER_SWEEP = "filter_sweep"    # LPF sweep ↓ on A, HPF sweep ↑ on B


class TransitionFX(Enum):
    """Optional effects applied during / after the transition."""
    NONE = "none"
    REVERB_TAIL = "reverb_tail"      # Reverb wash on outgoing Track A tail
    ECHO_OUT = "echo_out"            # Echo/delay feedback tail on Track A
    BACKSPIN = "backspin"            # Vinyl backspin on outgoing Track A tail


# ═══════════════════════════════════════════════════════════════════
#  TransitionConfig — Declarative Description of a Transition
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TransitionConfig:
    """
    Complete specification of a DJ transition from Track A → Track B.

    Every parameter of the transition is configurable:
    slice points, overlap length, BPM/Key sync direction,
    crossfade shape, EQ profile, and post-effects.
    """

    # ── Slice Points ──
    a_start_bar: float = 1           # First bar of Track A to include (default: 1)
    a_out_bar: float = 64            # Bar in Track A where transition begins
    b_in_bar: float = 1              # Bar in Track B where it enters the mix
    b_end_bar: Optional[float] = None  # Bar in Track B to stop (None = play to end)

    # ── Overlap Region ──
    overlap_bars: float = 16         # Duration of the crossfade in bars

    # ── BPM Sync ──
    bpm_sync: BpmSyncMode = BpmSyncMode.B_TO_A
    target_bpm: Optional[float] = None  # Absolute override (ignores sync mode)

    # ── Key Sync ──
    key_sync: KeySyncMode = KeySyncMode.NONE
    pitch_shift_semitones: Optional[float] = None  # Manual semitone override

    # ── Crossfade Curve ──
    crossfade_curve: CrossfadeCurve = CrossfadeCurve.EQUAL_POWER

    # ── DJ EQ ──
    eq_profile: EQProfile = EQProfile.BASS_SWAP

    # ── Transition Effects ──
    transition_fx: list[TransitionFX] = field(default_factory=list)

    def __repr__(self) -> str:
        fx_str = ", ".join(f.value for f in self.transition_fx) if self.transition_fx else "none"
        return (
            f"TransitionConfig(\n"
            f"  slice A:   bar{self.a_start_bar}~bar{self.a_out_bar} body + {self.overlap_bars} bars tail\n"
            f"  slice B:   bar{self.b_in_bar} head + body"
            + (f" → end@bar{self.b_end_bar}" if self.b_end_bar is not None else "") + "\n"
            f"  overlap:   {self.overlap_bars} bars\n"
            f"  bpm_sync:  {self.bpm_sync.value}"
            + (f" (target={self.target_bpm})" if self.target_bpm else "") + "\n"
            f"  key_sync:  {self.key_sync.value}"
            + (f" (shift={self.pitch_shift_semitones:+.1f}st)" if self.pitch_shift_semitones is not None else "") + "\n"
            f"  crossfade: {self.crossfade_curve.value}\n"
            f"  eq:        {self.eq_profile.value}\n"
            f"  fx:        [{fx_str}]\n"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════
#  Presets — Ready-to-use Transition Configurations
# ═══════════════════════════════════════════════════════════════════

PRESETS: dict[str, TransitionConfig] = {
    "smooth_blend": TransitionConfig(
        a_out_bar=64, b_in_bar=1, overlap_bars=16,
        bpm_sync=BpmSyncMode.B_TO_A, key_sync=KeySyncMode.NONE,
        crossfade_curve=CrossfadeCurve.EQUAL_POWER,
        eq_profile=EQProfile.BASS_SWAP, transition_fx=[],
    ),
    "energy_drop": TransitionConfig(
        a_out_bar=64, b_in_bar=1, overlap_bars=4,
        bpm_sync=BpmSyncMode.B_TO_A, key_sync=KeySyncMode.NONE,
        crossfade_curve=CrossfadeCurve.EXPONENTIAL,
        eq_profile=EQProfile.FILTER_SWEEP,
        transition_fx=[TransitionFX.REVERB_TAIL],
    ),
    "bass_swap": TransitionConfig(
        a_out_bar=64, b_in_bar=1, overlap_bars=8,
        bpm_sync=BpmSyncMode.B_TO_A, key_sync=KeySyncMode.NONE,
        crossfade_curve=CrossfadeCurve.EQUAL_POWER,
        eq_profile=EQProfile.BASS_SWAP, transition_fx=[],
    ),
    "echo_out": TransitionConfig(
        a_out_bar=64, b_in_bar=1, overlap_bars=8,
        bpm_sync=BpmSyncMode.B_TO_A, key_sync=KeySyncMode.NONE,
        crossfade_curve=CrossfadeCurve.S_CURVE,
        eq_profile=EQProfile.NONE,
        transition_fx=[TransitionFX.ECHO_OUT],
    ),
    "classic_cut": TransitionConfig(
        a_out_bar=64, b_in_bar=1, overlap_bars=1,
        bpm_sync=BpmSyncMode.B_TO_A, key_sync=KeySyncMode.NONE,
        crossfade_curve=CrossfadeCurve.LINEAR,
        eq_profile=EQProfile.NONE, transition_fx=[],
    ),
}


# ═══════════════════════════════════════════════════════════════════
#  TrackInfo — Internal per-track data holder
# ═══════════════════════════════════════════════════════════════════

@dataclass
class _TrackInfo:
    """Internal representation of a loaded track with its analysis data."""
    path: Path
    y: np.ndarray              # Audio data, shape (channels, samples)
    sr: int                    # Sample rate
    bpm: float                 # Detected BPM
    key: str                   # Detected key (e.g. "C")
    scale: str                 # Detected scale (e.g. "minor")
    downbeat_samples: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    # Sample positions of every downbeat (bar boundary), derived from detected beat grid

    @property
    def key_label(self) -> str:
        return f"{self.key} {self.scale}"


# ═══════════════════════════════════════════════════════════════════
#  Key Utilities — Semitone Distance Calculation
# ═══════════════════════════════════════════════════════════════════

# Chromatic pitch classes in order — used to compute semitone delta
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Enharmonic aliases
_ENHARMONIC = {"Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B"}


def _normalize_pitch(name: str) -> str:
    """Normalize enharmonic pitch names to sharp notation."""
    return _ENHARMONIC.get(name, name)


def _semitone_distance(from_key: str, from_scale: str, to_key: str, to_scale: str) -> float:
    """
    Calculate the semitone shift needed to move from (from_key, from_scale)
    to (to_key, to_scale). Positive = shift up, negative = shift down.
    Picks the shortest path around the circle (max ±6 semitones).
    """
    fk = _normalize_pitch(from_key)
    tk = _normalize_pitch(to_key)

    if fk not in _PITCH_CLASSES or tk not in _PITCH_CLASSES:
        print(f"    ⚠ Could not resolve key distance ({from_key}→{to_key}), skipping key sync.")
        return 0.0

    fi = _PITCH_CLASSES.index(fk)
    ti = _PITCH_CLASSES.index(tk)

    # If one is major and the other minor, shift minor to its relative major
    # so that the comparison is apples-to-apples.
    if from_scale == "minor":
        fi = (fi + 3) % 12  # relative major
    if to_scale == "minor":
        ti = (ti + 3) % 12

    delta = (ti - fi) % 12
    if delta > 6:
        delta -= 12  # shortest path
    return float(delta)


# ═══════════════════════════════════════════════════════════════════
#  AutomixEngine — The Core
# ═══════════════════════════════════════════════════════════════════

class AutomixEngine:
    """
    Professional DJ Automix Engine.

    Workflow:
      1. Add tracks with ``add_track()``
      2. Execute a single transition with ``mix_transition(idx_a, idx_b, config)``
         — or chain multiple with ``chain_mix(configs)``
      3. Export with ``export(audio, path)``
    """

    def __init__(self):
        self.tracks: list[_TrackInfo] = []

    # ── Track Loading ────────────────────────────────────────────

    def add_track(self, audio_path: str | Path) -> int:
        """
        Load a track, auto-detect its BPM and Key, and store it.
        Returns the track index.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        print(f"\n{'═'*60}")
        print(f"[+] Loading Track #{len(self.tracks)}: {path.name}")
        print(f"{'═'*60}")

        y, sr = librosa.load(path, sr=None, mono=False)
        if y.ndim == 1:
            y = y.reshape(1, -1)

        # Trim leading/trailing silence
        y, _ = librosa.effects.trim(y, top_db=40)

        # ── BPM & Beat Grid Estimation ──
        y_mono = librosa.to_mono(y)
        tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
        bpm = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
        beat_samples = librosa.frames_to_samples(beat_frames)

        # Downbeats = every 4th beat (bar boundaries)
        downbeat_samples = beat_samples[::4].astype(int)

        # ── Key Estimation (Essentia) ──
        key, scale = "Unknown", "Unknown"
        try:
            from essentia.standard import MonoLoader, KeyExtractor
            audio_ess = MonoLoader(filename=str(path))()
            key, scale, _strength = KeyExtractor()(audio_ess)
        except Exception as e:
            print(f"    ⚠ Key detection failed ({e}), defaulting to Unknown.")

        info = _TrackInfo(path=path, y=y, sr=sr, bpm=bpm, key=key, scale=scale,
                          downbeat_samples=downbeat_samples)
        self.tracks.append(info)

        duration = y.shape[1] / sr
        print(f"    BPM:      {bpm:.2f}")
        print(f"    Key:      {info.key_label}")
        print(f"    Bars:     {len(downbeat_samples)} detected downbeats")
        print(f"    Duration: {duration:.1f}s | Channels: {y.shape[0]} | SR: {sr}Hz")

        return len(self.tracks) - 1

    # ── Internal Helpers ─────────────────────────────────────────

    @staticmethod
    def _bars_to_samples(bars: float, bpm: float, sr: int) -> int:
        """Convert a bar count to a sample count (fallback when beat grid unavailable)."""
        beats = bars * 4.0
        duration_sec = (60.0 / bpm) * beats
        return int(np.round(duration_sec * sr))

    @staticmethod
    def _get_downbeat_sample(
        downbeats: np.ndarray, bar_number: int, bpm: float, sr: int
    ) -> int:
        """
        Get the sample position of a bar number from the detected beat grid.
        Falls back to BPM math if the bar index is out of range.

        Args:
            downbeats: Array of downbeat sample positions (0-indexed)
            bar_number: 1-indexed bar number
            bpm: BPM for fallback math
            sr: Sample rate for fallback math
        """
        idx = bar_number - 1  # convert to 0-indexed
        if 0 <= idx < len(downbeats):
            return int(downbeats[idx])
        else:
            # Fallback: extrapolate from the last known downbeat or use math
            if len(downbeats) > 0 and idx >= len(downbeats):
                # Extrapolate beyond detected grid
                bar_duration = AutomixEngine._bars_to_samples(1, bpm, sr)
                extra_bars = idx - (len(downbeats) - 1)
                return int(downbeats[-1]) + extra_bars * bar_duration
            else:
                # No grid at all — pure math
                return AutomixEngine._bars_to_samples(bar_number - 1, bpm, sr)

    @staticmethod
    def _time_stretch_audio(y: np.ndarray, rate: float) -> np.ndarray:
        """Time-stretch multi-channel audio by rate (>1 = faster)."""
        stretched = []
        for ch in range(y.shape[0]):
            stretched.append(librosa.effects.time_stretch(y[ch], rate=rate))
        min_len = min(len(ch) for ch in stretched)
        return np.array([ch[:min_len] for ch in stretched])

    @staticmethod
    def _pitch_shift_audio(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
        """Pitch-shift multi-channel audio by n_steps semitones."""
        shifted = np.zeros_like(y)
        for ch in range(y.shape[0]):
            shifted[ch] = librosa.effects.pitch_shift(y[ch], sr=sr, n_steps=n_steps)
        return shifted

    # ── BPM Sync ─────────────────────────────────────────────────

    def _sync_bpm(
        self, y_a: np.ndarray, bpm_a: float,
        y_b: np.ndarray, bpm_b: float,
        sr: int, config: TransitionConfig
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """
        Apply BPM synchronisation according to the config.
        Returns (y_a, y_b, working_bpm, rate_a, rate_b).
        rate_a/rate_b are the time-stretch rates applied (1.0 = no change).
        """
        rate_a = 1.0
        rate_b = 1.0

        if config.target_bpm is not None:
            target = config.target_bpm
            print(f"[♫] BPM Sync: Forcing both tracks to {target:.2f} BPM")
            if abs(bpm_a - target) > 0.1:
                rate_a = target / bpm_a
                print(f"    Stretching Track A by {rate_a:.4f}x ({bpm_a:.2f} → {target:.2f})")
                y_a = self._time_stretch_audio(y_a, rate_a)
            if abs(bpm_b - target) > 0.1:
                rate_b = target / bpm_b
                print(f"    Stretching Track B by {rate_b:.4f}x ({bpm_b:.2f} → {target:.2f})")
                y_b = self._time_stretch_audio(y_b, rate_b)
            return y_a, y_b, target, rate_a, rate_b

        mode = config.bpm_sync

        if mode == BpmSyncMode.NONE:
            print(f"[♫] BPM Sync: NONE (A={bpm_a:.2f}, B={bpm_b:.2f})")
            return y_a, y_b, bpm_a, 1.0, 1.0

        if mode == BpmSyncMode.B_TO_A:
            if abs(bpm_a - bpm_b) < 0.1:
                print(f"[♫] BPM Sync: Already matched ({bpm_a:.2f})")
                return y_a, y_b, bpm_a, 1.0, 1.0
            rate_b = bpm_a / bpm_b
            print(f"[♫] BPM Sync: B→A — Stretching Track B by {rate_b:.4f}x ({bpm_b:.2f} → {bpm_a:.2f})")
            y_b = self._time_stretch_audio(y_b, rate_b)
            return y_a, y_b, bpm_a, 1.0, rate_b

        if mode == BpmSyncMode.A_TO_B:
            if abs(bpm_a - bpm_b) < 0.1:
                print(f"[♫] BPM Sync: Already matched ({bpm_b:.2f})")
                return y_a, y_b, bpm_b, 1.0, 1.0
            rate_a = bpm_b / bpm_a
            print(f"[♫] BPM Sync: A→B — Stretching Track A by {rate_a:.4f}x ({bpm_a:.2f} → {bpm_b:.2f})")
            y_a = self._time_stretch_audio(y_a, rate_a)
            return y_a, y_b, bpm_b, rate_a, 1.0

        if mode == BpmSyncMode.MEET_IN_MIDDLE:
            target = (bpm_a + bpm_b) / 2.0
            print(f"[♫] BPM Sync: Meet-in-Middle — Target {target:.2f} BPM")
            if abs(bpm_a - target) > 0.1:
                rate_a = target / bpm_a
                print(f"    Stretching Track A by {rate_a:.4f}x ({bpm_a:.2f} → {target:.2f})")
                y_a = self._time_stretch_audio(y_a, rate_a)
            if abs(bpm_b - target) > 0.1:
                rate_b = target / bpm_b
                print(f"    Stretching Track B by {rate_b:.4f}x ({bpm_b:.2f} → {target:.2f})")
                y_b = self._time_stretch_audio(y_b, rate_b)
            return y_a, y_b, target, rate_a, rate_b

        return y_a, y_b, bpm_a, 1.0, 1.0

    # ── Key Sync ─────────────────────────────────────────────────

    def _sync_key(
        self, y_a: np.ndarray, key_a: str, scale_a: str,
        y_b: np.ndarray, key_b: str, scale_b: str,
        sr: int, config: TransitionConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply Key synchronisation according to the config.
        Returns (y_a, y_b).
        """
        mode = config.key_sync

        if mode == KeySyncMode.NONE and config.pitch_shift_semitones is None:
            print(f"[♫] Key Sync: NONE (A={key_a} {scale_a}, B={key_b} {scale_b})")
            return y_a, y_b

        # Determine the semitone shift
        if config.pitch_shift_semitones is not None:
            shift = config.pitch_shift_semitones
            # Manual override: always apply to B unless mode says A_TO_B
            if mode == KeySyncMode.A_TO_B:
                print(f"[♫] Key Sync: Manual pitch shift on Track A: {shift:+.1f} semitones")
                y_a = self._pitch_shift_audio(y_a, sr, shift)
            else:
                print(f"[♫] Key Sync: Manual pitch shift on Track B: {shift:+.1f} semitones")
                y_b = self._pitch_shift_audio(y_b, sr, shift)
            return y_a, y_b

        # Auto-detect delta
        if "Unknown" in (key_a, key_b):
            print(f"[♫] Key Sync: Cannot auto-detect (A={key_a}, B={key_b}), skipping.")
            return y_a, y_b

        if mode == KeySyncMode.B_TO_A:
            shift = _semitone_distance(key_b, scale_b, key_a, scale_a)
            if abs(shift) < 0.01:
                print(f"[♫] Key Sync: Keys already compatible ({key_a} {scale_a})")
                return y_a, y_b
            print(f"[♫] Key Sync: B→A — Shifting Track B by {shift:+.0f} semitones "
                  f"({key_b} {scale_b} → {key_a} {scale_a})")
            y_b = self._pitch_shift_audio(y_b, sr, shift)

        elif mode == KeySyncMode.A_TO_B:
            shift = _semitone_distance(key_a, scale_a, key_b, scale_b)
            if abs(shift) < 0.01:
                print(f"[♫] Key Sync: Keys already compatible ({key_b} {scale_b})")
                return y_a, y_b
            print(f"[♫] Key Sync: A→B — Shifting Track A by {shift:+.0f} semitones "
                  f"({key_a} {scale_a} → {key_b} {scale_b})")
            y_a = self._pitch_shift_audio(y_a, sr, shift)

        return y_a, y_b

    # ── Crossfade Curve Generation ───────────────────────────────

    @staticmethod
    def _generate_crossfade(curve: CrossfadeCurve, length: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate (fade_out, fade_in) arrays of the specified shape.
        Each is a 1-D numpy array of length ``length``.
        """
        t = np.linspace(0.0, 1.0, length)  # 0→1 over the overlap

        if curve == CrossfadeCurve.LINEAR:
            fade_in = t
            fade_out = 1.0 - t

        elif curve == CrossfadeCurve.EQUAL_POWER:
            # Constant-power crossfade: sqrt-based
            fade_in = np.sqrt(t)
            fade_out = np.sqrt(1.0 - t)

        elif curve == CrossfadeCurve.S_CURVE:
            # Smooth sigmoid — holds levels longer, fast transition in the middle
            fade_in = 0.5 * (1.0 + np.tanh(6.0 * (t - 0.5)))
            fade_out = 1.0 - fade_in

        elif curve == CrossfadeCurve.EXPONENTIAL:
            # Aggressive: fast attack / quick decay
            fade_in = np.power(t, 0.3)         # fast rise
            fade_out = np.power(1.0 - t, 0.3)  # fast drop

        else:
            fade_in = t
            fade_out = 1.0 - t

        return fade_out, fade_in

    # ── EQ Profiles ──────────────────────────────────────────────

    def _apply_eq_profile(
        self, a_tail: np.ndarray, b_head: np.ndarray,
        sr: int, profile: EQProfile
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply DJ EQ processing to the overlap regions.
        Returns (processed_a_tail, processed_b_head).
        """
        if profile == EQProfile.NONE:
            return a_tail, b_head

        nyq = 0.5 * sr
        n_samples = a_tail.shape[1]

        if profile == EQProfile.BASS_SWAP:
            # High-pass the outgoing track A (remove bass → let B's bass come through)
            cutoff = 400
            sos_hp = butter(4, cutoff / nyq, btype='high', output='sos')
            print(f"[EQ] Bass Swap: HPF@{cutoff}Hz on Track A tail")
            a_tail = sosfilt(sos_hp, a_tail)

        elif profile == EQProfile.FULL_EQ_FADE:
            # Three-band EQ crossover: gradually move each band from A → B
            # Low band (< 300Hz), Mid band (300-3000Hz), High band (> 3000Hz)
            lo_cut, hi_cut = 300, 3000
            print(f"[EQ] Full EQ Fade: 3-band crossover (Lo<{lo_cut}Hz, Mid, Hi>{hi_cut}Hz)")

            sos_lo = butter(4, lo_cut / nyq, btype='low', output='sos')
            sos_hi = butter(4, hi_cut / nyq, btype='high', output='sos')
            sos_bp = butter(4, [lo_cut / nyq, hi_cut / nyq], btype='band', output='sos')

            # Fade factors per band (bass goes first, mids in middle, highs last)
            t = np.linspace(0.0, 1.0, n_samples)
            # Bass: A fades out early (first half), B fades in early
            bass_fade_out = np.clip(1.0 - 2.0 * t, 0, 1)
            bass_fade_in = np.clip(2.0 * t, 0, 1)
            # Mids: centered crossfade
            mid_fade_out = 1.0 - t
            mid_fade_in = t
            # Highs: A fades out late, B fades in late
            hi_fade_out = np.clip(1.0 - 2.0 * (t - 0.5), 0, 1)
            hi_fade_in = np.clip(2.0 * (t - 0.5), 0, 1)

            a_lo = sosfilt(sos_lo, a_tail) * bass_fade_out
            a_mid = sosfilt(sos_bp, a_tail) * mid_fade_out
            a_hi = sosfilt(sos_hi, a_tail) * hi_fade_out
            a_tail = a_lo + a_mid + a_hi

            b_lo = sosfilt(sos_lo, b_head) * bass_fade_in
            b_mid = sosfilt(sos_bp, b_head) * mid_fade_in
            b_hi = sosfilt(sos_hi, b_head) * hi_fade_in
            b_head = b_lo + b_mid + b_hi

        elif profile == EQProfile.FILTER_SWEEP:
            # Smooth filter sweep: LPF sweeps down on A, HPF sweeps up on B
            print("[EQ] Filter Sweep: LPF↓ on A, HPF↑ on B")
            n_steps = 32  # number of filter steps across the overlap
            step_len = n_samples // n_steps

            a_processed = np.zeros_like(a_tail)
            b_processed = np.zeros_like(b_head)

            for i in range(n_steps):
                start = i * step_len
                end = min(start + step_len, n_samples)

                # A: LPF cutoff sweeps from 10000 Hz → 200 Hz
                progress = i / max(n_steps - 1, 1)
                a_cutoff = 10000.0 * (1.0 - progress) + 200.0 * progress
                a_cutoff = np.clip(a_cutoff, 100, nyq - 100)
                sos_a = butter(4, a_cutoff / nyq, btype='low', output='sos')
                a_processed[:, start:end] = sosfilt(sos_a, a_tail[:, start:end])

                # B: HPF cutoff sweeps from 8000 Hz → 20 Hz
                b_cutoff = 8000.0 * (1.0 - progress) + 20.0 * progress
                b_cutoff = np.clip(b_cutoff, 20, nyq - 100)
                sos_b = butter(4, b_cutoff / nyq, btype='high', output='sos')
                b_processed[:, start:end] = sosfilt(sos_b, b_head[:, start:end])

            a_tail = a_processed
            b_head = b_processed

        return a_tail, b_head

    # ── Transition FX ────────────────────────────────────────────

    def _apply_fx(
        self, a_tail: np.ndarray, sr: int,
        fx_list: list[TransitionFX]
    ) -> np.ndarray:
        """
        Apply post-effects to the outgoing Track A tail.
        Effects are applied in order.
        """
        for fx in fx_list:
            if fx == TransitionFX.NONE:
                continue

            elif fx == TransitionFX.REVERB_TAIL:
                print("[FX] Applying Reverb Tail on Track A...")
                # Simple convolution reverb using a synthetic impulse response
                reverb_time = 1.5  # seconds
                ir_len = int(reverb_time * sr)
                ir = np.random.randn(ir_len) * np.exp(-3.0 * np.linspace(0, 1, ir_len))
                ir = ir / np.max(np.abs(ir))  # normalize IR
                # Apply to each channel
                wet = np.zeros_like(a_tail)
                for ch in range(a_tail.shape[0]):
                    conv = fftconvolve(a_tail[ch], ir, mode='full')[:a_tail.shape[1]]
                    wet[ch] = conv
                # Blend: 30% wet, 70% dry — increasing wet toward the end
                t = np.linspace(0.0, 1.0, a_tail.shape[1])
                wet_amount = 0.1 + 0.5 * t  # ramp from 10% to 60%
                a_tail = a_tail * (1.0 - wet_amount) + wet * wet_amount

            elif fx == TransitionFX.ECHO_OUT:
                print("[FX] Applying Echo Out on Track A...")
                # Feedback delay — 3 taps
                delay_ms = 60000.0 / max(120.0, 120.0)  # quarter-note at ~120 BPM
                delay_samples = int((delay_ms / 1000.0) * sr)
                feedback = 0.5
                result = a_tail.copy()
                for tap in range(1, 4):
                    offset = tap * delay_samples
                    gain = feedback ** tap
                    if offset < result.shape[1]:
                        end = min(result.shape[1], a_tail.shape[1] - offset)
                        result[:, offset:offset + end] += a_tail[:, :end] * gain
                # Prevent clipping
                peak = np.max(np.abs(result))
                if peak > 1.0:
                    result /= peak
                a_tail = result

            elif fx == TransitionFX.BACKSPIN:
                print("[FX] Applying Backspin on Track A tail...")
                # Take the last 0.5 seconds of the tail and reverse it with pitch ramp
                spin_len = min(int(0.5 * sr), a_tail.shape[1])
                spin_section = a_tail[:, -spin_len:].copy()
                # Reverse
                spin_section = np.flip(spin_section, axis=1)
                # Apply volume fade-out
                fade = np.linspace(1.0, 0.0, spin_len)
                spin_section = spin_section * fade
                # Replace the end of a_tail with the backspin
                a_tail[:, -spin_len:] = spin_section

        return a_tail

    # ── Main Mix Pipeline ────────────────────────────────────────

    def mix_transition(
        self, idx_a: int, idx_b: int, config: TransitionConfig
    ) -> tuple[np.ndarray, int]:
        """
        Execute a single A→B transition.

        Args:
            idx_a: Index of Track A in self.tracks
            idx_b: Index of Track B in self.tracks
            config: TransitionConfig specifying all parameters

        Returns:
            (audio_array, sample_rate)
        """
        track_a = self.tracks[idx_a]
        track_b = self.tracks[idx_b]

        print(f"\n{'━'*60}")
        print(f"[MIX] {track_a.path.name}  →  {track_b.path.name}")
        print(f"{'━'*60}")
        print(config)

        # Work on copies to preserve originals for chain_mix
        y_a = track_a.y.copy()
        y_b = track_b.y.copy()
        sr = track_a.sr

        # Ensure both tracks use the same sample rate
        if track_b.sr != sr:
            print(f"[!] Resampling Track B from {track_b.sr}Hz → {sr}Hz")
            y_b_resampled = []
            for ch in range(y_b.shape[0]):
                y_b_resampled.append(librosa.resample(y_b[ch], orig_sr=track_b.sr, target_sr=sr))
            y_b = np.array(y_b_resampled)

        # Ensure same number of channels (pad mono → stereo if needed)
        if y_a.shape[0] != y_b.shape[0]:
            target_ch = max(y_a.shape[0], y_b.shape[0])
            if y_a.shape[0] < target_ch:
                y_a = np.repeat(y_a, target_ch, axis=0)
            if y_b.shape[0] < target_ch:
                y_b = np.repeat(y_b, target_ch, axis=0)

        # ── Step 1: BPM Sync ──
        y_a, y_b, working_bpm, rate_a, rate_b = self._sync_bpm(
            y_a, track_a.bpm, y_b, track_b.bpm, sr, config
        )

        # ── Step 2: Key Sync ──
        y_a, y_b = self._sync_key(
            y_a, track_a.key, track_a.scale,
            y_b, track_b.key, track_b.scale,
            sr, config
        )

        # ── Step 3: Slice using actual beat grid ──
        # Scale downbeat positions by stretch rate.
        # time_stretch(rate=R) makes audio R× faster → sample positions scale by 1/R.
        downbeats_a = (track_a.downbeat_samples / rate_a).astype(int)
        downbeats_b = (track_b.downbeat_samples / rate_b).astype(int)

        # Look up actual downbeat positions for each slice point
        start_sample_a = self._get_downbeat_sample(
            downbeats_a, int(config.a_start_bar), working_bpm, sr)
        out_sample_a = self._get_downbeat_sample(
            downbeats_a, int(config.a_out_bar), working_bpm, sr)
        in_sample_b = self._get_downbeat_sample(
            downbeats_b, int(config.b_in_bar), working_bpm, sr)

        # Overlap: use beat grid span for precise alignment
        overlap_end_bar_a = int(config.a_out_bar + config.overlap_bars)
        overlap_end_sample_a = self._get_downbeat_sample(
            downbeats_a, overlap_end_bar_a, working_bpm, sr)
        mix_samples = overlap_end_sample_a - out_sample_a

        # Sanity: if mix_samples ended up <= 0 (edge case), fallback to BPM math
        if mix_samples <= 0:
            mix_samples = self._bars_to_samples(config.overlap_bars, working_bpm, sr)

        print(f"[GRID] A downbeats: {len(downbeats_a)} | B downbeats: {len(downbeats_b)}")
        print(f"[GRID] A start@bar{int(config.a_start_bar)}={start_sample_a} | "
              f"A out@bar{int(config.a_out_bar)}={out_sample_a} | "
              f"overlap={mix_samples} samples")
        print(f"[GRID] B in@bar{int(config.b_in_bar)}={in_sample_b}")

        # Clamp to track lengths
        start_sample_a = min(start_sample_a, y_a.shape[1])
        out_sample_a = min(out_sample_a, y_a.shape[1])
        in_sample_b = min(in_sample_b, y_b.shape[1])

        a_body = y_a[:, start_sample_a:out_sample_a]
        a_tail = y_a[:, out_sample_a:out_sample_a + mix_samples]

        b_head = y_b[:, in_sample_b:in_sample_b + mix_samples]

        # b_body: from after the overlap to b_end_bar (or end of track)
        b_body_start = in_sample_b + mix_samples
        if config.b_end_bar is not None:
            b_end_sample = self._get_downbeat_sample(
                downbeats_b, int(config.b_end_bar), working_bpm, sr)
            b_end_sample = min(b_end_sample, y_b.shape[1])
            b_body = y_b[:, b_body_start:b_end_sample]
        else:
            b_body = y_b[:, b_body_start:]

        # Pad if source ran out
        if a_tail.shape[1] < mix_samples:
            pad = mix_samples - a_tail.shape[1]
            a_tail = np.pad(a_tail, ((0, 0), (0, pad)))
        if b_head.shape[1] < mix_samples:
            pad = mix_samples - b_head.shape[1]
            b_head = np.pad(b_head, ((0, 0), (0, pad)))

        print(f"[CUT] A body: {a_body.shape[1]} samples | A tail: {a_tail.shape[1]} samples")
        print(f"[CUT] B head: {b_head.shape[1]} samples | B body: {b_body.shape[1]} samples")

        # ── Step 4: Transition FX (on A tail, before crossfade) ──
        if config.transition_fx:
            a_tail = self._apply_fx(a_tail, sr, config.transition_fx)

        # ── Step 5: EQ Profile ──
        a_tail, b_head = self._apply_eq_profile(a_tail, b_head, sr, config.eq_profile)

        # ── Step 6: Crossfade ──
        fade_out, fade_in = self._generate_crossfade(config.crossfade_curve, mix_samples)
        print(f"[FADE] Crossfade: {config.crossfade_curve.value} over {mix_samples} samples")
        mixed_region = (a_tail * fade_out) + (b_head * fade_in)

        # ── Step 7: Stitch ──
        print("[STITCH] Assembling final mix...")
        final = np.concatenate([a_body, mixed_region, b_body], axis=-1)

        # Trim trailing silence
        final, _ = librosa.effects.trim(final, top_db=40)

        duration = final.shape[1] / sr
        print(f"[✓] Transition complete — {duration:.1f}s total")

        return final, sr

    # ── Multi-Track Chaining ─────────────────────────────────────

    def chain_mix(self, configs: list[TransitionConfig]) -> tuple[np.ndarray, int]:
        """
        Chain multiple transitions to build a full DJ set.

        Expects len(configs) == len(self.tracks) - 1.
        configs[i] describes the transition from track[i] → track[i+1].

        Returns:
            (audio_array, sample_rate)
        """
        n = len(self.tracks)
        if len(configs) != n - 1:
            raise ValueError(
                f"Expected {n-1} TransitionConfigs for {n} tracks, got {len(configs)}"
            )

        print(f"\n{'═'*60}")
        print(f"[CHAIN] Building DJ set: {n} tracks, {len(configs)} transitions")
        print(f"{'═'*60}")

        # First transition
        result, sr = self.mix_transition(0, 1, configs[0])

        # Subsequent transitions: treat the running result as "Track A"
        for i in range(1, len(configs)):
            cfg = configs[i]
            next_track = self.tracks[i + 1]

            print(f"\n[CHAIN] Appending Track #{i+1}: {next_track.path.name}")

            # Temporarily replace track 0 data with the running result
            temp_info = _TrackInfo(
                path=Path("__chain_result__"),
                y=result, sr=sr,
                bpm=self.tracks[i].bpm,  # bpm of the last track mixed in
                key=self.tracks[i].key,
                scale=self.tracks[i].scale,
            )
            original_0 = self.tracks[0]
            self.tracks[0] = temp_info
            result, sr = self.mix_transition(0, i + 1, cfg)
            self.tracks[0] = original_0  # restore

        duration = result.shape[1] / sr
        print(f"\n[✓] Chain mix complete — {duration:.1f}s total, {n} tracks")
        return result, sr

    # ── Export ────────────────────────────────────────────────────

    @staticmethod
    def export(audio_sr: tuple[np.ndarray, int], output_path: str | Path):
        """
        Export the mix to a file.

        Args:
            audio_sr: Tuple of (audio_array, sample_rate) as returned by mix_transition / chain_mix
            output_path: Output file path (.wav, .flac, etc.)
        """
        y, sr = audio_sr
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        y_out = y[0] if y.shape[0] == 1 else y.T
        sf.write(str(out_p), y_out, sr)
        print(f"\n[+] Exported mix to: {out_p}")
        print(f"    Duration: {y.shape[1]/sr:.1f}s | Channels: {y.shape[0]} | SR: {sr}Hz")


# ═══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DJ Automix Engine — Professional Transition Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preset mode (simplest)
  %(prog)s --track-a A.mp3 --track-b B.mp3 --preset smooth_blend

  # Full manual control
  %(prog)s --track-a A.mp3 --a-out-bar 64 \\
           --track-b B.mp3 --b-in-bar 16 \\
           --overlap 16 --bpm-sync b_to_a --key-sync b_to_a \\
           --crossfade equal_power --eq-profile bass_swap \\
           --fx reverb_tail --output my_mix.wav

Available presets: """ + ", ".join(PRESETS.keys())
    )

    # ── Tracks ──
    parser.add_argument("--track-a", type=str, required=True,
                        help="Path to the outgoing track (Track A)")
    parser.add_argument("--track-b", type=str, required=True,
                        help="Path to the incoming track (Track B)")

    # ── Preset (shortcut) ──
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(PRESETS.keys()),
                        help="Use a built-in transition preset (overrides manual params)")

    # ── Slice Points ──
    parser.add_argument("--a-start-bar", type=float, default=1,
                        help="First bar of Track A to include (default: 1)")
    parser.add_argument("--a-out-bar", type=float, default=64,
                        help="Bar number in Track A where the mix-out begins (default: 64)")
    parser.add_argument("--b-in-bar", type=float, default=1,
                        help="Bar number in Track B where the mix-in begins (default: 1)")
    parser.add_argument("--b-end-bar", type=float, default=None,
                        help="Bar number in Track B to stop playing (default: play to end)")
    parser.add_argument("--overlap", type=float, default=16,
                        help="Crossfade overlap duration in bars (default: 16)")

    # ── BPM ──
    parser.add_argument("--bpm-sync", type=str, default="b_to_a",
                        choices=["a_to_b", "b_to_a", "middle", "none"],
                        help="BPM sync mode (default: b_to_a)")
    parser.add_argument("--target-bpm", type=float, default=None,
                        help="Force both tracks to this absolute BPM")

    # ── Key ──
    parser.add_argument("--key-sync", type=str, default="none",
                        choices=["a_to_b", "b_to_a", "none"],
                        help="Key sync mode (default: none)")
    parser.add_argument("--pitch-shift", type=float, default=None,
                        help="Manual pitch shift in semitones (overrides auto key-sync)")

    # ── Crossfade ──
    parser.add_argument("--crossfade", type=str, default="equal_power",
                        choices=["linear", "equal_power", "s_curve", "exponential"],
                        help="Crossfade curve shape (default: equal_power)")

    # ── EQ ──
    parser.add_argument("--eq-profile", type=str, default="bass_swap",
                        choices=["none", "bass_swap", "full_eq", "filter_sweep"],
                        help="DJ EQ profile during transition (default: bass_swap)")

    # ── Effects ──
    parser.add_argument("--fx", type=str, nargs="*", default=[],
                        choices=["none", "reverb_tail", "echo_out", "backspin"],
                        help="Transition effects (space-separated, default: none)")

    # ── Output ──
    parser.add_argument("--output", type=str, default="automix_output.wav",
                        help="Output file path (default: automix_output.wav)")

    args = parser.parse_args()

    # ── Build Config ──
    if args.preset:
        config = copy.deepcopy(PRESETS[args.preset])
        # Allow overriding slice points even with preset
        if args.a_start_bar != 1:
            config.a_start_bar = args.a_start_bar
        if args.a_out_bar != 64:
            config.a_out_bar = args.a_out_bar
        if args.b_in_bar != 1:
            config.b_in_bar = args.b_in_bar
        if args.b_end_bar is not None:
            config.b_end_bar = args.b_end_bar
        if args.overlap != 16:
            config.overlap_bars = args.overlap
    else:
        fx_list = [TransitionFX(f) for f in args.fx if f != "none"] if args.fx else []
        config = TransitionConfig(
            a_start_bar=args.a_start_bar,
            a_out_bar=args.a_out_bar,
            b_in_bar=args.b_in_bar,
            b_end_bar=args.b_end_bar,
            overlap_bars=args.overlap,
            bpm_sync=BpmSyncMode(args.bpm_sync),
            target_bpm=args.target_bpm,
            key_sync=KeySyncMode(args.key_sync),
            pitch_shift_semitones=args.pitch_shift,
            crossfade_curve=CrossfadeCurve(args.crossfade),
            eq_profile=EQProfile(args.eq_profile),
            transition_fx=fx_list,
        )

    # ── Run ──
    engine = AutomixEngine()
    engine.add_track(args.track_a)
    engine.add_track(args.track_b)

    result = engine.mix_transition(0, 1, config)
    engine.export(result, args.output)


if __name__ == "__main__":
    main()
