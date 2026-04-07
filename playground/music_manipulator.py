"""
Music Manipulator Script

This script acts as a DJ's audio effect engine (DSP Toolkit).
It allows for direct modification of the audio file without altering the structural base.
Key capabilities:
1. Time-Stretching (Change tempo without changing pitch)
2. Pitch-Shifting (Change key without changing tempo)
3. Isolator EQ (Low-pass/High-pass filtering)
4. Classic DJ Effects (Gain adjustment, Reverse playback)

Example usage:
    uv run python playground/music_manipulator.py "Above & Beyond - Far From In Love.mp3" --time-stretch 1.05 --pitch-shift -1.0 --low-pass 800
"""
import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt

class DJManipulator:
    """
    A class that provides DJ-style audio manipulations (DSP):
    - Time Stretching (Change Tempo)
    - Pitch Shifting (Change Key)
    - Isolator EQs (Low-pass, High-pass)
    - Basic effects (Reverse, Gain)
    """
    def __init__(self, audio_path):
        self.audio_path = Path(audio_path)
        if not self.audio_path.exists():
            raise FileNotFoundError(f"Audio file '{audio_path}' not found.")
            
        print(f"[+] Loading {self.audio_path.name}...")
        # Load audio preserving its native sample rate and channels (mono=False)
        self.y, self.sr = librosa.load(self.audio_path, sr=None, mono=False)
        
        # If it's single channel (mono), reshape it to (1, N) so stereo loop logic works universally
        if self.y.ndim == 1:
            self.y = self.y.reshape(1, -1)
            
        print(f"    Original info -> Channels: {self.y.shape[0]} | Sample Rate: {self.sr}Hz | Duration: {(self.y.shape[1]/self.sr):.2f}s")

    def apply_pitch_shift(self, n_steps):
        """Shifts the pitch by n_steps (semitones) without affecting tempo."""
        print(f"[~] Applying Pitch Shift: {n_steps:+.2f} semitones...")
        y_shifted = np.zeros_like(self.y)
        for i in range(self.y.shape[0]):
            y_shifted[i] = librosa.effects.pitch_shift(self.y[i], sr=self.sr, n_steps=n_steps)
        self.y = y_shifted

    def apply_time_stretch(self, rate):
        """Stretches the time by a factor of 'rate'. >1.0 makes it faster, <1.0 makes it slower."""
        print(f"[~] Applying Time Stretch: {rate:.2f}x speed...")
        y_stretched = []
        for i in range(self.y.shape[0]):
            # librosa.effects.time_stretch expands or shrinks the array length
            y_stretched.append(librosa.effects.time_stretch(self.y[i], rate=rate))
        self.y = np.array(y_stretched)

    def apply_lowpass(self, cutoff=1000):
        """Applies a Low-pass filter (cuts high frequencies, useful for drops)."""
        print(f"[~] Applying Low-pass Filter: {cutoff}Hz cutoff...")
        nyq = 0.5 * self.sr
        normal_cutoff = cutoff / nyq
        sos = butter(4, normal_cutoff, btype='low', output='sos')
        self.y = sosfilt(sos, self.y)

    def apply_highpass(self, cutoff=300):
        """Applies a High-pass filter (cuts bass/low kicks, useful for buildups)."""
        print(f"[~] Applying High-pass Filter: {cutoff}Hz cutoff...")
        nyq = 0.5 * self.sr
        normal_cutoff = cutoff / nyq
        sos = butter(4, normal_cutoff, btype='high', output='sos')
        self.y = sosfilt(sos, self.y)

    def apply_gain(self, gain_db):
        """Adjusts the track volume."""
        print(f"[~] Applying Gain: {gain_db:+.1f}dB...")
        linear_gain = 10 ** (gain_db / 20.0)
        self.y = self.y * linear_gain

    def apply_reverse(self):
        """Reverses the audio (classic turntable backspin effect)."""
        print("[~] Reversing Audio Track...")
        self.y = np.flip(self.y, axis=1)

    def export(self, output_path):
        """Exports the modified track to a WAV/FLAC file."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        
        # Transpose back if stereo because soundfile expects (frames, channels)
        y_out = self.y[0] if self.y.shape[0] == 1 else self.y.T
        sf.write(str(out_p), y_out, self.sr)
        print(f"[+] Successfully saved manipulated track to: {out_p}")


def main():
    parser = argparse.ArgumentParser(description="DJ Music Manipulator (DSP Tools)")
    parser.add_argument("input_file", type=str, help="Path to the original audio file")
    parser.add_argument("--pitch-shift", type=float, default=None, help="Shift key up/down by N semitones (e.g. 1.5, -2.0)")
    parser.add_argument("--time-stretch", type=float, default=None, help="Stretch tempo. e.g. 1.05 = 5% faster (BPM * 1.05)")
    parser.add_argument("--low-pass", type=float, default=None, help="Lowpass filter cutoff in Hz (e.g. 1000)")
    parser.add_argument("--high-pass", type=float, default=None, help="Highpass filter cutoff in Hz (e.g. 300 to kill bass kick)")
    parser.add_argument("--gain", type=float, default=None, help="Adjust volume by Gain in dB (e.g. -3.0)")
    parser.add_argument("--reverse", action="store_true", help="Play track in reverse")
    parser.add_argument("--output", type=str, default=None, help="Output path for the manipulated file")
    args = parser.parse_args()

    dj = DJManipulator(args.input_file)

    if args.pitch_shift is not None:
        dj.apply_pitch_shift(args.pitch_shift)
    if args.time_stretch is not None:
        dj.apply_time_stretch(args.time_stretch)
    if args.low_pass is not None:
        dj.apply_lowpass(args.low_pass)
    if args.high_pass is not None:
        dj.apply_highpass(args.high_pass)
    if args.gain is not None:
        dj.apply_gain(args.gain)
    if args.reverse:
        dj.apply_reverse()
        
    out_path = args.output
    if not out_path:
        p = Path(args.input_file)
        out_path = p.with_name(f"{p.stem}_manipulated.wav")
        
    dj.export(out_path)

if __name__ == "__main__":
    main()
