"""
Music Splitter Script

This script slices a given audio track into fixed-length musical patterns (e.g., repeating 4-bar segments).
It works by either automatically estimating the track's BPM or taking a user-provided BPM,
then calculating the exact length of each pattern based on the specified number of bars and beats.

Useful for DJs, producers, or ML practitioners looking to split long tracks or DJ mixes into 
manageable, tempo-synced musical chunks.

Key features:
- Automatically estimates BPM using librosa (if not specified).
- Supports stereo audio preservation during splitting.
- Configurable number of bars and beats per bar.
- Optional time offset (to sync the first downbeat before slicing).

Example usage:
    uv run python playground/music_splitter.py "Above & Beyond - Far From In Love (Original Mix).mp3" --output-dir playground/output
"""
import argparse
import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Split music into fixed-length patterns (e.g., 4 bars).")
    parser.add_argument("input_file", type=str, help="Path to the input audio file")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the split patterns")
    parser.add_argument("--bpm", type=float, default=None, help="BPM of the audio (if known). If not provided, it will be estimated.")
    parser.add_argument("--bars", type=float, default=4.0, help="Number of bars per pattern (default: 4)")
    parser.add_argument("--beats-per-bar", type=int, default=4, help="Number of beats per bar (default: 4)")
    parser.add_argument("--offset-ms", type=float, default=0.0, help="Offset in milliseconds to start slicing (default: 0)")
    
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File '{input_path}' not found.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading '{input_path}'...")
    # Load audio while preserving the original sample rate and stereo channels
    y, sr = librosa.load(input_path, sr=None, mono=False)
    
    bpm = args.bpm
    if bpm is None:
        print("Estimating BPM...")
        # librosa needs mono to estimate tempo
        if y.ndim > 1:
            y_mono = librosa.to_mono(y)
        else:
            y_mono = y
            
        tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
        bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        print(f"Estimated BPM: {bpm:.2f}")
    else:
        print(f"Using provided BPM: {bpm}")
        
    beats_per_pattern = args.bars * args.beats_per_bar
    pattern_duration_sec = (60.0 / bpm) * beats_per_pattern
    pattern_len_samples = int(np.round(pattern_duration_sec * sr))
    
    offset_samples = int(np.round((args.offset_ms / 1000.0) * sr))
    
    # Determine total samples depending on whether audio is mono or stereo
    total_samples = y.shape[-1]
    
    print(f"Pattern duration: {pattern_duration_sec:.2f} seconds ({pattern_len_samples} samples)")
    
    if offset_samples >= total_samples:
        print("Error: Offset is beyond the end of the audio.")
        return
        
    current_sample = offset_samples
    pattern_idx = 0
    
    while current_sample < total_samples:
        end_sample = current_sample + pattern_len_samples
        
        if y.ndim > 1:
            # Stereo: shape is (channels, samples)
            chunk = y[:, current_sample:end_sample]
            # sf.write expects shape (samples, channels)
            chunk_to_write = chunk.T
        else:
            # Mono: shape is (samples,)
            chunk_to_write = y[current_sample:end_sample]
        
        # We can drop the last chunk if it's extremely short, but it's safe to write it out.
        
        out_filename = f"{input_path.stem}_pattern_{pattern_idx:03d}.wav"
        out_path = output_dir / out_filename
        
        sf.write(out_path, chunk_to_write, sr)
        print(f"Saved: {out_filename}")
        
        current_sample += pattern_len_samples
        pattern_idx += 1
        
    print(f"Successfully split into {pattern_idx} patterns.")

if __name__ == "__main__":
    main()
