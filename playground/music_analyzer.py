"""
Music Analyzer Script

This script acts as a DJ software-style audio analysis tool.
It extracts:
1. BPM and Beat Grid (using librosa)
2. Musical Key and Scale (using essentia)

It then visualizes the full waveform with the beat grid and saves the analysis data to a JSON file.

Example usage:
    uv run python playground/music_analyzer.py "Above & Beyond - Far From In Love (Original Mix).mp3"

"""
import argparse
import os
import json
import numpy as np
import subprocess
import librosa
import librosa.display
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
from essentia.standard import MonoLoader, KeyExtractor
from scipy.interpolate import interp1d

def analyze_track(audio_path, output_dir=None):
    filepath = Path(audio_path)
    if not filepath.exists():
        print(f"Error: Could not find '{filepath}'.")
        return

    print(f"Loading '{filepath}' for analysis...")
    y, sr = librosa.load(filepath, sr=None, mono=True)
    
    print("Extracting BPM and Beat Grid...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    print("Extracting Key and Scale (Essentia)...")
    try:
        audio_essentia = MonoLoader(filename=str(filepath))()
        key, scale, strength = KeyExtractor()(audio_essentia)
    except Exception as e:
        print(f"Warning: Failed to extract key with Essentia ({e}). Defaulting to Unknown.")
        key, scale = "Unknown", "Unknown"
        
    print("Extracting Phrases and Auto CUEs (32-beat grid + RMS)...")
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    rms_interp = interp1d(rms_times, rms, bounds_error=False, fill_value=0)(beat_times)
    
    print("Extracting approximate Vocal/Synth density (HPSS)...")
    y_harm, y_perc = librosa.effects.hpss(y)
    rms_harm = librosa.feature.rms(y=y_harm)[0]
    rms_h_interp = interp1d(rms_times, rms_harm, bounds_error=False, fill_value=0)(beat_times)
    
    phrases = []
    cues = []
    
    beats_per_phrase = 32
    for i in range(0, len(beat_times), beats_per_phrase):
        start_time = float(beat_times[i])
        end_time = float(beat_times[i+beats_per_phrase] if i+beats_per_phrase < len(beat_times) else beat_times[-1])
        
        chunk_energy = float(np.mean(rms_interp[i:i+beats_per_phrase]))
        chunk_harm = float(np.mean(rms_h_interp[i:i+beats_per_phrase]))
        
        label = "Drop / Main" if chunk_energy > np.mean(rms) * 1.2 else "Intro / Break"
        vocal_tag = "Has Vocals/Synths" if chunk_harm > np.mean(rms_harm) else "Instrumental/Beat"
        
        phrases.append({
            "start": start_time,
            "end": end_time,
            "label": label,
            "vocal_proxy": vocal_tag
        })
        
        cues.append({
            "time": start_time,
            "label": f"{label} Cue"
        })
        
    print(f"--- Analysis Complete ---")
    print(f"BPM: {bpm:.2f} | Key: {key} {scale} | Phrases: {len(phrases)} | Cues: {len(cues)}")
    
    analysis_data = {
        "audio_file": filepath.name,
        "bpm": bpm,
        "key": f"{key} {scale}",
        "beat_times": beat_times.tolist(),
        "phrases": phrases,
        "cues": cues
    }
    
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{filepath.stem}_analysis.json"
    else:
        json_path = filepath.with_name(f"{filepath.stem}_analysis.json")
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=4)
    print(f"Saved analysis to '{json_path}'.")    
        
    print("Generating Visualization...")
    fig, (ax_main, ax_overview) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 1.5]})
    
    # --- Main Waveform (ax_main) ---
    librosa.display.waveshow(y, sr=sr, alpha=0.5, color='b', ax=ax_main)
    
    downbeats = beat_times[::4]
    lightbeats = [b for i, b in enumerate(beat_times) if i % 4 != 0]
    
    # Plot beats
    ax_main.vlines(lightbeats, -1, 1, color='gray', alpha=0.5, linestyle='-', linewidth=1)
    ax_main.vlines(downbeats, -1, 1, color='red', alpha=0.9, linestyle='-', linewidth=2)
    ax_main.plot(downbeats, [1]*len(downbeats), 'rv', markersize=6)
    ax_main.plot(downbeats, [-1]*len(downbeats), 'r^', markersize=6)
    
    from matplotlib.ticker import FuncFormatter
    def time_to_bars(x, pos):
        bars = (x * bpm) / 240.0
        return f"{bars:.1f} Bars"
    ax_main.xaxis.set_major_formatter(FuncFormatter(time_to_bars))
    
    for c in cues:
        ax_main.axvline(c['time'], color='lime', linestyle='-', linewidth=2)
        
    playhead_main = ax_main.axvline(x=0, color='cyan', linewidth=3)
    playhead_main.set_visible(False)
    
    ax_main.set_title(f"Waveform & Analysis | {filepath.name} | BPM: {bpm:.2f} | Key: {key} {scale}")
    ax_main.set_ylabel("Amplitude")
    
    # --- Overview Waveform (ax_overview) ---
    librosa.display.waveshow(y, sr=sr, alpha=0.3, color='b', ax=ax_overview)
    
    colors_dict = {"Intro / Break": "salmon", "Drop / Main": "royalblue"}
    for p in phrases:
        c = colors_dict.get(p['label'], 'gray')
        ax_overview.axvspan(p['start'], p['end'], alpha=0.8, color=c)
        
    playhead_overview = ax_overview.axvline(x=0, color='cyan', linewidth=3)
    playhead_overview.set_visible(False)
    
    ax_overview.set_yticks([])
    ax_overview.set_xlabel("Time (s) / Structural Overview")
    
    plt.subplots_adjust(bottom=0.2, hspace=0.3)
    
    axplay = plt.axes([0.75, 0.05, 0.1, 0.075])
    axstop = plt.axes([0.86, 0.05, 0.1, 0.075])
    bplay = Button(axplay, 'Play')
    bstop = Button(axstop, 'Stop')
    
    play_process = None
    start_time_play = None
    
    def update_playhead():
        nonlocal start_time_play
        if start_time_play is not None and play_process is not None and play_process.poll() is None:
            elapsed = time.time() - start_time_play
            playhead_main.set_xdata([elapsed, elapsed])
            playhead_overview.set_xdata([elapsed, elapsed])
            fig.canvas.draw_idle()
            
    timer = fig.canvas.new_timer(interval=50)
    timer.add_callback(update_playhead)
    
    def play_audio(event):
        nonlocal play_process, start_time_play
        if play_process is None or play_process.poll() is not None:
            start_time_play = time.time()
            playhead_main.set_visible(True)
            playhead_overview.set_visible(True)
            timer.start()
            play_process = subprocess.Popen(["afplay", str(filepath)])
            
    def stop_audio(event=None):
        nonlocal play_process, start_time_play
        if play_process is not None:
            play_process.terminate()
            play_process = None
        start_time_play = None
        timer.stop()
        playhead_main.set_visible(False)
        playhead_overview.set_visible(False)
        fig.canvas.draw_idle()
            
    def on_close(event):
        stop_audio()
        
    bplay.on_clicked(play_audio)
    bstop.on_clicked(stop_audio)
    fig.canvas.mpl_connect('close_event', on_close)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="DJ-Style Music Analyzer (BPM, Grid, Key)")
    parser.add_argument("input_file", type=str, help="Path to the audio file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the analysis JSON (default: same as input file)")
    args = parser.parse_args()
    
    analyze_track(args.input_file, args.output_dir)

if __name__ == "__main__":
    main()
