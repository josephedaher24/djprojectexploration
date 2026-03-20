import os
import json
from essentia.standard import MusicExtractor


def main() -> None:
    # Path to the music directory
    music_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'music')

    # Find .mp3 files
    mp3_files = [f for f in os.listdir(music_dir) if f.endswith('.mp3')]

    if not mp3_files:
        print("No .mp3 files found in the music directory.")
        return

    # For testing, use the first .mp3 file
    audio_file = os.path.join(music_dir, mp3_files[0])
    print(f"Extracting features from: {audio_file}")

    # Create the MusicExtractor
    extractor = MusicExtractor()

    # Extract features
    features, metadata = extractor(audio_file)

    # Print some features
    print("Extracted features:")
    print(f"Duration: {metadata['metadata.audio_properties.length']:.2f} seconds")
    print(f"BPM: {features['rhythm.bpm']:.2f}")
    print(f"Key: {features['tonal.key_krumhansl.key']}")
    print(f"Scale: {features['tonal.key_krumhansl.scale']}")
    print(f"Danceability: {features['rhythm.danceability']:.2f}")
    print(f"Loudness: {features['lowlevel.average_loudness']:.2f} dB")

    # Additional features
    print(f"MFCC (mean): {features['lowlevel.mfcc.mean']}")
    print(f"Spectral Centroid (mean): {features['lowlevel.spectral_centroid.mean']:.2f}")
    print(f"Spectral Contrast (mean): {features['lowlevel.spectral_contrast_coeffs.mean']}")

    # Prepare data to save
    extracted_data = {
        'filename': os.path.basename(audio_file),
        'duration': metadata['metadata.audio_properties.length'],
        'bpm': features['rhythm.bpm'],
        'key': features['tonal.key_krumhansl.key'],
        'scale': features['tonal.key_krumhansl.scale'],
        'danceability': features['rhythm.danceability'],
        'loudness': features['lowlevel.average_loudness'],
        'mfcc_mean': features['lowlevel.mfcc.mean'].tolist(),  # Convert to list for JSON
        'spectral_centroid_mean': features['lowlevel.spectral_centroid.mean'],
        'spectral_contrast_mean': features['lowlevel.spectral_contrast_coeffs.mean'].tolist()
    }

    # Save to JSON file
    output_file = os.path.join(music_dir, 'extracted_features.json')
    with open(output_file, 'w') as f:
        json.dump(extracted_data, f, indent=4)

    print(f"Features saved to: {output_file}")
    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
