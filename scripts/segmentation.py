import librosa
import numpy as np
import webrtcvad
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import sys

# The script should accept an input file path as a command-line argument
# or use the default if none is provided
if len(sys.argv) > 1:
    audio_file = sys.argv[1]
else:
    # Default file (used when script is run directly)
    audio_file = "../audio_files/denoised.wav"

# Get output segments directory from command line or environment variable
segments_dir = "../audio_files/segments"  # Default
if len(sys.argv) > 2:
    segments_dir = sys.argv[2]
elif 'SEGMENTS_DIR' in os.environ:
    segments_dir = os.environ['SEGMENTS_DIR']
elif os.path.exists("audio_files/segments"):
    segments_dir = "audio_files/segments"

# Make sure the segments directory exists
if not os.path.exists(segments_dir):
    print(f"Creating segments directory at {segments_dir}")
    try:
        os.makedirs(segments_dir)
    except Exception as e:
        print(f"Error creating segments directory: {e}")
    
# If the file doesn't exist, try a direct path as fallback
if not os.path.exists(audio_file):
    print(f"Warning: File {audio_file} not found, checking alternatives")
    if os.path.exists("audio_files/denoised.wav"):
        audio_file = "audio_files/denoised.wav"
    elif os.path.exists("audio_files/0638.wav"):
        audio_file = "audio_files/0638.wav"

print(f"Loading audio from: {audio_file}")
print(f"Segments will be saved to: {segments_dir}")
y, sr = librosa.load(audio_file, sr=None)  # Load as float32 (-1 to 1)

# Function to apply VAD and get speech timestamps
def get_speech_timestamps(audio, sample_rate, frame_duration_ms=30):
    vad = webrtcvad.Vad(1)  # Low aggressiveness for better speech detection
    frame_size = int(sample_rate * frame_duration_ms / 1000)

    # Convert audio to PCM 16-bit
    audio_int16 = (audio * 32767).astype(np.int16)

    num_frames = len(audio_int16) // frame_size
    frames = np.array_split(audio_int16[:num_frames * frame_size], num_frames)

    vad_result = [vad.is_speech(frame.tobytes(), sample_rate) for frame in frames]

    timestamps = []
    start = None

    for i, is_speech in enumerate(vad_result):
        timestamp = i * frame_duration_ms / 1000  # Convert to seconds

        if is_speech and start is None:
            start = timestamp  # Speech started
        elif not is_speech and start is not None:
            timestamps.append((start, timestamp))  # Speech ended
            start = None

    if start is not None:
        timestamps.append((start, len(audio) / sample_rate))

    return timestamps

# Get VAD-based Speech Timestamps
speech_timestamps = get_speech_timestamps(y, sr)

# Energy-based Speaker Turn Detection
def detect_energy_shifts(audio, sample_rate, window_size=1024, threshold_ratio=2.0):
    energy = np.array([np.sum(np.abs(audio[i:i+window_size])) for i in range(0, len(audio), window_size)])
    
    # Compute threshold for detecting large energy shifts
    mean_energy = np.mean(energy)
    threshold = mean_energy * threshold_ratio

    segment_boundaries = []
    
    for i in range(1, len(energy)):
        if abs(energy[i] - energy[i-1]) > threshold:
            segment_boundaries.append(i * window_size / sample_rate)  # Convert to seconds

    return segment_boundaries

energy_boundaries = detect_energy_shifts(y, sr)

# Merge VAD and Energy-Based Boundaries
final_segments = []

for start, end in speech_timestamps:
    split_points = [b for b in energy_boundaries if start < b < end]  # Only within current speech segment

    # Create sub-segments
    segment_start = start
    for split in split_points:
        final_segments.append((segment_start, split))
        segment_start = split

    final_segments.append((segment_start, end))  # Add last segment

# Adaptive Windowing (Merging small segments)
def merge_short_segments(segments, min_duration=1.0):
    merged = []
    prev_start, prev_end = segments[0]

    for start, end in segments[1:]:
        if end - prev_start < min_duration:  
            prev_end = end  # Merge
        else:
            merged.append((prev_start, prev_end))
            prev_start, prev_end = start, end

    merged.append((prev_start, prev_end))
    return merged

final_segments = merge_short_segments(final_segments, min_duration=1.5)

# Save Segmented Audio Files
for i, (start, end) in enumerate(final_segments):
    segment = y[int(start * sr): int(end * sr)]
    output_file = os.path.join(segments_dir, f"segment_{i+1}.wav")
    wav.write(output_file, sr, (segment * 32767).astype(np.int16))
    # print(f"Saved {output_file} ({round(end - start, 2)} sec)")

# # Plot Segmentation
# plt.figure(figsize=(12, 4))
# librosa.display.waveshow(y, sr=sr, alpha=0.5)
# for start, end in final_segments:
#     plt.axvspan(start, end, color='red', alpha=0.3)
# plt.title("Speaker Segmentation")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.show()
