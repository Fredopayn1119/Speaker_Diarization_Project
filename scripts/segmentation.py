import os
import numpy as np
import librosa
import webrtcvad
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa.display
from scipy.spatial.distance import cosine

# VAD Detection
def get_speech_timestamps(audio, sample_rate, frame_duration_ms=30):
    vad = webrtcvad.Vad(1)
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    audio_int16 = (audio * 32767).astype(np.int16)
    num_frames = len(audio_int16) // frame_size
    frames = np.array_split(audio_int16[:num_frames * frame_size], num_frames)

    vad_result = [vad.is_speech(frame.tobytes(), sample_rate) for frame in frames]

    timestamps = []
    start = None
    for i, is_speech in enumerate(vad_result):
        timestamp = i * frame_duration_ms / 1000
        if is_speech and start is None:
            start = timestamp
        elif not is_speech and start is not None:
            timestamps.append((start, timestamp))
            start = None
    if start is not None:
        timestamps.append((start, len(audio) / sample_rate))
    return timestamps

# MFCC Speaker Turns
def refine_speaker_turns(audio, sr, segment, win_size=1.0, hop_size=0.5, threshold=0.6):
    start_time, end_time = segment
    segment_audio = audio[int(start_time * sr):int(end_time * sr)]
    duration = end_time - start_time

    if duration < 2 * win_size:
        return [segment]  # too short, return as-is

    times = np.arange(0, duration - win_size, hop_size)
    boundaries = [start_time]
    prev_mfcc = None

    for t in times:
        frame = segment_audio[int(t * sr):int((t + win_size) * sr)]
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        if prev_mfcc is not None:
            dist = cosine(prev_mfcc, mfcc_mean)
            if dist > threshold:
                boundaries.append(start_time + t)
        prev_mfcc = mfcc_mean

    boundaries.append(end_time)
    refined_segments = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    return refined_segments

# Merge Short Segments
def merge_short_segments(segments, min_duration=1.0):
    if not segments:
        return []
    merged = []
    prev_start, prev_end = segments[0]
    for start, end in segments[1:]:
        if end - prev_start < min_duration:
            prev_end = end
        else:
            merged.append((prev_start, prev_end))
            prev_start, prev_end = start, end
    merged.append((prev_start, prev_end))
    return merged

# Main Logic
def segment_audio_native(input_audio_path, segments_output_dir, plot_output_path=None, show_plot=False):
    if not os.path.exists(segments_output_dir):
        os.makedirs(segments_output_dir)
    
    print(f"Loading audio from: {input_audio_path}")
    y, sr = librosa.load(input_audio_path, sr=None)

    # Step 1: VAD - Rough speech regions
    speech_timestamps = get_speech_timestamps(y, sr)

    # Step 2: Refine speaker changes within speech using MFCC similarity
    refined_segments = []
    for start, end in speech_timestamps:
        refined = refine_speaker_turns(y, sr, (start, end), win_size=1.0, hop_size=0.5, threshold=0.6)
        refined_segments.extend(refined)

    # Step 3: Merge short segments
    final_segments = merge_short_segments(refined_segments, min_duration=1.5)
    

    # Step 4: Save segments
    for i, (start, end) in enumerate(final_segments):
        segment = y[int(start * sr):int(end * sr)]
        out_file = os.path.join(segments_output_dir, f"segment_{i+1}.wav")
        wav.write(out_file, sr, (segment * 32767).astype(np.int16))
        print(f"Saved: {out_file} ({round(end - start, 2)} sec)")

    # Step 5: Plotting
    if plot_output_path or show_plot:
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.5)
        for start, end in final_segments:
            plt.axvspan(start, end, color='red', alpha=0.3)
        plt.title("Speaker Segmentation")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        if plot_output_path:
            plt.savefig(plot_output_path)
            print(f"Segmentation plot saved to: {plot_output_path}")
        if show_plot:
            plt.show()
        plt.close()

    return final_segments

if __name__ == "__main__":
    segments = segment_audio_native(
        input_audio_path="../audio/denoised.wav",
        segments_output_dir="../audio/segments",
        plot_output_path="../audio/segmentation_plot.png",
        show_plot=False
    )
