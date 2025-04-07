import librosa
import numpy as np
import webrtcvad
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import os
import sys
import argparse
from pathlib import Path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Audio segmentation for speaker diarization")
    parser.add_argument("--input", type=str, default="../audio_files/denoised.wav",
                      help="Input audio file path")
    parser.add_argument("--output_dir", type=str, default="../audio_files/segments",
                      help="Directory to save segmented audio files")
    parser.add_argument("--vad_mode", type=int, choices=[0, 1, 2, 3], default=1,
                      help="VAD aggressiveness (0-3, higher is more aggressive)")
    parser.add_argument("--min_duration", type=float, default=1.0,
                      help="Minimum segment duration in seconds")
    parser.add_argument("--sliding_window", action="store_true",
                      help="Use sliding window for more accurate boundaries")
    parser.add_argument("--frame_duration_ms", type=int, default=30,
                      help="Frame duration in milliseconds for VAD")
    parser.add_argument("--adaptive_threshold", action="store_true",
                      help="Use adaptive energy thresholding")
    parser.add_argument("--plot", action="store_true",
                      help="Plot segmentation results")
    parser.add_argument("--use_bic", action="store_true",
                      help="Use BIC for speaker change detection")
    
    return parser.parse_args()

def ensure_dir_exists(directory):
    """Ensure the directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)
    return directory

def load_audio(file_path):
    """Load audio file with verification and fallbacks"""
    # Try to load the provided file
    if os.path.exists(file_path):
        print(f"Loading audio from: {file_path}")
        y, sr = librosa.load(file_path, sr=None)
        return y, sr, file_path
    
    # Fallback paths to try
    alternatives = [
        Path("audio_files/denoised.wav"),
        Path("audio_files/0638.wav"),
        Path("../audio_files/denoised.wav"),
        Path("../audio_files/0638.wav")
    ]
    
    for alt_path in alternatives:
        if os.path.exists(alt_path):
            print(f"File {file_path} not found, using alternative: {alt_path}")
            y, sr = librosa.load(str(alt_path), sr=None)
            return y, sr, str(alt_path)
    
    raise FileNotFoundError(f"Could not find audio file {file_path} or any alternatives")

def get_speech_segments_vad(audio, sample_rate, vad_mode=1, frame_duration_ms=30, sliding_window=False):
    """
    Use WebRTC VAD to detect speech segments with optional sliding window
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    sample_rate : int
        Sample rate of the audio
    vad_mode : int (0-3)
        Aggressiveness mode (higher is more aggressive)
    frame_duration_ms : int
        Frame duration in milliseconds
    sliding_window : bool
        Whether to use sliding window for better boundary detection
    
    Returns:
    --------
    List of tuples with (start_time, end_time) in seconds
    """
    vad = webrtcvad.Vad(vad_mode)
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    
    # Convert audio to PCM 16-bit
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # If sliding window is enabled, use overlapping frames
    if sliding_window:
        # 50% overlap for sliding windows
        hop_length = frame_size // 2
        num_frames = (len(audio_int16) - frame_size) // hop_length + 1
        
        # Pre-allocate results array for efficiency
        vad_result = np.zeros(num_frames, dtype=bool)
        
        # Process each frame
        for i in range(num_frames):
            start_idx = i * hop_length
            frame = audio_int16[start_idx:start_idx + frame_size]
            
            # Only process if we have a full frame
            if len(frame) == frame_size:
                vad_result[i] = vad.is_speech(frame.tobytes(), sample_rate)
    else:
        # Standard non-overlapping frames
        num_frames = len(audio_int16) // frame_size
        frames = np.array_split(audio_int16[:num_frames * frame_size], num_frames)
        vad_result = [vad.is_speech(frame.tobytes(), sample_rate) for frame in frames]
    
    # Convert frame decisions to time segments
    timestamps = []
    start = None
    
    # Time multiplier depends on whether we're using sliding window
    time_multiplier = (frame_duration_ms / 2 / 1000) if sliding_window else (frame_duration_ms / 1000)
    
    for i, is_speech in enumerate(vad_result):
        timestamp = i * time_multiplier  # Convert to seconds
        
        if is_speech and start is None:
            start = timestamp  # Speech started
        elif not is_speech and start is not None:
            timestamps.append((start, timestamp))  # Speech ended
            start = None
    
    # Include final segment if we ended in speech
    if start is not None:
        timestamps.append((start, len(audio) / sample_rate))
    
    return timestamps

def compute_mfccs(audio, sr, n_mfcc=13, frame_length=512, hop_length=256):
    """
    Compute MFCC features for audio
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    sr : int
        Sample rate
    n_mfcc : int
        Number of MFCC coefficients
    frame_length : int
        Frame length for FFT
    hop_length : int
        Hop length between frames
        
    Returns:
    --------
    numpy.ndarray
        MFCC features (n_mfcc x n_frames)
    """
    # Pre-emphasis to enhance high frequencies
    preemphasis = 0.97
    emphasized_audio = np.append(audio[0], audio[1:] - preemphasis * audio[:-1])
    
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(
        y=emphasized_audio, 
        sr=sr, 
        n_mfcc=n_mfcc,
        n_fft=frame_length,
        hop_length=hop_length
    )
    
    # Add delta and delta-delta features for better speaker discrimination
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Combine all features
    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    
    return features

def detect_speaker_changes_bic(audio, sr, window_size=2.0, step_size=0.5, threshold=1250):
    """
    Detect speaker changes using Bayesian Information Criterion (BIC)
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    sr : int
        Sample rate
    window_size : float
        Analysis window size in seconds
    step_size : float
        Step size between windows in seconds
    threshold : float
        BIC threshold for change detection
        
    Returns:
    --------
    List of change points in seconds
    """
    # Convert times to samples
    window_samples = int(window_size * sr)
    step_samples = int(step_size * sr)
    
    # Compute MFCCs for the entire audio
    mfccs = compute_mfccs(audio, sr)
    
    # Frame times in seconds
    frame_times = librosa.frames_to_time(
        np.arange(mfccs.shape[1]), 
        sr=sr, 
        hop_length=256
    )
    
    change_points = []
    
    # Slide through audio computing BIC
    for start_idx in range(0, mfccs.shape[1] - int(window_size * sr / 256), int(step_size * sr / 256)):
        end_idx = start_idx + int(window_size * sr / 256)
        
        # Get the window's MFCC features
        window_features = mfccs[:, start_idx:end_idx]
        
        # Skip if window is too small
        if window_features.shape[1] < 20:
            continue
        
        # Find the optimal split point within this window
        middle = window_features.shape[1] // 2
        
        # Calculate BIC for different possible split points
        max_bic = -np.inf
        best_split = middle
        
        # Try different split points around the middle of the window
        for split in range(middle - 10, middle + 10):
            if split <= 0 or split >= window_features.shape[1]:
                continue
                
            # Get the two segments
            segment1 = window_features[:, :split].T  # Transpose for easier covariance calculation
            segment2 = window_features[:, split:].T
            
            # Only compute BIC if we have enough frames
            if segment1.shape[0] < 5 or segment2.shape[0] < 5:
                continue
                
            # Compute full segment statistics
            full_cov = np.cov(window_features.T, rowvar=False)
            
            # Compute individual segment statistics
            cov1 = np.cov(segment1, rowvar=False)
            cov2 = np.cov(segment2, rowvar=False)
            
            # Handle potential singular matrices
            try:
                # Number of features
                d = window_features.shape[0]
                
                # Number of frames in each segment
                n1 = segment1.shape[0]
                n2 = segment2.shape[0]
                n = n1 + n2
                
                # BIC calculation
                bic = 0.5 * (n * np.log(np.linalg.det(full_cov)) - 
                             n1 * np.log(np.linalg.det(cov1)) - 
                             n2 * np.log(np.linalg.det(cov2))) - 0.5 * 0.5 * d * (d+1) * np.log(n)
                
                if bic > max_bic:
                    max_bic = bic
                    best_split = split
            except:
                # Skip in case of singular matrices
                continue
        
        # If BIC exceeds threshold, we found a change point
        if max_bic > threshold:
            # Convert frame index to time
            change_time = frame_times[start_idx + best_split]
            change_points.append(change_time)
    
    return change_points

def detect_energy_shifts(audio, sample_rate, window_size=1024, adaptive=False, frame_length=2048):
    """
    Detect significant energy shifts in audio that could indicate speaker changes
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    sample_rate : int
        Sample rate of the audio
    window_size : int
        Window size for energy computation
    adaptive : bool
        Whether to use adaptive thresholding
    frame_length : int
        Frame length for spectrogram (only used if adaptive=True)
        
    Returns:
    --------
    List of boundary points in seconds
    """
    # Compute energy
    energy = np.array([
        np.sum(np.abs(audio[i:i+window_size])**2) 
        for i in range(0, len(audio) - window_size, window_size//2)
    ])
    
    # Apply smoothing to reduce noise
    energy_smooth = signal.medfilt(energy, kernel_size=5)
    
    if adaptive:
        # Use adaptive thresholding based on local statistics
        segment_boundaries = []
        
        # Compute rolling statistics
        window_length = 20  # Number of frames to consider
        
        for i in range(window_length, len(energy_smooth) - window_length):
            # Get local window
            local_window = energy_smooth[i-window_length:i+window_length]
            
            # Compute local statistics
            local_mean = np.mean(local_window)
            local_std = np.std(local_window)
            
            # Set adaptive threshold
            threshold = local_mean + 2.5 * local_std
            
            # Detect sudden changes
            if energy_smooth[i] > threshold and energy_smooth[i] > energy_smooth[i-1] * 1.5:
                segment_boundaries.append((i * window_size//2) / sample_rate)
    else:
        # Use fixed threshold based on global statistics
        mean_energy = np.mean(energy_smooth)
        threshold = mean_energy * 2.0  # Adjust this multiplier as needed
        
        # Detect changes where energy difference exceeds threshold
        segment_boundaries = []
        
        for i in range(1, len(energy_smooth)):
            if abs(energy_smooth[i] - energy_smooth[i-1]) > threshold:
                segment_boundaries.append((i * window_size//2) / sample_rate)
    
    return segment_boundaries

def merge_speaker_changes(vad_segments, change_points, min_duration=1.0):
    """
    Merge VAD segments with speaker change points to create final segments
    
    Parameters:
    -----------
    vad_segments : list
        List of (start, end) tuples from VAD
    change_points : list
        List of speaker change times in seconds
    min_duration : float
        Minimum segment duration in seconds
        
    Returns:
    --------
    List of (start, end) tuples for final segments
    """
    # Initialize with empty segments list
    final_segments = []
    
    # Process each VAD segment
    for start, end in vad_segments:
        # Find change points within this segment
        segment_changes = [cp for cp in change_points if start < cp < end]
        
        # If no changes, keep the segment as is
        if not segment_changes:
            final_segments.append((start, end))
            continue
        
        # Add the initial segment
        segment_start = start
        for change in sorted(segment_changes):
            # Only add if segment is long enough
            if change - segment_start >= 0.2:  # Minimum 200ms
                final_segments.append((segment_start, change))
            segment_start = change
            
        # Add the final piece if needed
        if end - segment_start >= 0.2:
            final_segments.append((segment_start, end))
    
    # Merge short segments
    merged_segments = []
    
    if not final_segments:
        return []
        
    prev_start, prev_end = final_segments[0]
    
    for start, end in final_segments[1:]:
        # If the current segment starts close to where the previous ended
        # or the previous segment is too short, merge them
        if (start - prev_end < 0.3) or (prev_end - prev_start < min_duration):
            prev_end = end  # Extend previous segment
        else:
            # Add the previous segment and start a new one
            merged_segments.append((prev_start, prev_end))
            prev_start, prev_end = start, end
    
    # Add the last segment
    merged_segments.append((prev_start, prev_end))
    
    # One more pass to ensure minimum duration
    final_merged = []
    
    if not merged_segments:
        return []
        
    prev_start, prev_end = merged_segments[0]
    
    for start, end in merged_segments[1:]:
        if prev_end - prev_start < min_duration:
            # Merge with next segment if current is too short
            prev_end = end
        else:
            final_merged.append((prev_start, prev_end))
            prev_start, prev_end = start, end
    
    # Add the last segment
    final_merged.append((prev_start, prev_end))
    
    return final_merged

def save_segments(audio, sr, segments, output_dir):
    """
    Save audio segments to files
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    sr : int
        Sample rate
    segments : list
        List of (start, end) tuples in seconds
    output_dir : str
        Directory to save segments
    """
    # Ensure output directory exists
    ensure_dir_exists(output_dir)
    
    # Save each segment
    for i, (start, end) in enumerate(segments):
        # Convert times to samples
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        
        # Extract segment
        segment = audio[start_sample:end_sample]
        
        # Skip if segment is too short
        if len(segment) < sr * 0.1:  # Skip segments shorter than 100ms
            continue
            
        # Save as WAV file
        output_file = os.path.join(output_dir, f"segment_{i+1}.wav")
        wav.write(output_file, sr, (segment * 32767).astype(np.int16))
        print(f"Saved {output_file} ({round(end - start, 2)} sec)")

def plot_segmentation(audio, sr, segments, title="Speaker Segmentation"):
    """
    Plot the audio waveform with segment boundaries
    
    Parameters:
    -----------
    audio : numpy.ndarray
        Audio signal
    sr : int
        Sample rate
    segments : list
        List of (start, end) tuples in seconds
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot waveform
    times = np.arange(len(audio)) / sr
    plt.plot(times, audio, color='black', alpha=0.5)
    
    # Plot segment boundaries with alternating colors
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (start, end) in enumerate(segments):
        color = colors[i % len(colors)]
        plt.axvspan(start, end, color=color, alpha=0.3)
        
        # Add segment number
        plt.text((start + end) / 2, 0, str(i+1), 
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='black',
                 fontweight='bold')
    
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(os.path.dirname(output_dir), "segmentation_plot.png"))
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set input/output paths
    audio_file = args.input
    output_dir = args.output_dir
    
    # Load audio file
    try:
        y, sr, audio_file = load_audio(audio_file)
        print(f"Loaded audio: {audio_file}, {len(y)/sr:.2f} seconds, {sr} Hz")
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir = ensure_dir_exists(output_dir)
    
    print(f"Processing audio using VAD mode {args.vad_mode}...")
    
    # Step 1: Get speech segments with VAD
    vad_segments = get_speech_segments_vad(
        y, sr, 
        vad_mode=args.vad_mode,
        frame_duration_ms=args.frame_duration_ms,
        sliding_window=args.sliding_window
    )
    
    print(f"Found {len(vad_segments)} speech segments using VAD")
    
    # Step 2: Detect speaker changes
    if args.use_bic:
        print("Detecting speaker changes using BIC...")
        speaker_changes = detect_speaker_changes_bic(y, sr)
    else:
        print("Detecting speaker changes using energy shifts...")
        speaker_changes = detect_energy_shifts(
            y, sr, 
            adaptive=args.adaptive_threshold
        )
    
    print(f"Detected {len(speaker_changes)} potential speaker changes")
    
    # Step 3: Merge VAD segments with speaker changes
    final_segments = merge_speaker_changes(
        vad_segments, 
        speaker_changes,
        min_duration=args.min_duration
    )
    
    print(f"Created {len(final_segments)} final segments")
    
    # Step 4: Save segmented audio
    save_segments(y, sr, final_segments, output_dir)
    
    # Step 5: Plot results if requested
    if args.plot:
        plot_segmentation(y, sr, final_segments)
    
    print(f"Segmentation complete. Segments saved to: {output_dir}")
