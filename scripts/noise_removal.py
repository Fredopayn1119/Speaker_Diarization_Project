import librosa
import noisereduce as nr
import webrtcvad
import numpy as np
import scipy.io.wavfile as wav
from pdb import set_trace as bp

# Load Audio File
audio_file = "../audio_files/EN2001a.Array1-01.wav"
y, sr = librosa.load(audio_file, sr=None)  # Load audio file

# Convert to int16 PCM for noise reduction
y_int16 = (y * 32767).astype(np.int16)

# Apply Noise Reduction
y_denoised = nr.reduce_noise(y=y_int16, sr=sr, stationary=False, prop_decrease=0.7) # Increase noise reduction strength

# Energy-based Filtering to Preserve Soft Speech
def energy_filter(audio, threshold=0.01):
    energy = np.abs(audio)
    mask = energy > threshold
    return audio[mask]

y_denoised = energy_filter(y_denoised, threshold=0.005)  # Keep low-energy speech

# Voice Activity Detection (VAD)
def vad(audio, sample_rate, frame_duration_ms=30):
    vad = webrtcvad.Vad(1)  # Set aggressiveness mode (0-3)
    
    # Ensure correct sample rate
    if sample_rate not in [8000, 16000, 32000, 48000]:
        raise ValueError("WebRTC VAD only supports 8kHz, 16kHz, 32kHz, or 48kHz.")
    
    
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    
    # Convert audio to PCM 16-bit
    audio_int16 = (audio * 32767).astype(np.int16)
    
    num_frames = len(audio_int16) // frame_size
    frames = np.array_split(audio_int16[:num_frames * frame_size], num_frames)
    vad_result = [vad.is_speech(frame.tobytes(), sample_rate) for frame in frames]

    speech_frames = [frames[i] for i, is_speech in enumerate(vad_result) if is_speech]
    
    return np.concatenate(speech_frames) if speech_frames else np.array([])


# Apply VAD on the denoised audio
y_vad = vad(y_denoised / 32767.0, sr) # Scale back to float (-1 to 1)

# Save the processed output
output_file = "../audio_files/denoised.wav"
wav.write(output_file, sr, (y_vad * 32767).astype(np.int16))

print(f"Processed file saved as {output_file}")
