import subprocess

# Define input and output file paths
input_audio = "../audio_files/0638.wav"
output_audio = "../audio_files/denoised.wav"

# Run the pyrnnoise denoise command via subprocess
subprocess.run(['denoise', input_audio, output_audio, '--plot'])

print("Denoising completed.")


# Algorithm	| Strengths	| Weaknesses
# MMSE-STSA	| Good for steady noise	| Less effective for varying noise
# RNNoise (Deep Learning) |	Great for real-world noise |	Needs GPU for best performance
# Spectral Gating |	Simple & fast |	Can leave slight artifacts