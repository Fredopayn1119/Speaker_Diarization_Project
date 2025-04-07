import subprocess
import os
import shutil

def noise_removal_native(input_audio, output_audio):
    """
    Removes noise from an input audio file using pyrnnoise's 'denoise' command.
    
    Args:
        input_audio (str): Path to the input audio file.
        output_audio (str): Path to save the denoised audio.
        plot_output_path (str, optional): Path to save the plot image.
    """
    try:
        subprocess.run(['denoise', input_audio, output_audio, '--plot'], check=True)
        print("Denoising completed.")

    except subprocess.CalledProcessError as e:
        print(f"Error during denoising: {e}")

if __name__ == "__main__":
    noise_removal_native("../audio/freddy_ritika.wav", "../audio/denoised.wav")

# Algorithm	| Strengths	| Weaknesses
# MMSE-STSA	| Good for steady noise	| Less effective for varying noise
# RNNoise (Deep Learning) |	Great for real-world noise |	Needs GPU for best performance
# Spectral Gating |	Simple & fast |	Can leave slight artifacts