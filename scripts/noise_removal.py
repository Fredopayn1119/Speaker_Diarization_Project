import subprocess
import os
import shutil

def noise_removal_native(input_audio, output_audio):
    """
    Uses 'denoise' command to clean up audio.
    """
    try:
        subprocess.run(['denoise', input_audio, output_audio, '--plot'], check=True)
        print("Denoising completed.")

    except subprocess.CalledProcessError as e:
        print(f"Error during denoising: {e}")

if __name__ == "__main__":
    noise_removal_native("../audio/freddy_ritika.wav", "../audio/denoised.wav")


# RNNoise: Great for real-world noise, needs GPU for best performance
