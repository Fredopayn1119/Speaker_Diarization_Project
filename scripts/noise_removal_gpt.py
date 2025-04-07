import subprocess
import os
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import argparse

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False


def denoise_audio(input_path, output_path, method="rnnoise", plot=False, 
                  sensitivity=0.5, stationary=False):
    """
    Denoise audio using various algorithms
    
    Parameters:
    -----------
    input_path : str
        Path to input audio file
    output_path : str
        Path to save denoised audio file
    method : str
        Algorithm to use: 'rnnoise', 'spectral', or 'native'
    plot : bool
        Whether to plot before/after waveforms
    sensitivity : float
        Noise reduction sensitivity (0.0-1.0)
    stationary : bool
        Whether to assume noise is stationary (consistent)
    """
    print(f"Denoising audio using {method} method...")
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Load the audio file
    try:
        if method == "native":
            # Load audio with librosa for native processing
            audio, sr = librosa.load(input_path, sr=None)
        else:
            # We'll use the file path directly for external tools
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return False
    
    # Apply denoising based on selected method
    if method == "rnnoise":
        # RNNoise external command
        try:
            cmd = ['denoise', input_path, output_path]
            if plot:
                cmd.append('--plot')
            
            process = subprocess.run(cmd, 
                                    capture_output=True, 
                                    text=True, 
                                    check=True)
            print(process.stdout)
            if process.stderr:
                print(f"Warning: {process.stderr}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error running RNNoise: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
            return False
        except FileNotFoundError:
            print("RNNoise command not found. Please install pyrnnoise")
            print("You can install with: pip install pyrnnoise")
            return False
            
    elif method == "spectral":
        # Use SoX for spectral noise reduction
        try:
            noise_profile = os.path.join(os.path.dirname(output_path), "noise_profile.prof")
            
            # Extract noise profile from the beginning of the file (adjust as needed)
            subprocess.run([
                'sox', input_path, '-n', 'trim', '0', '0.5', 'noiseprof', noise_profile
            ], check=True)
            
            # Apply noise reduction
            noisered_amount = 0.21 + sensitivity * 0.3  # Scale to reasonable values
            subprocess.run([
                'sox', input_path, output_path, 'noisered', 
                noise_profile, str(noisered_amount)
            ], check=True)
            
            # Clean up temporary file
            if os.path.exists(noise_profile):
                os.remove(noise_profile)
                
        except subprocess.CalledProcessError as e:
            print(f"Error running SoX: {e}")
            return False
        except FileNotFoundError:
            print("SoX command not found. Please install SoX")
            print("macOS: brew install sox, Ubuntu: apt install sox")
            return False
            
    elif method == "native":
        # Pure Python implementation using noisereduce library
        if not NOISEREDUCE_AVAILABLE:
            print("The noisereduce library is not installed.")
            print("Install with: pip install noisereduce")
            return False
            
        try:
            # Estimate noise from a portion of the signal (adjust as needed)
            noise_sample = audio[:int(sr * 0.5)]
            
            # Apply noise reduction
            reduced_noise = nr.reduce_noise(
                y=audio, 
                sr=sr,
                y_noise=noise_sample,
                prop_decrease=sensitivity,
                stationary=stationary
            )
            
            # Save the output file
            wavfile.write(output_path, sr, (reduced_noise * 32767).astype(np.int16))
            
            if plot:
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                plt.title("Original Audio")
                librosa.display.waveshow(audio, sr=sr)
                plt.subplot(2, 1, 2)
                plt.title("Denoised Audio")
                librosa.display.waveshow(reduced_noise, sr=sr)
                plt.tight_layout()
                plt.savefig(f"{os.path.splitext(output_path)[0]}_comparison.png")
                plt.show()
                
        except Exception as e:
            print(f"Error in native denoising: {e}")
            return False
    else:
        print(f"Unknown denoising method: {method}")
        return False
        
    print(f"Denoising completed. Output saved to {output_path}")
    return True


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Audio noise removal utility")
    parser.add_argument("--input", default="../audio_files/0638.wav", 
                        help="Input audio file path")
    parser.add_argument("--output", default="../audio_files/denoised.wav",
                        help="Output denoised file path")
    parser.add_argument("--method", default="rnnoise", 
                        choices=["rnnoise", "spectral", "native"],
                        help="Denoising method to use")
    parser.add_argument("--plot", action="store_true",
                        help="Plot before/after waveforms")
    parser.add_argument("--sensitivity", type=float, default=0.5,
                        help="Noise reduction sensitivity (0.0-1.0)")
    parser.add_argument("--stationary", action="store_true",
                        help="Assume noise is stationary (for native method)")
    
    args = parser.parse_args()
    
    # Run denoising
    denoise_audio(
        args.input, 
        args.output, 
        method=args.method,
        plot=args.plot,
        sensitivity=args.sensitivity,
        stationary=args.stationary
    )