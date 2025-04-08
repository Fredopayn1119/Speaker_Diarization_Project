import subprocess
import sys

def run_webapp():
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", 
                       "app.py"], 
                       check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        print("Make sure streamlit is installed by running: pip install streamlit")
    except KeyboardInterrupt:
        print("Web app stopped.")





if __name__ == "__main__":
    print("Starting Speaker Diarization and Transcription Web App...")
    print("Once the app is running, you can upload audio files (.wav or .mp3)")
    print("and perform speaker diarization with transcription.")
    print("\nThe web interface will open in your browser shortly...")
    run_webapp()