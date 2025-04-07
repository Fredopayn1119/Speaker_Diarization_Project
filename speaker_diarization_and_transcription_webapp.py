import streamlit as st
import os
import tempfile
import json
from diarization_pipeline import noise_removal, segment_audio, extract_features, cluster_speakers, transcribe_audio, ensure_directory_exists

st.set_page_config(page_title="Speaker Diarization and Transcription Web App", layout="centered")

st.markdown("""
    <style>
    .main {
        background-image: linear-gradient(to bottom right, #38ebd95f, #9d38eb5f);
    }

    .stFileUploader > label div[role='button'] {
        background-color: #1f77b4;
        color: white;
        padding: 0.5em 1em;
        border-radius: 6px;
        border: none;
        text-align: center;
        transition: background-color 0.3s ease;
    }
   
    .stButton button {
        background-color: #1f78b4;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
    }

    .stButton button:hover {
        background-color:  white;
        color: #3bd440;
        border-color: #3bd440;
        cursor: pointer
    }

    .stTextArea textarea {
        background-color: #eaf1f8;
        color: #333;
    }
    .text {
        color: black;
    }
    </style>
""", unsafe_allow_html=True)


st.title("Speaker Diarization and Transcription")
st.markdown("Upload an audio file and click the button to generate a transcript.")

audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if audio:
    st.audio(audio, format='audio/wav')

if audio and st.button("Generate Transcript"):
    with st.spinner("Running speaker diarization..."):
        # Create a temporary directory to store processing files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file to the temporary directory
            audio_path = os.path.join(temp_dir, "input_audio.wav")
            with open(audio_path, "wb") as f:
                f.write(audio.getbuffer())
            
            # Set up output directory
            output_dir = os.path.join(temp_dir, "output")
            ensure_directory_exists(output_dir)
            
            # Run the diarization pipeline
            try:
                # Step 1: Noise Removal
                st.text("Removing background noise...")
                denoised_file = os.path.join(output_dir, "denoised.wav")
                cleaned_audio = noise_removal(audio_path, denoised_file, False)
                
                # Step 2: Audio Segmentation
                st.text("Segmenting audio...")
                segment_dir = segment_audio(cleaned_audio, output_dir, False)
                
                # Step 3: Feature Extraction
                st.text("Extracting speaker features...")
                embeddings = extract_features(segment_dir, output_dir, "dvector", False)
                
                # Step 4: Speaker Clustering
                st.text("Clustering speakers...")
                diarization_result = cluster_speakers(embeddings, output_dir, "ahc", None, False)
                
                # Step 5: Transcription
                st.text("Transcribing audio...")
                final_result = transcribe_audio(segment_dir, diarization_result, output_dir, "tiny")
                
                # Process the results into a readable transcript
                return_value = []
                
                # Load the full result from the JSON file
                result_path = os.path.join(output_dir, "full_result.json")
                with open(result_path, 'r') as f:
                    result_data = json.load(f)
                
                # Extract speaker segments with transcription
                for speaker_id, segments in result_data["speaker_segments"].items():
                    speaker_name = f"Speaker {int(speaker_id) + 1}"
                    for segment in segments:
                        if "transcription" in segment and segment["transcription"].strip():
                            return_value.append(f"[{speaker_name}]: {segment['transcription']}")
                            return_value.append("\n")
                
                if not return_value:
                    return_value = ["No transcribable speech detected in the audio."]
            
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                return_value = [f"Error: {str(e)}"]
        
        # Write results to file
        new_file = open("generated_transcript.txt", 'w')
        for i in return_value:
            new_file.write(i)
        new_file.close()

        with open("generated_transcript.txt", 'r') as file:
            transcript = file.read()
        
        st.subheader("Transcript")
        st.text_area("Diarized Transcript", transcript, height=300)


