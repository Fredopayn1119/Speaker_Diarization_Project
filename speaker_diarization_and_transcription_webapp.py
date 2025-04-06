

import streamlit as st

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

        #return_value is set to a dummy value for demstration purposes - replace the dummy value with the string list return by the fulll diarization function, which should be in the same format as the dummy value
        return_value = ["[Person1]: Hey there.","\n","[Person2]: How are you?","\n","[Person1]: I'm good."] #speaker_diarization_pipeline("audio_file.wav")
        new_file = open("generated_transcript.txt",'w')
        for i in return_value:
            new_file.write(i)

        new_file.close()

        with open("generated_transcript.txt",'r') as file:
            transcript = file.read()
        
        st.subheader("Transcript")
        st.text_area("Diarized Transcript", transcript, height=300)


        