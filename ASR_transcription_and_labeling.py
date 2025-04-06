#ensure that SpeechRecognition is installed
#pip install SpeechRecognition

import speech_recognition as sr

# INPUT:
  # speech_chunks: is assumed to be a variable containing the list of of the .wav files 
  #                names of each speech chunk.
  # cluster_labels: is assumed to be a variable containing the list of labels
  #                corresponsing to each speech chunks in the same order.
# OUTPUT:
  # ordered_transcript: a list containing the tagged transcriptions in the order that 
  # they came in originally.
  #


def ASR_transcription_and_tagging(speech_chunks, cluster_labels):

  recognizer = sr.Recognizer()

  ordered_trascript = []

  f = open('transcript.txt', 'w')

  for filename in speech_chunks:
    
    #if speech_chunks contains the filesnames including their extensions, replace
    #this line with
    #speech_chunk_file = filename
    speech_chunk_file = ""+filename+".wav"

    with sr.AudioFile(speech_chunk_file) as source:
        speech_chunk_speech = recognizer.record(sr.AudioFile(speech_chunk_file))
        try:
            speech_to_text = recognizer.recognize_google(speech_chunk_speech)
        except sr.UnknownValueError:
            speech_to_text = "[[Unknown speech]]"
        except sr.RequestError:
            speech_to_text = "[[Request Error]]"


        file_and_speaker_label_index = speech_chunks.index(filename)

        f.write("["+cluster_labels[file_and_speaker_label_index]+"]: "+speech_to_text)
        f.write("\n")

        ordered_trascript.append("["+cluster_labels[file_and_speaker_label_index]+"]: "+speech_to_text)
        ordered_trascript.append("\n")

  f.close()

  return ordered_trascript