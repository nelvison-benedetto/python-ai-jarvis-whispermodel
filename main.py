import ctranslate2  # x faster_whisper
import sounddevice  #registra audio
import numpy
import time
from pynput import keyboard   #input keyboard
from scipy.io.wavfile import write   #create .wav files
import tempfile  #creazione file temporanei
import os
from faster_whisper import WhisperModel  #WHISPER(chatgpt openai) IS NOT REAL TIME TRASCRIPTION!! USE CHUNKS AND FILE .WAV TO WORK!
from pydub import AudioSegment   #covert mp3-->wav
import pyttsx3  #speak text

#PICOVOICE
import pvporcupine
import pyaudio
import struct
from tkinter import messagebox


#device = "cuda" if torch.cuda.is_available() else "cpu"  #use cpu to elaborate
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  #ignora conflitti runtime OpenMP, problema con multilingua
folder_prj = "D:\\Python_Projects\\ML_&_OpenCV_Projects\\ownprj_FasterWhisepr_AIaudio"

mp3_file_path = "D:\\Python_Projects\\ML_&_OpenCV_Projects\\ownprj_FasterWhisepr_AIaudio\\ipants1.mp3"
wav_file_path_from_mp3 = "D:\\Python_Projects\\ML_&_OpenCV_Projects\\ownprj_FasterWhisepr_AIaudio\\ipants1_mp.wav"

class WhisperTranscribe:

    def __init__(self, model_size = "large-v3", sample_rate = 44100):
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.model = WhisperModel(model_size, device = "cpu", compute_type="int8")
            #int8_float16 precisione piu bassa rispetto a float16/32
        self.is_recording = False
        self.stop_running = False
    def on_press(self,key):
        if key == keyboard.Key.space:
            if not self.is_recording:
                self.is_recording = True
                print("Start recording...")
        elif key == keyboard.Key.esc:
            self.stop_running = True
            return False

    def on_release(self,key):
        if key  == keyboard.Key.space:
            if self.is_recording:
                self.is_recording = False
                print("End recording...")


    # def record_audio(self):
    #     print("Start function record_audio...")
    #     recording = numpy.array([], dtype="float64").reshape(0, 1)  #create empty array di tipo float64, reshape transform only 1 column
    #     frames_x_buffer = int(self.sample_rate * 0.2)  #in ogni chunk registra 0.2 sec
    #
    #     with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
    #         while True:
    #             if self.is_recording: #se space is pressed, viene creato un nuovo chunk lunghezza = frames_x_buffer
    #                 print("new chunk")
    #                 chunk = sounddevice.rec(frames_x_buffer, samplerate=self.sample_rate, channels=1, dtype="float64")  #channel 1 è audio main, 2 è stereo
    #                    #frames_x_buffer durate del frammento(i.e.0.2sec)
    #                 sounddevice.wait()  #attende la fine del rec del chunk
    #                 recording = numpy.vstack([recording, chunk]) #add the chunk to the existing array recording
    #
    #             if not self.is_recording and len(recording) > 0:
    #                 print("end loop rec")
    #                 break
    #         listener.join()  #fa aspettare codice principale,lo fa ripartire quando ciclo di ascolto tastiera pynput è terminato
    #     print("Recording finished. Total length:", recording.shape[0])  #show dimension of first dimension's array, so the number of rows
    #     return recording
    #
    # def save_temp_audio(self, recording, folder_prj):
    #     print("Start func save_temp_audio...")
    #     recording_int16 = (recording * 32767).astype(numpy.int16)  #trasform array float64 in int16 con formula per salvare in .wav
    #     wav_file_path = os.path.join(folder_prj, "recorded_audio.wav")
    #     write(wav_file_path, self.sample_rate, recording_int16)  #write funct of scipy.io.wavfile, write array recording_int16 in the file of destination
    #     print(f"Audio saved to {wav_file_path}")
    #     return wav_file_path
    #
    # def convert_mp3_to_wav(self, mp3_file_path, wav_file_path_from_mp3):
    #     print("Converting MP3 to WAV...")
    #     audio = AudioSegment.from_mp3(mp3_file_path)
    #     audio = audio.set_frame_rate(self.sample_rate)  # Assicurati che il sample rate sia corretto
    #     audio.export(wav_file_path_from_mp3, format="wav")
    #     print(f"MP3 converted to WAV and saved to {wav_file_path_from_mp3}")

    # def transcribe_audio(self, wav_file_path):  #USE FAST_WHISPER
    #     print("start func transcribe_audio...")
    #     segments, info = self.model.transcribe(wav_file_path, beam_size=5)  #DEFAULT SETS X GITHUB FAST_WHISPER
    #     #print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    #     full_transcription = ""
    #     for segment in segments:
    #         print(segment.text)
    #         full_transcription += segment.text + ""
    #     #os.remove(wav_file_path)  #delete the tempory audio file (generated from recording) used for transcription !!!
    #     print("end funct transcriber_whisper...")
    #     return full_transcription


    def speak_text(self, text):
        engine = pyttsx3.init()  #start engine
        engine.setProperty('rate', 150)  # Velocità della voce
        voices = engine.getProperty('voices')  #set voices
        italian_voice = None
        for voice in voices:
            #print(voice.name)
            if 'italian' in voice.name.lower() or 'italiano' in voice.name.lower():
                italian_voice = voice
                break
        if italian_voice:
            engine.setProperty('voice', italian_voice.id)  # Imposta la voce in italiano
        else:
            print("Nessuna voce italiana trovata")
        # engine.setProperty('voice', voices[0].id)  # Cambia voce se necessario
        engine.say(text)
        engine.runAndWait()


    def record_and_transcribe(self):
        print("Start function record_and_transcribe...")
        frames_x_buffer = int(self.sample_rate * 2)  #2 second chunks
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            while True:
                if self.is_recording:
                    print("Recording new chunk...")
                    chunk = sounddevice.rec(frames_x_buffer, samplerate=self.sample_rate, channels=1, dtype="float64")
                    sounddevice.wait()
                    # Process and transcribe the chunk
                    self.process_and_transcribe_chunk(chunk)  #work trascribe one chuck for one chunk
                if self.stop_running:
                    break
                time.sleep(0.05)  #evita busy-wait a 100% CPU quando non registra
            listener.join()

    def process_and_transcribe_chunk(self, chunk):
        #DON'T USE MEMORY, SLOWER
        recording_int16 = (chunk * 32767).astype(numpy.int16)
        temp_file_path = os.path.join(folder_prj, "temp_chunk.wav")
        write(temp_file_path, self.sample_rate, recording_int16)  #write funct of scipy.io.wavfile, write array recording_int16 in the file of destination

        # Transcribe the chunk
        transcription = self.transcribe_audio(temp_file_path)
        print(transcription)

        # Optionally, delete the temp file
        #os.remove(temp_file_path)
        #'''USE MEMORY, FASTER'''  #DON'T WORK ON WHISPER (OPENAI)!!
        #print("Processing chunk...")
        # transcription = self.transcribe_audio_from_memory(chunk)
        # print(transcription)

    def transcribe_audio_from_memory(self, audio_data):  #DON'T WORK X WHISPER(OPENAI) !!
        # Convert audio data to the format required by the model
        # For example, you might need to convert it to a specific tensor format
        # Here, we assume `self.model` can handle numpy arrays directly
        transcription = self.model.transcribe_from_memory(audio_data, sample_rate=self.sample_rate)
           #function whisper transcribe_from_memory DON'T EXIST X WHISPER(OPENAI)!!
        return transcription


    def run_manager(self):
        print("Hold the spacebar to recording...")
        while not self.stop_running:
            self. record_and_transcribe()

            # recording = self.record_audio()
            # wav_file_path = self.save_temp_audio(recording, folder_prj)
            # fwhisper_transcription = self.transcribe_audio(wav_file_path)
            # print(fwhisper_transcription)
            # print("Press spacebar to start recording again, or esc to exit")



class WakeWordListener:  #PICOVOICE DETECT KEYWORD
    def __init__(self, access_key, keywords=['picovoice', 'bumblebee']):
        self.access_key = access_key
        self.keywords = keywords
        self.porcupine = None  #instance porcupine
        self.pa = None   #instance pyaudio
        self.audio_stream = None   #stream catched
        self.initialize()

    def initialize(self):
        #nizializza Porcupine e PyAudio
        self.porcupine = pvporcupine.create(
            access_key=self.access_key,
            keywords=self.keywords
        )
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self.pa.open(  #open mic (sample rate and frame must match with those of picovoice)
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )

    def get_next_audio_frame(self):
        """Cattura e restituisce il prossimo frame audio dal microfono."""
        pcm = self.audio_stream.read(self.porcupine.frame_length)  #read a frame from mic
        pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)  #convert x pico porcupine
        return pcm

    def listen(self):
        #avvia l'ascolto delle parole chiave
        print("Listening for wake words...")
        try:
            while True:  #listening
                audio_frame = self.get_next_audio_frame()
                keyword_index = self.porcupine.process(audio_frame)
                if keyword_index >= 0:  #se è >0 allora almeno una keyword detected
                    print(f"Detected '{self.keywords[keyword_index]}'")
                    # Puoi inserire qui il codice da eseguire quando una parola chiave è rilevata
                    #messagebox.showinfo("Popup", "Keyword Detected!")
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.cleanup()  #clean instances pyaudio, porcupine, close stream

    def cleanup(self):
        #libera le risorse quando non servono più
        if self.audio_stream is not None:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.pa is not None:
            self.pa.terminate()
        if self.porcupine is not None:
            self.porcupine.delete()


if __name__ == "__main__":
    fwhisper_transcriber = WhisperTranscribe()
    #RECORDING & TRASCRIBE
    fwhisper_transcriber.run_manager()

    #READ A FILE MP3
    # fwhisper_transcriber.convert_mp3_to_wav(mp3_file_path, wav_file_path_from_mp3)
    # fwhisper_transcription_fromMp3 = fwhisper_transcriber.transcribe_audio(wav_file_path_from_mp3)
    # #SPEAK TEXT
    # fwhisper_transcriber.speak_text(fwhisper_transcription_fromMp3)

    #PICOVOICE DETECT KEYWORD TO ACTIVE SOMETHING
    # ACCESS_KEY = 'qrazoc171FxIZT+fqWFnzGOI7lstTvWmDotAFB2hExzckjtM+h+Z3w=='  #ONLINE VERIFICATION WITH CLOUD!
    # listener = WakeWordListener(access_key=ACCESS_KEY, keywords=['picovoice', 'bumblebee'])
    # listener.listen()

#PICO VOICE x jarvis sleep and wake up at keyword