import ctranslate2        # richiesto internamente da faster_whisper
import sounddevice
import numpy
import os
import struct

from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from pydub import AudioSegment
import pyttsx3
import pvporcupine
import pyaudio

'''
  Tu dici "picovoice"                                                                                                                                                                   → Jarvis: "Al tuo servizio Sir"
    -> registra quello che dici                                                                                                                                                      
    -> stampa a schermo la trascrizione                      
    -> fine
'''

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  #evita conflitti runtime OpenMP

FOLDER_PRJ  = "D:\\Python_Projects\\ML_&_OpenCV_Projects\\ownprj_FasterWhisepr_AIaudio"
ACCESS_KEY  = "qrazoc171FxIZT+fqWFnzGOI7lstTvWmDotAFB2hExzckjtM+h+Z3w=="  # verifica online Picovoice
WAKE_WORD   = "picovoice"
SAMPLE_RATE = 16000   # Hz — richiesto da Porcupine, perfect anche per Whisper


class JarvisAssistant:

    def __init__(self, access_key=ACCESS_KEY, model_size="large-v3"):
        self.sample_rate = SAMPLE_RATE

        #Whisper (STT)
        print("Caricamento modello Whisper (prima volta può richiedere qualche minuto)...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Modello caricato.")

        #TTS (pyttsx3 inizializzato una volta sola)
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", 150)
        self._set_italian_voice()

        #Porcupine (wake word)
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=[WAKE_WORD]
        )
        self.pa = pyaudio.PyAudio()
        self.audio_stream = self._open_porcupine_stream()

    #setup

    def _open_porcupine_stream(self):
        return self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )

    def _set_italian_voice(self):
        for voice in self.tts.getProperty("voices"):
            if "italian" in voice.name.lower() or "italiano" in voice.name.lower():
                self.tts.setProperty("voice", voice.id)
                return
        print("Nessuna voce italiana trovata, uso voce di default.")

    #TTS

    def speak(self, text):
        print(f"Jarvis: {text}")
        self.tts.say(text)
        self.tts.runAndWait()

    #STT

    def _transcribe_wav(self, wav_path):
        #trascrive un file .wav con faster-whisper
        segments, _ = self.model.transcribe(wav_path, beam_size=5)
        return " ".join(seg.text for seg in segments).strip()

    def convert_mp3_to_wav(self, mp3_path, wav_path):
        #converte un file MP3 in WAV al sample rate corretto
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(self.sample_rate)
        audio.export(wav_path, format="wav")
        print(f"Convertito: {mp3_path} → {wav_path}")

    #registrazione

    def _record_until_silence(self, max_duration=10.0, silence_threshold=0.01, silence_duration=1.5):
        """
        registra a chunk da 0.5s finché non rileva silenzio per `silence_duration` secondi
        oppure si raggiunge `max_duration`. Return array numpy float32.
        """
        chunk_sec      = 0.5
        chunk_size     = int(self.sample_rate * chunk_sec)
        silence_needed = int(silence_duration / chunk_sec)

        recording    = []
        silent_count = 0

        print("Sto ascoltando... (silenzio per fermarsi)")
        for _ in range(int(max_duration / chunk_sec)):
            chunk = sounddevice.rec(chunk_size, samplerate=self.sample_rate, channels=1, dtype="float32")
            sounddevice.wait()
            recording.append(chunk)

            rms = numpy.sqrt(numpy.mean(chunk ** 2))
            if rms < silence_threshold:
                silent_count += 1
                if silent_count >= silence_needed:
                    break
            else:
                silent_count = 0

        return numpy.vstack(recording)

    #wake word

    def _check_wake_word(self):
        """Legge un frame dal microfono e restituisce True se la wake word è rilevata."""
        pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
        return self.porcupine.process(pcm) >= 0

    #main principale

    def run(self):
        print(f"\nJarvis pronto — di' '{WAKE_WORD}' per attivarlo  |  Ctrl+C per uscire\n")
        try:
            while True:
                if self._check_wake_word():
                    print("[Wake word rilevata]")

                    #stop stream PyAudio per liberare il microfono a sounddevice
                    self.audio_stream.stop_stream()

                    self.speak("Al tuo servizio Sir")

                    audio_data  = self._record_until_silence()
                    audio_int16 = (audio_data * 32767).astype(numpy.int16)
                    temp_path   = os.path.join(FOLDER_PRJ, "temp_jarvis.wav")
                    write(temp_path, self.sample_rate, audio_int16)

                    transcription = self._transcribe_wav(temp_path)

                    if transcription:
                        print(f"Tu: {transcription}")
                    else:
                        print("Non ho capito nulla, riprova.")

                    # Riavvia lo stream Porcupine per il prossimo wake word
                    self.audio_stream.start_stream()
                    print(f"\nIn ascolto per '{WAKE_WORD}'...\n")

        except KeyboardInterrupt:
            print("\nArrivederci.")
        finally:
            self._cleanup()

    #cleanup

    def _cleanup(self):
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.pa:
            self.pa.terminate()
        if self.porcupine:
            self.porcupine.delete()


if __name__ == "__main__":
    jarvis = JarvisAssistant()
    jarvis.run()
