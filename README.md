# Jarvis — Voice-Activated AI Assistant

Un assistente vocale intelligente che combina wake-word detection, trascrizione speech-to-text offline e sintesi vocale. Il sistema ascolta continuamente in attesa della parola d'attivazione, registra il comando vocale, lo trascrive con Whisper e risponde a voce.

---

## Indice

- [Architettura del sistema](#architettura-del-sistema)
- [Stack tecnologico](#stack-tecnologico)
- [Pipeline audio](#pipeline-audio)
- [Struttura del progetto](#struttura-del-progetto)
- [Installazione](#installazione)
- [Configurazione](#configurazione)
- [Avvio](#avvio)
- [Dettagli implementativi](#dettagli-implementativi)
- [Note di sicurezza](#note-di-sicurezza)

---

## Architettura del sistema

Il sistema è strutturato in tre stadi sequenziali che si alternano ciclicamente:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CICLO PRINCIPALE                           │
│                                                                     │
│  ┌──────────────┐    wake word     ┌──────────────┐                │
│  │  FASE 1      │ ─────────────── ▶│  FASE 2      │                │
│  │  Ascolto     │                  │  Registrazione│               │
│  │  continuo    │◀─────────────── │  vocale       │               │
│  │  (PyAudio +  │   riprende       │  (sounddevice │               │
│  │  Porcupine)  │   ascolto        │  + VAD)       │               │
│  └──────────────┘                  └──────┬───────┘                │
│                                           │ .wav                    │
│                                           ▼                         │
│                                    ┌──────────────┐                │
│                                    │  FASE 3      │                │
│                                    │  Trascrizione │               │
│                                    │  (Whisper    │                │
│                                    │  large-v3)   │                │
│                                    └──────┬───────┘                │
│                                           │ testo                  │
│                                           ▼                         │
│                                    ┌──────────────┐                │
│                                    │  OUTPUT      │                │
│                                    │  Console +   │                │
│                                    │  TTS (pyttsx3│                │
│                                    └──────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

**Motivo del cambio tra PyAudio e sounddevice:** Porcupine richiede uno stream PCM int16 a livello di frame (`frame_length` fisso), mentre la registrazione post-wake-word è più efficiente con sounddevice che lavora su blocchi float32 e supporta natively il rilevamento del silenzio via RMS. I due stream non possono coesistere, quindi vengono aperti/chiusi alternativamente.

---

## Stack tecnologico

### Speech-to-Text — `faster-whisper`

| Parametro | Valore | Motivazione |
|-----------|--------|-------------|
| Modello | `large-v3` | Massima accuratezza multilingua |
| Device | `cpu` | Compatibilità universale (no GPU richiesta) |
| Compute type | `int8` | Quantizzazione 8-bit: ~4x meno RAM, latenza ridotta |
| Beam size | `5` | Bilanciamento accuratezza/velocità nel beam search |

`faster-whisper` è un reimplementazione di OpenAI Whisper costruita su **CTranslate2**, un engine di inferenza ottimizzato per CPU che applica:
- Quantizzazione dinamica INT8
- Cache del modello su disco dopo il primo download (~3 GB)
- Inferenza locale, nessuna chiamata a API esterne

### Wake-Word Detection — `pvporcupine`

Picovoice Porcupine è un motore di keyword spotting **on-device** basato su una rete neurale leggera ottimizzata per l'esecuzione in tempo reale su CPU. Caratteristiche:
- Latenza < 1ms per frame
- Sample rate fisso: **16.000 Hz**
- Frame length fisso (determinato da `porcupine.frame_length`)
- Formato audio: **int16 PCM mono**
- Verifica della licenza via API key Picovoice (richiede connessione internet all'avvio)

Wake word configurata: `"picovoice"`

### Text-to-Speech — `pyttsx3`

Engine TTS locale che si interfaccia con i sintetizzatori vocali del sistema operativo (SAPI5 su Windows). Configurazione:
- Voce italiana selezionata automaticamente tramite ricerca nell'ID della voce
- Speech rate: **150 parole/minuto**
- Nessuna dipendenza cloud

### Acquisizione audio

| Libreria | Fase | Formato |
|----------|------|---------|
| `pyaudio` | Wake-word detection | int16 PCM, 16kHz, mono |
| `sounddevice` | Registrazione comando | float32, 16kHz, mono |
| `scipy.io.wavfile` | Salvataggio su disco | int16 PCM WAV |
| `pydub` | Conversione MP3 → WAV | Resampling a 16kHz |

---

## Pipeline audio

### Fase 1 — Wake-word detection (continua)

```
Microfono
    │
    ▼ (PyAudio stream, frame_length campioni per volta)
int16 PCM frames
    │
    ▼ struct.unpack('h' * frame_length, buffer)
Lista di interi int16
    │
    ▼ porcupine.process(pcm_frame)
keyword_index (int)
    │
    ├── -1  → nessun match, continua ad ascoltare
    └──  0  → "picovoice" rilevato → Fase 2
```

### Fase 2 — Registrazione con VAD (Voice Activity Detection)

La registrazione avviene in chunk da **0.5 secondi** con rilevamento del silenzio tramite **RMS (Root Mean Square)**:

```
RMS = sqrt(mean(samples²))
```

- **Silenzio:** RMS < 0.01
- **Stop:** 3 chunk consecutivi silenziosi (1.5 secondi di silenzio)
- **Timeout massimo:** 10 secondi di registrazione totale
- **Output:** array float32 concatenato

```python
# Pseudocodice del loop di registrazione
chunks = []
silence_count = 0

while True:
    chunk = sounddevice.rec(chunk_samples, samplerate=16000, dtype='float32')
    rms = numpy.sqrt(numpy.mean(chunk**2))

    if rms < SILENCE_THRESHOLD:
        silence_count += 1
    else:
        silence_count = 0
        chunks.append(chunk)

    if silence_count >= 3 or total_duration >= MAX_DURATION:
        break
```

### Fase 3 — Trascrizione Whisper

```
float32 array
    │
    ▼ numpy.int16 conversion: (audio * 32767).astype(np.int16)
int16 array
    │
    ▼ scipy.io.wavfile.write("temp_jarvis.wav", 16000, data)
File WAV temporaneo
    │
    ▼ model.transcribe(wav_path, beam_size=5)
Generator di segmenti
    │
    ▼ " ".join(segment.text for segment in segments)
Stringa testo finale
```

---

## Struttura del progetto

```
ownprj_FasterWhisepr_AIaudio/
├── main.py              # Applicazione principale — classe JarvisAssistant
├── ipants1.mp3          # Audio di test per la funzione convert_mp3_to_wav()
├── jensen1.mp3          # Audio di test per la funzione convert_mp3_to_wav()
├── .gitignore
└── README.md
```

### Classe `JarvisAssistant` — metodi principali

| Metodo | Responsabilità |
|--------|---------------|
| `__init__()` | Carica modello Whisper, inizializza TTS, apre stream Porcupine |
| `_open_porcupine_stream()` | Crea PyAudio stream con i parametri richiesti da Porcupine |
| `_set_italian_voice()` | Seleziona la voce italiana nell'engine pyttsx3 |
| `speak(text)` | Sintetizza e riproduce testo via TTS |
| `_check_wake_word()` | Legge un frame e ritorna True se la wake word è rilevata |
| `_record_until_silence()` | Registra audio con VAD, ritorna array float32 |
| `_transcribe_wav(path)` | Trascrive un file WAV con faster-whisper |
| `convert_mp3_to_wav(mp3, wav)` | Converte MP3 in WAV a 16kHz (utility) |
| `run()` | Loop principale dell'applicazione |
| `_cleanup()` | Chiude stream, engine e modelli in modo ordinato |

---

## Installazione

### Prerequisiti

- Python 3.10 o superiore
- Microfono funzionante
- ~3 GB di spazio libero per il modello Whisper (scaricato automaticamente al primo avvio)
- Connessione internet (solo al primo avvio, per verifica API key Picovoice e download modello)

### Dipendenze

```bash
pip install faster-whisper pvporcupine pyttsx3 sounddevice pyaudio scipy numpy pydub
```

**Nota su PyAudio su Windows:** se l'installazione fallisce, scarica il wheel precompilato da [Unofficial Python Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) e installalo con `pip install <file>.whl`.

---

## Configurazione

Le costanti principali si trovano all'inizio di `main.py`:

```python
ACCESS_KEY = "..."      # API key Picovoice (obbligatoria)
WAKE_WORD = "picovoice" # Parola di attivazione
SAMPLE_RATE = 16000     # Hz — non modificare (richiesto da Porcupine e Whisper)
FOLDER_PRJ = "..."      # Path della cartella di progetto
```

**Per cambiare la wake word:** Porcupine supporta un set predefinito di keyword (es. `"hey google"`, `"alexa"`, `"jarvis"`, ecc.) nella versione gratuita. Keyword personalizzate richiedono un piano a pagamento Picovoice.

**Per cambiare il modello Whisper:** sostituire `"large-v3"` con `"medium"`, `"small"`, `"base"` o `"tiny"` per bilanciare accuratezza e velocità di trascrizione.

---

## Avvio

```bash
python main.py
```

Output atteso all'avvio:

```
Caricamento modello Whisper (prima volta può richiedere qualche minuto)...
Modello caricato.

Jarvis pronto — di' 'picovoice' per attivarlo  |  Ctrl+C per uscire
```

**Utilizzo:**
1. Pronuncia `"picovoice"` — Jarvis risponde _"Al tuo servizio Sir"_
2. Parla il tuo comando
3. Fai una pausa di almeno 1.5 secondi per fermare la registrazione
4. La trascrizione appare in console
5. Ripeti dal punto 1, oppure premi `Ctrl+C` per uscire

---

## Dettagli implementativi

### Perché int8 e non float16?

Su CPU, float16 non porta vantaggi rispetto a float32 poiché la maggior parte delle CPU x86 non ha istruzioni native float16. La quantizzazione int8 invece sfrutta le istruzioni SIMD (AVX2/AVX-512) per operazioni intere, risultando in inferenza 2-4x più veloce con degradazione minima dell'accuratezza.

### Conflitto KMP_DUPLICATE_LIB_OK

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

NumPy e CTranslate2 (usato da faster-whisper) possono includere versioni diverse della libreria runtime OpenMP (`libiomp5md.dll` su Windows). Senza questa variabile, il processo termina con errore. La variabile sopprime il check ma non risolve il conflitto strutturale — è una soluzione pragmatica comune in ambienti conda/venv su Windows.

### Conversione float32 → int16

```python
audio_int16 = (audio_float32 * 32767).astype(numpy.int16)
```

I campioni audio registrati da sounddevice sono float32 normalizzati nell'intervallo `[-1.0, 1.0]`. Il formato WAV standard richiede int16 (range `[-32768, 32767]`), quindi si scala moltiplicando per `2^15 - 1 = 32767`.

### Gestione delle risorse

Il metodo `_cleanup()` viene chiamato sia su `KeyboardInterrupt` che su eccezioni non gestite, garantendo che:
- Lo stream PyAudio venga chiuso e il processo pyaudio terminato
- Il motore pyttsx3 venga fermato
- L'istanza Porcupine venga deallocata (rilascia la licenza)

---

## Note di sicurezza

- **API key esposta:** la chiave Picovoice in `main.py` è visibile nel codice sorgente. Per progetti condivisi, spostarla in una variabile d'ambiente (`os.environ["PICOVOICE_KEY"]`) o in un file `.env` (non committato).
- **Path hardcoded:** `FOLDER_PRJ` contiene un percorso assoluto Windows-specifico. Usare `pathlib.Path(__file__).parent` per renderlo portabile.
- **File WAV temporaneo:** `temp_jarvis.wav` viene sovrascritto ad ogni comando e non eliminato alla chiusura. Aggiungere una pulizia esplicita in `_cleanup()` se la privacy del contenuto audio è rilevante.
