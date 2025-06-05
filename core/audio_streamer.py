import asyncio
import pyaudio
from google import genai

# Constants
MODEL_NAME = "gemini-2.5-flash-preview-native-audio-dialog"
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000 # Native audio output is 24kHz
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK_SIZE = 1024 # Samples per chunk

# Gemini Client Initialization
# The prompt mentions "client = genai.Client()".
# Ensure GOOGLE_GENAI_API_KEY is set in the environment for this to work.
client = genai.Client()

# Live Session Configuration
LIVE_CONNECT_CONFIG = genai.types.LiveConnectConfig(
    response_modalities=["AUDIO"], # Essential for native audio output
    input_audio_transcription=True,
    output_audio_transcription=True,
    # model_config=genai.types.ModelConfig( # Optional: if you need specific model settings
    #     proactivity=genai.types.Proactivity(proactive_audio=True) # Example: if proactive audio is desired
    # )
)

class AudioStreamer:
    def __init__(self):
        self.audio_interface = pyaudio.PyAudio()
        self.audio_in_queue = asyncio.Queue()  # Audio from Gemini to speaker
        self.audio_out_queue = asyncio.Queue() # Audio from mic to Gemini (used by send_realtime_audio)
        self.transcript_queue = asyncio.Queue() # Transcriptions for UI
        self.session = None