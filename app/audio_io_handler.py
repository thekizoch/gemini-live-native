import asyncio
import pyaudio

# Audio format constants
CHUNK_SIZE = 1024  # Number of frames per buffer for PyAudio
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS_INPUT = 1  # Number of audio channels for input (mono)
CHANNELS_OUTPUT = 1 # Number of audio channels for output (mono)
INPUT_RATE = 16000  # Sampling rate for input audio (Hz)
OUTPUT_RATE = 24000  # Sampling rate for output audio (Hz) - Gemini native audio output rate

class AudioIOHandler:
    """
    Handles local audio input (microphone) and output (speaker) using PyAudio.
    Designed to be used with asyncio for non-blocking operations.
    """

    def __init__(self, mic_to_gemini_queue: asyncio.Queue, gemini_to_speaker_queue: asyncio.Queue):
        """
        Initializes the AudioIOHandler.

        Args:
            mic_to_gemini_queue: An asyncio.Queue to send microphone audio chunks to.
            gemini_to_speaker_queue: An asyncio.Queue to receive audio chunks from for playback.
        """
        self.mic_to_gemini_queue = mic_to_gemini_queue
        self.gemini_to_speaker_queue = gemini_to_speaker_queue
        
        self._pyaudio_instance = None # Initialized on first use or explicitly
        self._input_stream = None
        self._output_stream = None
        
        self._stop_event = asyncio.Event() # Signals threads to stop
        self._loop = asyncio.get_running_loop() # Capture the loop for threadsafe calls

        self._is_recording_active_flag = False # Internal flag for loop state
        self._is_playback_active_flag = False  # Internal flag for loop state
    
    def _ensure_pyaudio_instance(self):
        if self._pyaudio_instance is None:
            self._pyaudio_instance = pyaudio.PyAudio()

    def _blocking_record_loop(self):
        """
        Synchronous blocking loop for recording audio.
        This method is intended to be run in a separate thread via asyncio.to_thread.
        """
        self._ensure_pyaudio_instance()
        self._is_recording_active_flag = True
        try:
            self._input_stream = self._pyaudio_instance.open(
                format=FORMAT,
                channels=CHANNELS_INPUT,
                rate=INPUT_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            print("AudioIOHandler: Microphone recording stream opened.")
            while not self._stop_event.is_set():
                try:
                    data = self._input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    if not data: # Should not happen with a live stream unless error
                        print("AudioIOHandler: No data read from microphone, ending record loop.")
                        break
                    
                    future = asyncio.run_coroutine_threadsafe(self.mic_to_gemini_queue.put(data), self._loop)
                    future.result() # Wait for the put to complete in the event loop

                except IOError as e:
                    # PyAudio specific error check for input overflow might be needed
                    # For example, e.args[0] might contain paInputOverflowed for some versions/platforms
                    if "Input overflowed" in str(e): # Generic check
                        print("AudioIOHandler: Input overflowed. Skipping frame.")
                        continue
                    elif self._stop_event.is_set():
                        print("AudioIOHandler: Recording loop: Stop event received during IOError.")
                        break
                    print(f"AudioIOHandler: IOError reading from microphone: {e}")
                    break 
                except Exception as e:
                    if self._stop_event.is_set():
                        print("AudioIOHandler: Recording loop: Stop event received during generic Exception.")
                        break
                    print(f"AudioIOHandler: Unexpected error in recording loop: {e}")
                    break
        except Exception as e:
            print(f"AudioIOHandler: Failed to start or maintain microphone recording: {e}")
        finally:
            if self._input_stream:
                try:
                    if self._input_stream.is_active(): # Check if stream is active before stopping
                         self._input_stream.stop_stream()
                    self._input_stream.close()
                except Exception as e_close:
                    print(f"AudioIOHandler: Error closing input stream: {e_close}")
                self._input_stream = None
            self._is_recording_active_flag = False
            print("AudioIOHandler: Microphone recording stream closed and loop finished.")

    async def record_audio(self):
        """
        Starts recording audio from the default microphone.
        Runs the PyAudio blocking calls in a separate thread.
        """
        if self._is_recording_active_flag : # Check internal flag
            print("AudioIOHandler: Recording is already in progress.")
            return
        
        self._stop_event.clear() # Clear stop event before starting a new session
        print("AudioIOHandler: Starting audio recording task...")
        try:
            await asyncio.to_thread(self._blocking_record_loop)
        except Exception as e:
            print(f"AudioIOHandler: Exception from record_audio thread: {e}")
        finally:
            print("AudioIOHandler: Audio recording task awaited and finished.")


    def _blocking_play_loop(self):
        """
        Synchronous blocking loop for playing audio.
        This method is intended to be run in a separate thread via asyncio.to_thread.
        """
        self._ensure_pyaudio_instance()
        self._is_playback_active_flag = True
        try:
            self._output_stream = self._pyaudio_instance.open(
                format=FORMAT,
                channels=CHANNELS_OUTPUT,
                rate=OUTPUT_RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE 
            )
            print("AudioIOHandler: Speaker playback stream opened.")
            while not self._stop_event.is_set():
                audio_chunk = None
                try:
                    # Get data from the asyncio queue from this thread
                    # Use a timeout to periodically check the stop_event
                    # asyncio.wait_for needs to run in the loop, so we submit the coro
                    get_coro = self.gemini_to_speaker_queue.get()
                    timed_get_coro = asyncio.wait_for(get_coro, timeout=0.1)
                    
                    future = asyncio.run_coroutine_threadsafe(timed_get_coro, self._loop)
                    audio_chunk = future.result() # Blocks this thread until get (with timeout) completes

                    if audio_chunk is None: 
                        print("AudioIOHandler: Playback loop: Received None sentinel, stopping.")
                        asyncio.run_coroutine_threadsafe(self.gemini_to_speaker_queue.task_done(), self._loop).result()
                        break
                    
                    self._output_stream.write(audio_chunk)
                    asyncio.run_coroutine_threadsafe(self.gemini_to_speaker_queue.task_done(), self._loop).result()

                except asyncio.TimeoutError: # From wait_for timeout
                    continue # Expected if queue is empty, allows checking _stop_event
                except Exception as e: 
                    if self._stop_event.is_set():
                        print("AudioIOHandler: Playback loop: Stop event received during Exception.")
                        if audio_chunk is not None and audio_chunk is not Ellipsis : # Ellipsis might be used internally by Queue
                             asyncio.run_coroutine_threadsafe(self.gemini_to_speaker_queue.task_done(), self._loop).result()
                        break
                    print(f"AudioIOHandler: Error in playback loop: {e}")
                    if audio_chunk is not None and audio_chunk is not Ellipsis:
                         asyncio.run_coroutine_threadsafe(self.gemini_to_speaker_queue.task_done(), self._loop).result()
                    break
        except Exception as e:
            print(f"AudioIOHandler: Failed to start or maintain speaker playback: {e}")
        finally:
            if self._output_stream:
                try:
                    if self._output_stream.is_active():
                        self._output_stream.stop_stream()
                    self._output_stream.close()
                except Exception as e_close:
                    print(f"AudioIOHandler: Error closing output stream: {e_close}")
                self._output_stream = None
            self._is_playback_active_flag = False
            print("AudioIOHandler: Speaker playback stream closed and loop finished.")

    async def play_audio(self):
        """
        Starts playing audio to the default speaker.
        Runs the PyAudio blocking calls in a separate thread.
        """
        if self._is_playback_active_flag: # Check internal flag
            print("AudioIOHandler: Playback is already in progress.")
            return
        
        # self._stop_event should be managed by the caller via start/stop calls.
        # It's cleared in record_audio, assuming record_audio is called first in a session.
        # If play_audio can be called independently, ensure _stop_event is clear if needed.
        # For this design, stop() sets it, and it's cleared when starting a new "session" (e.g. record_audio)
        print("AudioIOHandler: Starting audio playback task...")
        try:
            await asyncio.to_thread(self._blocking_play_loop)
        except Exception as e:
            print(f"AudioIOHandler: Exception from play_audio thread: {e}")
        finally:
            print("AudioIOHandler: Audio playback task awaited and finished.")

    async def stop(self):
        """
        Signals the recording and playback loops to stop and cleans up PyAudio resources.
        This method should be called after the tasks running record_audio and play_audio
        have been awaited or cancelled.
        """
        print("AudioIOHandler: Stop requested.")
        if not self._stop_event.is_set():
            self._stop_event.set()
            print("AudioIOHandler: Stop event set.")

        # Attempt to unblock the playback queue if it's waiting
        # This helps the _blocking_play_loop exit cleanly if it's stuck on queue.get()
        try:
            # Don't wait indefinitely, the loop might already be stopping
            await asyncio.wait_for(self.gemini_to_speaker_queue.put(None), timeout=0.5)
            print("AudioIOHandler: None sentinel put into playback queue to unblock.")
        except asyncio.QueueFull:
            print("AudioIOHandler: Playback queue full, couldn't put None sentinel. Loop might be stopping or full.")
        except asyncio.TimeoutError:
            print("AudioIOHandler: Timeout putting None sentinel to playback queue.")
        except Exception as e:
            print(f"AudioIOHandler: Error putting None sentinel to playback queue: {e}")
        
        # The threads running _blocking_record_loop and _blocking_play_loop
        # should detect _stop_event and terminate. The tasks that launched them
        # (created by FastAPI endpoint) must be awaited or cancelled to ensure
        # these threads actually finish before PyAudio is terminated.

        if self._pyaudio_instance:
            print("AudioIOHandler: Terminating PyAudio instance.")
            try:
                # Terminate can also be blocking, run in thread if issues arise
                # For now, direct call.
                await asyncio.to_thread(self._pyaudio_instance.terminate)
                self._pyaudio_instance = None
                print("AudioIOHandler: PyAudio instance terminated.")
            except Exception as e:
                print(f"AudioIOHandler: Error terminating PyAudio instance: {e}")
        
        print("AudioIOHandler: Stop processing completed.")

    def is_recording(self) -> bool:
        """Returns True if the recording loop is currently active."""
        return self._is_recording_active_flag

    def is_playing(self) -> bool:
        """Returns True if the playback loop is currently active."""
        return self._is_playback_active_flag