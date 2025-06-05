import asyncio
import os
import traceback # For more detailed exception logging if needed
from google import genai
from google.genai import types # For types.LiveConnectConfig

# Ensure GOOGLE_GENAI_API_KEY is available, genai.configure will use it.
# This is typically loaded by python-dotenv in main.py before this module is heavily used.
if not os.environ.get("GOOGLE_GENAI_API_KEY"):
    print("Warning: GOOGLE_GENAI_API_KEY not found in environment. GeminiLiveService may fail to initialize.")

class GeminiLiveService:
    """
    Manages the connection and audio streaming with the Gemini Live API.
    """

    MODEL_NAME = "gemini-2.5-flash-preview-native-audio-dialog"

    def __init__(self, mic_to_gemini_queue: asyncio.Queue, gemini_to_speaker_queue: asyncio.Queue):
        self.service_id = id(self) # For distinguishing logs if multiple instances exist
        print(f"GeminiService [{self.service_id}]: Initializing...")
        self.mic_to_gemini_queue = mic_to_gemini_queue
        self.gemini_to_speaker_queue = gemini_to_speaker_queue
        
        self.api_client: genai.Client | None = None 
        self.session: genai.live.AsyncLiveSession | None = None 
        
        self._stop_event = asyncio.Event()
        self._is_session_active_flag = False
        print(f"GeminiService [{self.service_id}]: Initialized. _stop_event initially: {self._stop_event.is_set()}")

    async def _send_audio_stream(self):
        print(f"GeminiService [{self.service_id} SEND]: Starting audio send stream. _stop_event: {self._stop_event.is_set()}")
        sent_chunks = 0
        try:
            while not self._stop_event.is_set():
                if not self.session:
                    print(f"GeminiService [{self.service_id} SEND]: No session, stopping send stream. _stop_event: {self._stop_event.is_set()}")
                    if not self._stop_event.is_set(): self._stop_event.set(); print(f"GeminiService [{self.service_id} SEND]: Set _stop_event (no session).")
                    break
                
                audio_chunk = None
                try:
                    audio_chunk = await asyncio.wait_for(self.mic_to_gemini_queue.get(), timeout=0.1)
                    if audio_chunk is None:
                        print(f"GeminiService [{self.service_id} SEND]: Received None sentinel from mic_queue, stopping. _stop_event: {self._stop_event.is_set()}")
                        self.mic_to_gemini_queue.task_done()
                        if not self._stop_event.is_set(): self._stop_event.set(); print(f"GeminiService [{self.service_id} SEND]: Set _stop_event (None from mic_queue).")
                        break
                    
                    if self._stop_event.is_set():
                        print(f"GeminiService [{self.service_id} SEND]: _stop_event set after queue.get(), stopping before send.")
                        if audio_chunk is not None: self.mic_to_gemini_queue.task_done()
                        break

                    await self.session.send(input={"data": audio_chunk, "mime_type": "audio/pcm"})
                    sent_chunks += 1
                    self.mic_to_gemini_queue.task_done()

                except asyncio.TimeoutError:
                    continue
                # Removed specific catch for types.StopCandidateException as it caused AttributeError
                # and the example doesn't explicitly catch it in its send loop.
                # General exceptions will be caught below.
                except Exception as e:
                    print(f"GeminiService [{self.service_id} SEND]: Error in send loop: {type(e).__name__} - {e}. Session: {self.session is not None}. _stop_event: {self._stop_event.is_set()}")
                    if audio_chunk is not None: self.mic_to_gemini_queue.task_done()
                    if not self._stop_event.is_set(): self._stop_event.set(); print(f"GeminiService [{self.service_id} SEND]: Set _stop_event (Exception in loop).")
                    break
        except asyncio.CancelledError:
            print(f"GeminiService [{self.service_id} SEND]: Send audio stream task cancelled. _stop_event: {self._stop_event.is_set()}")
        finally:
            print(f"GeminiService [{self.service_id} SEND]: Audio send stream finished. Total chunks sent: {sent_chunks}. _stop_event: {self._stop_event.is_set()}")

    async def _receive_audio_stream(self):
        print(f"GeminiService [{self.service_id} RECV]: Starting audio receive stream. _stop_event: {self._stop_event.is_set()}")
        received_responses = 0
        put_to_speaker_queue_count = 0
        try:
            outer_loop_count = 0
            while not self._stop_event.is_set():
                outer_loop_count += 1
                print(f"GeminiService [{self.service_id} RECV]: Outer loop iter {outer_loop_count}. Session: {self.session is not None}. _stop_event: {self._stop_event.is_set()}")
                if not self.session:
                    print(f"GeminiService [{self.service_id} RECV]: No session, stopping receive stream. _stop_event: {self._stop_event.is_set()}")
                    if not self._stop_event.is_set(): self._stop_event.set(); print(f"GeminiService [{self.service_id} RECV]: Set _stop_event (no session).")
                    break
                
                try:
                    print(f"GeminiService [{self.service_id} RECV]: Attempting 'async for response in self.session.receive()'. Session type: {type(self.session)}. _stop_event: {self._stop_event.is_set()}")
                    entered_async_for = False
                    async for response in self.session.receive():
                        entered_async_for = True
                        received_responses += 1

                        if self._stop_event.is_set():
                            print(f"GeminiService [{self.service_id} RECV]: _stop_event set during response iteration, breaking from response loop.")
                            break 
                        
                        if response.error:
                            print(f"GeminiService [{self.service_id} RECV]: Response error: {response.error}. Stopping. _stop_event: {self._stop_event.is_set()}")
                            if not self._stop_event.is_set(): self._stop_event.set(); print(f"GeminiService [{self.service_id} RECV]: Set _stop_event (response error).")
                            break 

                        if response.data:
                            await self.gemini_to_speaker_queue.put(response.data)
                            put_to_speaker_queue_count +=1
                        
                        if response.results:
                            for res_idx, result in enumerate(response.results):
                                if hasattr(result, 'text') and result.text:
                                     print(f"GeminiService [{self.service_id} RECV]: (Transcript/Result {res_idx}): {result.text}")
                    
                    if not entered_async_for:
                        print(f"GeminiService [{self.service_id} RECV]: 'async for' loop for self.session.receive() was NOT entered. _stop_event: {self._stop_event.is_set()}")
                    else:
                        print(f"GeminiService [{self.service_id} RECV]: 'async for' loop for self.session.receive() completed. Total responses iterated: {received_responses}. _stop_event: {self._stop_event.is_set()}")
                    
                    print(f"GeminiService [{self.service_id} RECV]: session.receive() iterator finished or response error. Stopping receive stream. _stop_event: {self._stop_event.is_set()}")
                    if not self._stop_event.is_set(): self._stop_event.set(); print(f"GeminiService [{self.service_id} RECV]: Set _stop_event (after async for).")
                    break 

                # Removed specific catch for types.StopCandidateException as it caused AttributeError
                # and the example doesn't explicitly catch it.
                except asyncio.CancelledError: # Still catch CancelledError to re-raise if needed
                    print(f"GeminiService [{self.service_id} RECV]: Task directly cancelled (inner try). _stop_event: {self._stop_event.is_set()}")
                    raise
                except Exception as e: # Catch other exceptions that might occur during receive
                    print(f"GeminiService [{self.service_id} RECV]: Exception in inner try (receive loop): {type(e).__name__} - {e}. Session: {self.session is not None}. _stop_event: {self._stop_event.is_set()}")
                    if not self._stop_event.is_set(): self._stop_event.set(); print(f"GeminiService [{self.service_id} RECV]: Set _stop_event (inner Exception).")
                    break
            
            print(f"GeminiService [{self.service_id} RECV]: Exited outer loop. _stop_event: {self._stop_event.is_set()}")

        except asyncio.CancelledError:
             print(f"GeminiService [{self.service_id} RECV]: Receive audio stream task cancelled by gather. _stop_event: {self._stop_event.is_set()}")
        finally:
            print(f"GeminiService [{self.service_id} RECV]: FINALLY block. Total responses: {received_responses}, Items put to speaker_queue: {put_to_speaker_queue_count}. _stop_event: {self._stop_event.is_set()}.")
            item_to_put = None
            print(f"GeminiService [{self.service_id} RECV]: FINALLY: Putting {type(item_to_put)} (sentinel) into gemini_to_speaker_queue.")
            try:
                await asyncio.wait_for(self.gemini_to_speaker_queue.put(item_to_put), timeout=1.0)
                print(f"GeminiService [{self.service_id} RECV]: FINALLY: Placed None sentinel in gemini_to_speaker_queue.")
            except Exception as e:
                print(f"GeminiService [{self.service_id} RECV]: FINALLY: Error putting None to speaker_queue: {type(e).__name__} - {e}")
            print(f"GeminiService [{self.service_id} RECV]: Audio receive stream finished. _stop_event: {self._stop_event.is_set()}")


    async def start_session(self):
        print(f"GeminiService [{self.service_id} MAIN]: Attempting to start session. Current _is_session_active_flag: {self._is_session_active_flag}, _stop_event: {self._stop_event.is_set()}")
        if self._is_session_active_flag:
            print(f"GeminiService [{self.service_id} MAIN]: Session is already considered active.")
            return

        self._stop_event.clear(); print(f"GeminiService [{self.service_id} MAIN]: Cleared _stop_event. Now: {self._stop_event.is_set()}")
        self._is_session_active_flag = False 
        
        try:
            api_key = os.getenv("GOOGLE_GENAI_API_KEY")
            if not api_key:
                 msg = f"GeminiService [{self.service_id} MAIN]: CRITICAL - GOOGLE_GENAI_API_KEY not found. Cannot start."
                 print(msg)
                 raise ValueError(msg)
            
            self.api_client = genai.Client(api_key=api_key)
            print(f"GeminiService [{self.service_id} MAIN]: API Client created.")

            live_config = types.LiveConnectConfig(
                response_modalities=["AUDIO"]
            )
            
            print(f"GeminiService [{self.service_id} MAIN]: Connecting to model {self.MODEL_NAME} using 'async with'...")
            async with self.api_client.aio.live.connect(
                model=self.MODEL_NAME,
                config=live_config
            ) as live_api_session:
                self.session = live_api_session
                print(f"GeminiService [{self.service_id} MAIN]: Session object obtained: {type(self.session)}. Session id: {id(self.session)}")
                self._is_session_active_flag = True
                print(f"GeminiService [{self.service_id} MAIN]: Session connected. _is_session_active_flag: {self._is_session_active_flag}. Starting send/receive streams. _stop_event: {self._stop_event.is_set()}")

                results = await asyncio.gather(
                    self._send_audio_stream(),
                    self._receive_audio_stream(),
                    return_exceptions=True
                )
                
                print(f"GeminiService [{self.service_id} MAIN]: Send/receive streams gather call completed. Results: {results}. _stop_event: {self._stop_event.is_set()}")
                for i, res_item in enumerate(results):
                    task_name = "_send_audio_stream" if i == 0 else "_receive_audio_stream"
                    if isinstance(res_item, Exception) and not isinstance(res_item, asyncio.CancelledError):
                        print(f"GeminiService [{self.service_id} MAIN]: Exception in gathered task '{task_name}': {type(res_item).__name__} - {res_item}")
                        if not self._stop_event.is_set(): self._stop_event.set(); print(f"GeminiService [{self.service_id} MAIN]: Set _stop_event (gather exception in {task_name}).")
                    elif isinstance(res_item, asyncio.CancelledError):
                         print(f"GeminiService [{self.service_id} MAIN]: Gathered task '{task_name}' was CancelledError: {res_item}")
                    else:
                        print(f"GeminiService [{self.service_id} MAIN]: Gathered task '{task_name}' completed with result: {type(res_item)} (value: {res_item}).")

            print(f"GeminiService [{self.service_id} MAIN]: 'async with' block exited, Live API session closed. _stop_event: {self._stop_event.is_set()}")

        except ValueError as ve: 
            print(f"GeminiService [{self.service_id} MAIN]: Configuration error: {ve}")
        except Exception as e:
            print(f"GeminiService [{self.service_id} MAIN]: Error during session management: {type(e).__name__} - {e}. _stop_event: {self._stop_event.is_set()}")
        finally:
            print(f"GeminiService [{self.service_id} MAIN]: start_session FINALLY block. _is_session_active_flag: {self._is_session_active_flag}, _stop_event: {self._stop_event.is_set()}")
            if not self._stop_event.is_set():
                print(f"GeminiService [{self.service_id} MAIN]: FINALLY: _stop_event was false, setting true.")
                self._stop_event.set()
            else:
                print(f"GeminiService [{self.service_id} MAIN]: FINALLY: _stop_event was already true.")

            self.session = None 
            self._is_session_active_flag = False
            print(f"GeminiService [{self.service_id} MAIN]: Session fully cleaned up and state reset. _is_session_active_flag: {self._is_session_active_flag}, _stop_event: {self._stop_event.is_set()}")

    async def stop_session(self):
        print(f"GeminiService [{self.service_id} MAIN]: Stop session requested. Current _stop_event: {self._stop_event.is_set()}, _is_session_active_flag: {self._is_session_active_flag}")
        
        if not self._stop_event.is_set():
            self._stop_event.set()
            print(f"GeminiService [{self.service_id} MAIN]: Stop event has been set by stop_session(). Now: {self._stop_event.is_set()}")

            try:
                print(f"GeminiService [{self.service_id} MAIN]: Attempting to put None sentinel in mic_to_gemini_queue for sender.")
                self.mic_to_gemini_queue.put_nowait(None) 
                print(f"GeminiService [{self.service_id} MAIN]: Placed None sentinel in mic_to_gemini_queue for sender.")
            except asyncio.QueueFull:
                print(f"GeminiService [{self.service_id} MAIN]: mic_to_gemini_queue full, couldn't place None sentinel.")
            except Exception as e: 
                print(f"GeminiService [{self.service_id} MAIN]: Error putting sentinel to mic_to_gemini_queue: {type(e).__name__} - {e}")
        else:
            print(f"GeminiService [{self.service_id} MAIN]: Stop event was already set.")
        
        print(f"GeminiService [{self.service_id} MAIN]: Stop session request processed. Running tasks should now terminate. _stop_event: {self._stop_event.is_set()}")

    def is_active(self) -> bool:
        active = self._is_session_active_flag and not self._stop_event.is_set()
        return active

