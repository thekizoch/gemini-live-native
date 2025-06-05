import asyncio
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.audio_io_handler import AudioIOHandler
from app.gemini_service import GeminiLiveService

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Gemini Live Native Audio Streamer")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Global state variables
audio_handler: AudioIOHandler | None = None
gemini_service: GeminiLiveService | None = None
session_tasks: list[asyncio.Task] = []
is_session_running: bool = False
session_lock = asyncio.Lock()  # Lock to prevent race conditions during start/stop operations

@app.on_event("startup")
async def startup_event():
    """
    Event handler for application startup.
    Checks for necessary configurations e.g. API keys.
    """
    print("FastAPI application starting up...")
    api_key = os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key:
        print("CRITICAL: GOOGLE_GENAI_API_KEY is not set in the environment. Session start will fail.")
    else:
        print("GOOGLE_GENAI_API_KEY found.")
    # Other initializations can go here if needed.
    # PyAudio is initialized on-demand by AudioIOHandler.

@app.on_event("shutdown")
async def shutdown_event():
    """
    Event handler for application shutdown.
    Ensures graceful termination of any running sessions.
    """
    print("FastAPI application shutting down...")
    if is_session_running or session_tasks: # Check if there's anything to stop
        print("Shutdown event: Session is active or tasks exist, attempting to stop...")
        # Use the lock to ensure atomicity if a stop operation could be concurrently triggered
        async with session_lock:
            await stop_session_internal(initiated_by="shutdown")

async def stop_session_internal(initiated_by: str = "endpoint"):
    """
    Internal helper function to stop the current audio streaming session.
    Manages stopping services, cancelling tasks, and cleaning up resources.
    """
    global audio_handler, gemini_service, session_tasks, is_session_running

    print(f"Stopping session internally (initiated by {initiated_by})...")
    
    # Early exit if no session is considered running and no tasks are present.
    if not is_session_running and not session_tasks:
        print("Stop internal: No active session or tasks to stop.")
        is_session_running = False # Ensure flag consistency
        return

    # Set the global running flag to false immediately to prevent new operations
    # and signal loops that rely on this flag (though dedicated stop events are preferred).
    is_session_running = False

    # 1. Signal services to stop their operations.
    # These methods should trigger internal stop events/flags within the services.
    if gemini_service:
        print("Signaling Gemini service to stop...")
        await gemini_service.stop_session() 
    
    if audio_handler:
        print("Signaling Audio I/O handler to stop...")
        await audio_handler.stop() # This also handles PyAudio termination.

    # 2. Cancel and await all top-level session tasks.
    # These tasks run the main methods of AudioIOHandler and GeminiLiveService.
    # Cancellation helps interrupt any await calls if services don't stop cleanly from signals.
    if session_tasks:
        print(f"Cancelling {len(session_tasks)} session tasks...")
        for task in session_tasks:
            if not task.done():
                task.cancel()
        
        print("Awaiting completion of cancelled tasks...")
        results = await asyncio.gather(*session_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            task_name = session_tasks[i].get_name() if hasattr(session_tasks[i], 'get_name') and session_tasks[i].get_name() else f"Task-{i}"
            if isinstance(result, asyncio.CancelledError):
                print(f"Task '{task_name}' was cancelled successfully.")
            elif isinstance(result, Exception):
                print(f"Task '{task_name}' raised an exception during/after cancellation: {type(result).__name__}('{result}')")
            else:
                print(f"Task '{task_name}' completed with result: {result}")
        session_tasks.clear()
        print("All session tasks processed.")

    # 3. Clear global service instances.
    audio_handler = None
    gemini_service = None
    
    print("Session stopped and resources presumed cleaned up.")


@app.post("/start-session", status_code=200)
async def start_session_endpoint():
    """
    Endpoint to start a new audio streaming session with Gemini.
    Initializes audio I/O, connects to Gemini, and starts streaming.
    """
    global audio_handler, gemini_service, session_tasks, is_session_running
    
    async with session_lock:
        if is_session_running:
            print("Start request failed: Session is already running.")
            raise HTTPException(status_code=400, detail="Session is already running.")

        print("Attempting to start new session...")
        try:
            # Verify API key presence before proceeding
            api_key = os.getenv("GOOGLE_GENAI_API_KEY")
            if not api_key:
                print("ERROR: GOOGLE_GENAI_API_KEY not found. Cannot start session.")
                raise HTTPException(status_code=500, detail="Server configuration error: GOOGLE_GENAI_API_KEY is not set.")

            # Create asyncio Queues for communication between audio handler and Gemini service
            mic_to_gemini_queue = asyncio.Queue(maxsize=100) # Maxsize for backpressure
            gemini_to_speaker_queue = asyncio.Queue(maxsize=100)

            # Initialize service components
            audio_handler = AudioIOHandler(
                mic_to_gemini_queue=mic_to_gemini_queue,
                gemini_to_speaker_queue=gemini_to_speaker_queue
            )
            gemini_service = GeminiLiveService(
                mic_to_gemini_queue=mic_to_gemini_queue,
                gemini_to_speaker_queue=gemini_to_speaker_queue
            )

            # Create and store asyncio tasks for each long-running operation
            task_record_audio = asyncio.create_task(audio_handler.record_audio(), name="record_audio_task")
            task_play_audio = asyncio.create_task(audio_handler.play_audio(), name="play_audio_task")
            task_gemini_session = asyncio.create_task(gemini_service.start_session(), name="gemini_session_task")
            
            session_tasks.extend([task_record_audio, task_play_audio, task_gemini_session])
            is_session_running = True # Set running state
            
            print("Session started successfully. Tasks are running.")
            return JSONResponse(content={"message": "Session started successfully. Speak into your microphone."})

        except HTTPException: # Re-raise HTTP exceptions
            raise
        except Exception as e:
            print(f"Error during session startup: {type(e).__name__} - {e}")
            # Attempt to clean up any partially initialized resources
            is_session_running = False # Ensure flag is reset
            current_tasks_to_clean = list(session_tasks) # Copy before clearing
            session_tasks.clear()

            if audio_handler: await audio_handler.stop()
            if gemini_service: await gemini_service.stop_session()
            
            for task in current_tasks_to_clean:
                if not task.done(): task.cancel()
            if current_tasks_to_clean:
                await asyncio.gather(*current_tasks_to_clean, return_exceptions=True)
            
            audio_handler = None
            gemini_service = None
            raise HTTPException(status_code=500, detail=f"Failed to start session due to an internal error: {str(e)}")


@app.post("/stop-session", status_code=200)
async def stop_session_endpoint():
    """
    Endpoint to stop the currently active audio streaming session.
    """
    async with session_lock:
        if not is_session_running and not session_tasks:
            print("Stop request: Session not running or no tasks to stop.")
            # Call internal stop to ensure consistency, even if already "stopped"
            await stop_session_internal(initiated_by="endpoint_already_stopped")
            return JSONResponse(content={"message": "Session was not running or already stopped."})
        
        print("Stop session endpoint called. Initiating stop procedure.")
        await stop_session_internal(initiated_by="endpoint_active_stop")
        return JSONResponse(content={"message": "Session stopped successfully."})

@app.get("/", response_class=FileResponse)
async def get_index_html():
    """
    Serves the main HTML page for the user interface.
    """
    # This path assumes 'static/index.html' exists relative to the project root
    # where uvicorn is typically run.
    static_file_path = "static/index.html"
    if not os.path.exists(static_file_path):
        # This fallback is less ideal; configuration should ensure correct path.
        print(f"Warning: '{static_file_path}' not found directly. Trying '../static/index.html'. Current PWD: {os.getcwd()}")
        static_file_path = "../static/index.html" # For cases where PWD might be 'app'
        if not os.path.exists(static_file_path):
             raise HTTPException(status_code=404, detail=f"Main UI file '{static_file_path}' not found.")
    return FileResponse(static_file_path)

# This block is for direct execution of this file (e.g., `python app/main.py`)
# However, using `uvicorn app.main:app --reload` or `uv run start` is preferred.
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server directly for development (use 'uv run start' for configured execution)...")
    # Ensure .env is loaded if running this way
    if not os.getenv("GOOGLE_GENAI_API_KEY"):
        print("WARNING: GOOGLE_GENAI_API_KEY not loaded. Direct run might fail if .env is not in search path.")
    
    # Uvicorn configuration should match the one in pyproject.toml for consistency.
    uvicorn.run(
        "app.main:app",  # Path to the FastAPI app instance
        host="0.0.0.0",
        port=8000,
        reload=True      # Enable auto-reload for development
    )