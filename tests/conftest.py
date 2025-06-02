import pytest
import httpx
import time

# Define the models to be cleaned up.
# This list should ideally be centralized if used in multiple places.
APPROVED_MODELS_TO_CLEANUP = ["llama3.1:latest", "phi4:latest", "qwen3:8b"]
OLLAMA_BASE_URL = "http://localhost:11434" # Or from a config

@pytest.fixture(scope="session", autouse=True)
def cleanup_ollama_models_session(request):
    """
    Register a finalizer to unload Ollama models after all tests have run.
    Uses httpx directly instead of AsyncClient to avoid async/event loop issues.
    """

    # This function runs at the end of the session
    def cleanup_models():
        print("\n\nINFO: Attempting to unload Ollama models post-test session...")
        
        for model_name in APPROVED_MODELS_TO_CLEANUP:
            try:
                # Send request with keep_alive="0s" to unload the model using direct httpx
                print(f"  INFO: Sending request to unload model: {model_name} (keep_alive='0s')")
                
                # Make a minimal request with HTTP directly
                with httpx.Client(timeout=10.0) as client:
                    response = client.post(
                        f"{OLLAMA_BASE_URL}/api/generate",
                        json={
                            "model": model_name,
                            "prompt": ".",
                            "options": {"num_predict": 1},
                            "keep_alive": "0s"
                        }
                    )
                    
                    if response.status_code == 200:
                        print(f"    INFO: Unload request sent for {model_name}.")
                    elif response.status_code == 404:
                        print(f"    WARN: Model {model_name} not found or not loaded.")
                    else:
                        print(f"    WARN: API error during unload attempt for model {model_name}: {response.status_code} - {response.text}")
                
                # Wait a moment to allow the server to process the unload
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ERROR: Unexpected error during unload attempt for model {model_name}: {e}")
        
        print("INFO: Ollama model unload attempt finished.")

    # Register the cleanup function to run at the end of the session
    request.addfinalizer(cleanup_models)
    
    # No yield needed since this is not a generator fixture
    return

