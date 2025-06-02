import pytest
import pytest_asyncio
import asyncio
import tonic_ollama_client as toc
from typing import Dict, Any, cast

SUPPORTED_MODELS_LIST = [
    "llama3.1:latest",
    "phi4:latest",
    "qwen3:8b",
    "mistral:latest",
]

@pytest_asyncio.fixture(scope="function")
async def live_client_session():
    """Provides a function-scoped TonicOllamaClient for integration tests."""
    # Use the new attribute name for attempts if it was intended for server readiness
    client = toc.create_client(debug=True, max_server_startup_attempts=3) 
    yield client

# Base class containing test logic
class BaseModelTests:
    """Base class for model integration tests."""

    async def _ensure_server_and_model_available(self, client: toc.TonicOllamaClient, model_to_check: str):
        """Helper to check server readiness and model availability (via simple chat)."""
        if not model_to_check:
            pytest.skip("MODEL_NAME not available for test")
        try:
            await client.ensure_server_ready()
            try:
                await client.get_async_client().show(model=model_to_check)
            except toc.ResponseError as e:
                if e.status_code == 404: # Model not found
                    pytest.skip(f"Model {model_to_check} not found on Ollama server. Please pull it first. Error: {e}")
                else: # Other API error
                    pytest.fail(f"API error checking model {model_to_check} availability: {e}")

        except toc.OllamaServerNotRunningError:
            pytest.skip("Ollama server not running. Start it with 'ollama serve'")
        except Exception as e:
            pytest.fail(f"Unexpected error during server/model readiness check for {model_to_check}: {e}")


    async def test_server_is_ready_for_model(self, live_client_session: toc.TonicOllamaClient, MODEL_NAME: str):
        """Tests ensure_server_ready and basic model availability."""
        await self._ensure_server_and_model_available(live_client_session, MODEL_NAME)

    async def test_live_chat_specific(self, live_client_session: toc.TonicOllamaClient, MODEL_NAME: str):
        """Tests live chat functionality."""
        await self._ensure_server_and_model_available(live_client_session, MODEL_NAME)
        
        # Explicitly specify we want a non-streaming response with type annotation
        response: Dict[str, Any] = await live_client_session.chat(
            model=MODEL_NAME,
            message="What is the capital of France? Respond with only the city name.",
            temperature=0.1,
            stream=False  # Explicitly non-streaming
        )
        
        # Now we can safely use dictionary access
        assert "message" in response
        assert "content" in response["message"]
        assert isinstance(response["message"]["content"], str)
        content_lower = response["message"]["content"].lower()
        assert len(content_lower) > 0
        assert "paris" in content_lower or ("france" in content_lower and len(content_lower) < 100) or len(content_lower) < 20

    async def test_live_embedding_specific(self, live_client_session: toc.TonicOllamaClient, MODEL_NAME: str):
        """Tests live embedding generation."""
        await self._ensure_server_and_model_available(live_client_session, MODEL_NAME)
        embedding = await live_client_session.generate_embedding(
            model=MODEL_NAME,
            text="This is a test for embeddings."
        )
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

@pytest.mark.integration
@pytest.mark.parametrize("MODEL_NAME", SUPPORTED_MODELS_LIST)
class TestModelIntegration(BaseModelTests):
    """Parameterized integration tests for various Ollama models."""
