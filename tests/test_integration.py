import pytest
import pytest_asyncio
import asyncio
import tonic_ollama_client as toc

SUPPORTED_MODELS_LIST = [
    "llama3.1:latest",
    "phi4:latest",
    "qwen2:7b",
]

@pytest_asyncio.fixture(scope="session")
async def live_client_session():
    """Provides a session-scoped TonicOllamaClient for integration tests."""
    client = toc.create_client(debug=True, max_readiness_attempts=3)
    yield client

class BaseSpecificModelTests:
    """Base class for integration tests targeting a specific model."""

    async def _ensure_model_ready(self, client: toc.TonicOllamaClient, model_to_check: str):
        """Helper to check model readiness."""
        if not model_to_check:
            pytest.skip("MODEL_NAME not available for test")
        try:
            await client.check_model_ready(model_to_check)
        except toc.ModelNotReadyError as e:
            pytest.fail(f"Model {model_to_check} could not be made ready for test: {e}")

    async def test_model_is_explicitly_ready(self, live_client_session: toc.TonicOllamaClient, MODEL_NAME: str):
        """Tests check_model_ready functionality."""
        await self._ensure_model_ready(live_client_session, MODEL_NAME)

    async def test_live_chat_specific(self, live_client_session: toc.TonicOllamaClient, MODEL_NAME: str):
        """Tests live chat functionality."""
        await self._ensure_model_ready(live_client_session, MODEL_NAME)
        response = await live_client_session.chat(
            model=MODEL_NAME,
            message="What is the capital of France? Respond with only the city name.",
            temperature=0.1
        )
        
        assert "message" in response
        assert "content" in response["message"]
        assert isinstance(response["message"]["content"], str)
        content_lower = response["message"]["content"].lower()
        assert len(content_lower) > 0
        assert "paris" in content_lower or ("france" in content_lower and len(content_lower) < 100) or len(content_lower) < 20

    async def test_live_embedding_specific(self, live_client_session: toc.TonicOllamaClient, MODEL_NAME: str):
        """Tests live embedding generation."""
        await self._ensure_model_ready(live_client_session, MODEL_NAME)
        embedding = await live_client_session.generate_embedding(
            model=MODEL_NAME,
            text="This is a test for embeddings."
        )
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

@pytest.mark.integration
@pytest.mark.parametrize("MODEL_NAME", SUPPORTED_MODELS_LIST)
class TestModelIntegration(BaseSpecificModelTests):
    """Parameterized integration tests for various Ollama models."""

    async def test_model_removal_and_pulling(self, live_client_session: toc.TonicOllamaClient, MODEL_NAME: str):
        """Tests removing and then re-pulling the specified model."""
        client = live_client_session
        current_model_name = MODEL_NAME 
        
        try:
            await client.check_model_ready(current_model_name)
        except toc.ModelNotReadyError as e:
            pytest.fail(f"Model {current_model_name} could not be made ready for pre-test setup: {e}")

        ollama_client = client.get_async_client()
        
        try:
            toc.fancy_print(client.console, f"Attempting to remove model {current_model_name} for testing...", style="yellow")
            await ollama_client.delete(model=current_model_name)
            toc.fancy_print(client.console, f"Model {current_model_name} removed successfully for testing.", style="cyan")
        except Exception as e: # Catching generic ollama.ResponseError if model not found, or other issues
            toc.fancy_print(client.console, f"Note: Could not remove model {current_model_name} (may already be gone or in use): {e}", style="yellow")

        await asyncio.sleep(2)
        
        toc.fancy_print(client.console, f"Attempting to re-validate/pull model {current_model_name} after deletion attempt...", style="yellow")
        try:
            await client.check_model_ready(current_model_name) # This will pull if not found
        except toc.ModelNotReadyError as e:
            pytest.fail(f"Model {current_model_name} could not be re-pulled and made ready: {e}")
        
        models_after = await ollama_client.list()
        model_names_after = [m.get('name') for m in models_after.get('models', [])]
        assert current_model_name in model_names_after, f"Model {current_model_name} not found after re-pull attempt."
        
        response = await client.chat(
            model=current_model_name,
            message="Hello! This is a test after re-pulling the model.",
            temperature=0.1
        )
        assert response.get("message", {}).get("content"), f"Chat with {current_model_name} after re-pull failed."
