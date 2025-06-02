import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from rich.console import Console
from tonic_ollama_client import (
    TonicOllamaClient,
    create_client,
    OllamaServerNotRunningError,
)

APPROVED_MODELS = ["llama3.1:latest", "phi4:latest", "qwen3:8b", "mistral:latest"]
DEFAULT_TEST_MODEL = "llama3.1:latest"


class TestTonicOllamaClientMethods:
    """Tests for TonicOllamaClient class methods."""

    @pytest_asyncio.fixture
    async def client_instance(self):
        """Provides a TonicOllamaClient instance with mocked AsyncClient."""
        with patch('tonic_ollama_client.AsyncClient') as MockedOllamaAsyncClient:
            mock_ollama_instance = MockedOllamaAsyncClient.return_value
            mock_ollama_instance.chat = AsyncMock()
            mock_ollama_instance.embeddings = AsyncMock()
            # .list and .pull are not directly used by TonicOllamaClient's core logic anymore
            # mock_ollama_instance.list = AsyncMock() 
            # mock_ollama_instance.pull = AsyncMock()
            
            client = TonicOllamaClient(debug=False)
            # Store the mock for assertion if needed, though get_async_client returns it
            yield client, mock_ollama_instance

    def test_initialization_defaults(self):
        client = TonicOllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.max_server_startup_attempts == 3  # Changed attribute
        assert client.debug is False
        assert isinstance(client.console, Console)
        assert client.conversations == {}

    def test_initialization_custom(self):
        custom_console = Console()
        client = TonicOllamaClient(
            base_url="http://test-url:1234",
            max_server_startup_attempts=5,  # Changed parameter and attribute
            debug=True,
            console=custom_console
        )
        assert client.base_url == "http://test-url:1234"
        assert client.max_server_startup_attempts == 5  # Changed attribute
        assert client.debug is True
        assert client.console == custom_console

    def test_get_async_client_creates_once(self, client_instance):
        client, mock_ollama_instance_from_fixture = client_instance
        first_async_client = client.get_async_client()
        # Assert it's the mock instance we expect from the patch
        assert first_async_client is mock_ollama_instance_from_fixture
        second_async_client = client.get_async_client()
        assert first_async_client is second_async_client

    @patch('socket.create_connection', return_value=MagicMock())
    def test_is_ollama_server_running_sync_true(self, mock_socket_conn, client_instance): # Renamed test
        client, _ = client_instance
        assert client._is_ollama_server_running_sync() is True # Use _sync version
        mock_socket_conn.assert_called_once()

    @patch('socket.create_connection', side_effect=ConnectionRefusedError)
    def test_is_ollama_server_running_sync_false(self, mock_socket_conn, client_instance): # Renamed test
        client, _ = client_instance
        assert client._is_ollama_server_running_sync() is False # Use _sync version
        mock_socket_conn.assert_called_once()

    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running_sync', return_value=True) # Patch _sync version
    async def test_ensure_server_ready_server_responsive(self, mock_is_server_running_sync, client_instance): # Renamed and refactored
        client, _ = client_instance # mock_ollama not needed here
        
        await client.ensure_server_ready() # No model_name argument
        mock_is_server_running_sync.assert_called_once()
        # Assertions for mock_ollama.list, chat, pull are removed as ensure_server_ready doesn't do this

    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running_sync', side_effect=[False, False, True]) # Patch _sync version
    async def test_ensure_server_ready_becomes_responsive_after_retries(self, mock_is_server_running_sync, client_instance): # New test
        client, _ = client_instance
        client.max_server_startup_attempts = 3
        
        await client.ensure_server_ready()
        assert mock_is_server_running_sync.call_count == 3


    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running_sync', return_value=False) # Patch _sync version
    async def test_ensure_server_ready_server_not_running_raises_error(self, mock_is_server_running_sync, client_instance): # Renamed and refactored
        client, _ = client_instance
        client.max_server_startup_attempts = 2 # For faster test
        with pytest.raises(OllamaServerNotRunningError):
            await client.ensure_server_ready() # No model_name argument
        assert mock_is_server_running_sync.call_count == 2

    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running_sync', return_value=False) # Patch _sync version
    async def test_chat_server_not_running_via_ensure_ready(self, mock_is_server_running_sync, client_instance): # Renamed and refactored
        client, _ = client_instance # mock_ollama not needed for this specific failure path
        client.max_server_startup_attempts = 1 # Ensure it fails quickly
        
        # Reset the mock to clear any previous calls
        mock_is_server_running_sync.reset_mock()
        
        with pytest.raises(OllamaServerNotRunningError):
            await client.chat(model=DEFAULT_TEST_MODEL, message="Hello")
        
        # Verify it was called at least once - we don't care about exact count
        # due to potential retries from both ensure_server_ready and API_RETRY_CONFIG
        assert mock_is_server_running_sync.call_count > 0, "Server check should be called at least once"

    @pytest.mark.asyncio
    @patch('tonic_ollama_client.TonicOllamaClient._is_ollama_server_running_sync', return_value=False) # Patch _sync version
    async def test_generate_embedding_server_not_running_via_ensure_ready(self, mock_is_server_running_sync, client_instance): # Renamed and refactored
        client, _ = client_instance # mock_ollama not needed for this specific failure path
        client.max_server_startup_attempts = 1 # Ensure it fails quickly
        
        # Reset the mock to clear any previous calls
        mock_is_server_running_sync.reset_mock()
        
        with pytest.raises(OllamaServerNotRunningError):
            await client.generate_embedding(model=DEFAULT_TEST_MODEL, text="Embedding test")
        
        # Verify it was called at least once - we don't care about exact count
        # due to potential retries from both ensure_server_ready and API_RETRY_CONFIG
        assert mock_is_server_running_sync.call_count > 0, "Server check should be called at least once"

    @pytest.mark.asyncio
    async def test_conversation_management(self, client_instance):
        client, _ = client_instance
        conv_id = await client.create_conversation("test-conv")
        assert "test-conv" in client.list_conversations()
        
        client.conversations[conv_id].append({"role": "user", "content": "message1"})
        messages = client.get_conversation(conv_id)
        assert len(messages) == 1
        
        client.clear_conversation(conv_id)
        assert len(client.get_conversation(conv_id)) == 0
        
        client.delete_conversation(conv_id)
        assert "test-conv" not in client.list_conversations()
        with pytest.raises(ValueError):
            client.get_conversation(conv_id)

    @pytest.mark.asyncio
    async def test_client_close_method(self, client_instance):
        """Test the client's close method (default: unload all configured models)."""
        client, mock_ollama_instance = client_instance
        
        client.get_async_client()

        mock_ollama_instance._client = AsyncMock() 
        mock_ollama_instance._client.aclose = AsyncMock()
        mock_ollama_instance.generate = AsyncMock()

        # client.models_to_unload_on_close is already set by default or by constructor
        # For this test, we rely on the default DEFAULT_MODELS_TO_UNLOAD_ON_CLOSE
        from tonic_ollama_client import DEFAULT_MODELS_TO_UNLOAD_ON_CLOSE
        
        await client.close() # Call without arguments

        assert mock_ollama_instance.generate.call_count == len(DEFAULT_MODELS_TO_UNLOAD_ON_CLOSE)
        for model_name in DEFAULT_MODELS_TO_UNLOAD_ON_CLOSE:
            mock_ollama_instance.generate.assert_any_call(
                model=model_name,
                prompt=".",
                options={"num_predict": 1},
                keep_alive="0s"
            )
        
        mock_ollama_instance._client.aclose.assert_called_once()
        assert client.async_client is None

    @pytest.mark.asyncio
    async def test_client_close_method_specific_model(self, client_instance):
        """Test closing the client and unloading a specific model."""
        client, mock_ollama_instance = client_instance
        client.get_async_client() # Initialize async_client

        mock_ollama_instance._client = AsyncMock()
        mock_ollama_instance._client.aclose = AsyncMock()
        mock_ollama_instance.generate = AsyncMock()

        specific_model_to_unload = "phi4:latest"
        await client.close(model_to_unload=specific_model_to_unload)

        # Verify generate was called only for the specific model
        assert mock_ollama_instance.generate.call_count == 1
        mock_ollama_instance.generate.assert_called_once_with(
            model=specific_model_to_unload,
            prompt=".",
            options={"num_predict": 1},
            keep_alive="0s"
        )
        
        # Verify the underlying httpx client's aclose was called
        mock_ollama_instance._client.aclose.assert_called_once()
        assert client.async_client is None


    @pytest.mark.asyncio
    async def test_client_close_method_no_async_client_initialized(self):
        """Test close method when async_client was never initialized."""
        client = TonicOllamaClient(debug=False)
        # No spy needed for ollama.AsyncClient methods as it shouldn't be created
        
        await client.close() # Should run without error
        
        # Assert that async_client remains None
        assert client.async_client is None
        # No API calls should have been attempted.

    @pytest.mark.asyncio
    async def test_client_close_method_custom_models_to_unload(self, client_instance):
        """Test close method with a custom list of models to unload (when called with no args)."""
        custom_models = ["custom_model1:latest", "custom_model2:latest"]
        
        with patch('tonic_ollama_client.AsyncClient') as MockedOllamaAsyncClient:
            mock_ollama_instance = MockedOllamaAsyncClient.return_value
            mock_ollama_instance.generate = AsyncMock()
            mock_ollama_instance._client = AsyncMock() # Mock the internal httpx client
            mock_ollama_instance._client.aclose = AsyncMock()

            client = TonicOllamaClient(debug=False, models_to_unload_on_close=custom_models)
            client.get_async_client()

            await client.close() # Call without arguments to use the custom list

            assert mock_ollama_instance.generate.call_count == len(custom_models)
            for model_name in custom_models:
                mock_ollama_instance.generate.assert_any_call(
                    model=model_name,
                    prompt=".",
                    options={"num_predict": 1},
                    keep_alive="0s"
                )
            mock_ollama_instance._client.aclose.assert_called_once()
            assert client.async_client is None


def test_create_client_default():
    client = create_client()
    assert isinstance(client, TonicOllamaClient)
    assert client.debug is False
    from tonic_ollama_client import DEFAULT_MODELS_TO_UNLOAD_ON_CLOSE
    assert client.models_to_unload_on_close == DEFAULT_MODELS_TO_UNLOAD_ON_CLOSE

def test_create_client_custom():
    console = Console()
    custom_unload_list = ["test_model:v1"]
    client = create_client(
        base_url="http://custom:1111", 
        debug=True, 
        console=console,
        models_to_unload_on_close=custom_unload_list
    )
    assert client.base_url == "http://custom:1111"
    assert client.debug is True
    assert client.console is console
    assert client.models_to_unload_on_close == custom_unload_list